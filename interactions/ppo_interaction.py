import torch as torch
import gymnasium as gym
from agents.ppo_agents import PPOActorNetwork, PPOCriticNetwork
from tqdm import tqdm
from configs.agent_configs.a_ppo_agents import critic_configs, actor_configs
import torch.distributions as dist
from replay_buffer import PPOReplayBuffer
import numpy as np
from collections import deque
import random
from utils import get_environment, get_render_mode
from torch.distributions import Categorical


class PPO_interaction:
    def __init__(self, interaction_config, actor_configs, critic_configs):
        # load parallel TexasHoldem or LunarLander, etc.
        self.env      = get_environment(interaction_config, train=True)
        self.test_env = get_environment(interaction_config, train=False)

        self.env.reset()
        self.agents = list(self.env.possible_agents)

        # Only these are the “real” players
        self.player_agents = [
            a for a in self.env.possible_agents
            if a.startswith("player")
        ]
        first_obs, _, _, _, _ = self.env.last()
        first_agent = self.env.agent_selection

        obs_vec = first_obs["observation"]
        self.obs_size = obs_vec.shape[0]

        act_space = self.env.action_space(first_agent)
        if hasattr(act_space, "n"):
            self.action_size = act_space.n
        elif hasattr(act_space, "nvec"):
            self.action_size = int(act_space.nvec[0])
        else:
            raise ValueError("Unsupported action space")


        self.num_episodes = interaction_config.training_episodes
        self.testing_episodes = interaction_config.testing_episodes
        self.update_frequency = interaction_config.update_frequency
        self.critic = PPOCriticNetwork(obs_size = self.obs_size,
                                 action_size=self.action_size,
                                 critic_config=critic_configs)
        self.policy = PPOActorNetwork(obs_size = self.obs_size,
                                 action_size=self.action_size,
                                 actor_config=actor_configs)
        self.old_policy = PPOActorNetwork(obs_size = self.obs_size,
                                 action_size=self.action_size,
                                 actor_config=actor_configs)
        self.old_policy.load_state_dict(self.policy.state_dict())

        # Initialize hyperparemeters
        self.c = interaction_config.c # clipping factor (0 <= x <= 1) for PPO
        self.batch_size = interaction_config.batch_size
        self.k = interaction_config.k # number of training epochs

        # Initialize memory
        self.memory = PPOReplayBuffer(capacity=interaction_config.capacity,
                                   batch_size=self.batch_size)

    def act(self, state):
        with torch.no_grad():
            action_probs = self.old_policy.model(state)
        distribution = dist.Categorical(action_probs)
        action = distribution.sample()
        logprob = distribution.log_prob(action)
        return action.item(), logprob.item()

    def act_multiagent(self, state_vec: np.ndarray, action_mask: np.ndarray):
        with torch.no_grad():
            # old_policy.model returns probabilities [batch=1, action_size]
            probs = self.old_policy.model(torch.Tensor(state_vec).unsqueeze(0))
            probs = probs.squeeze(0)  # -> [action_size]

        mask = torch.Tensor(action_mask)  # float tensor of 0/1
        masked = probs * mask             # zero out illegal
        total = masked.sum().item()

        if total == 0.0:
            # fallback: uniform over legal moves
            legal_count = mask.sum().item()
            masked = mask / legal_count
        else:
            masked = masked / total

        dist = Categorical(masked)
        a = dist.sample().item()
        return a, dist.log_prob(torch.tensor(a)).item()

    def update(self):
        # First shuffle the data in memory
        actor_losses, critic_losses = [], []
        mem_list = list(self.memory.buffer)
        random.shuffle(mem_list) # shuffled list of tuples. s, s_ are numpy ndarrays, a, r, dones are floats or bools.

        # Get vectors for states, actions, rewards, s_,  dones etc.
        s_vector = torch.Tensor(np.array([t[0] for t in mem_list])) #shape: [capacity, obs_size]
        a_vector = torch.Tensor([t[1] for t in mem_list]) #shape: [capacity, 1]
        logprobs_vector = torch.Tensor([t[2] for t in mem_list]) #shape: [capacity, 1]
        advantages_vector = torch.Tensor([t[3] for t in mem_list]) #shape: [capacity, 1]
        returns_vector = torch.Tensor([t[4] for t in mem_list]) #shape: [capacity, 1]

        # Normalize advantages vector
        advantages_vector = (advantages_vector - advantages_vector.mean()) / (advantages_vector.std() + 1e-8)

        # Update gradients
        for i in range(self.k):
            for idx in range(0, s_vector.shape[0], self.batch_size):

                batch_s = s_vector[idx: idx+ self.batch_size] #shape: [batch_size,]
                batch_a = a_vector[idx: idx+ self.batch_size]
                batch_logprobs = logprobs_vector[idx: idx+ self.batch_size]
                batch_advantages = advantages_vector[idx: idx+self.batch_size]
                batch_returns = returns_vector[idx: idx+self.batch_size]

                # Get new probabilities. We will backpropagate on these.
                new_probs = self.policy.model(batch_s)
                new_prob_distribution = dist.Categorical(new_probs) # Do this so we can get the entropy of the prob dist
                new_logprobs = new_prob_distribution.log_prob(batch_a) # gets the logprobs of the actions that were actually taken
                entropy = new_prob_distribution.entropy().mean()

                ratios = torch.exp(new_logprobs - batch_logprobs.detach())

                # Actor loss
                surr1 = ratios.unsqueeze(-1) * batch_advantages
                surr2 = torch.clamp(ratios.unsqueeze(-1), 1-self.c, 1+self.c) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() + 0.01 * entropy

                # Critic loss
                values_pred = self.critic.model(batch_s)
                critic_loss = 0.5 * (values_pred - batch_returns).pow(2).mean()

                # Update critic
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

                # Update actor
                self.policy.optimizer.zero_grad()
                actor_loss.backward()
                self.policy.optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        print(f"[PPO UPDATE] avg_actor_loss={np.mean(actor_losses):.4f}, "
              f"avg_critic_loss={np.mean(critic_losses):.4f}")

        # clear the buffer and fix old policy
        self.memory.clear()
        self.old_policy.load_state_dict(self.policy.state_dict())

    def compute_gae(self, rewards, values, dones, gamma, lam):
        T = len(rewards)
        advantages  = torch.zeros(T)
        gae = 0
        for t in reversed(range(T)):
            delta = rewards[t] + (gamma*values[t+1] * (1 - dones[t])) - values[t]
            gae = delta + gamma*lam*(1- dones[t])*gae
            advantages[t] = gae
        returns = advantages + torch.Tensor(values[:-1])
        return advantages, returns

    def train_multiagent(self):
        train_scores = []

        for ep in range(self.num_episodes):
            # Buffers for this HAND
            states, actions, logps = [], [], []
            rewards, dones, values = [], [], []
            self.env.reset()
            agent_returns = {agent: 0.0 for agent in self.player_agents}


            for agent in self.env.agent_iter():
                obs, reward, termination, truncation, info = self.env.last()

                if termination or truncation:
                    action = None
                    values.append(0.0)
                else:
                    state_vec   = obs["observation"]
                    action_mask = obs["action_mask"]

                    v = self.critic.model(torch.Tensor(state_vec).unsqueeze(0)).item()

                    a, lp = self.act_multiagent(state_vec, action_mask)

                    states.append(state_vec)
                    actions.append(a)
                    logps.append(lp)
                    values.append(v)

                    action = a

                self.env.step(action)

                rewards.append(reward)
                dones.append(termination or truncation)

                if agent in agent_returns:
                    agent_returns[agent] += reward


            values.append(0.0)

            advantages, returns = self.compute_gae(
                rewards, values, dones,
                self.critic.gamma, self.critic.lam
            )

            for s, a, lp, adv, ret in zip(states, actions, logps, advantages, returns):
                self.memory.push(s, a, lp, adv.item(), ret.item())

            if len(self.memory) >= self.update_frequency:
                self.update()


            r0 = agent_returns[self.player_agents[0]]
            r1 = agent_returns[self.player_agents[1]]
            train_scores.append((r0, r1))
            # print(f"[Episode {ep:03d}] P0={r0:.2f}, P1={r1:.2f}")

        return train_scores, self.policy

    def test_multiagent(self):
        scores = []
        for _ in range(self.testing_episodes):
            total_r = 0
            self.test_env.reset()
            for agent in self.test_env.agent_iter():
                obs, reward, termination, truncation, info = self.test_env.last()
                if termination or truncation:
                    action = None
                else:
                    state_vec   = obs["observation"]
                    action_mask = obs["action_mask"]
                    a, _        = self.act_multiagent(state_vec, action_mask)
                    action = a
                self.test_env.step(action)
                total_r += reward
            scores.append(total_r)
        return scores


    def train(self):
        """
        Uses PPO to train agent
        """
        train_scores = []
        step = 0
        for e in tqdm(range(self.num_episodes)):
            states, actions, rewards, logprobs, dones, values = [], [], [], [], [], []

            s, _ = self.env.reset()
            s_tensor = torch.Tensor(s)
            done = False
            episode_score = 0
            while not done:
                # Get action and logprob associated with action (old policy)
                a, logprob = self.act(s_tensor)

                # Get value associated with state
                v = self.critic.model(s_tensor)

                # Step in environment
                s_, r, terminated, truncated, _ = self.env.step(a)
                done = terminated or truncated

                # Append to states
                states.append(s)
                actions.append(a)
                rewards.append(r)
                logprobs.append(logprob)
                dones.append(done)
                values.append(v)

                # self.memory.push(s, a, r, s_, done)
                s = s_
                episode_score += r
                step += 1

            # End of trajectory
            # Append the value of the current state for the GAE calculation
            values.append(torch.tensor(0.0))

            # Compute Generalized Advantage Estimation
            advantages, returns = self.compute_gae(rewards, values, dones, self.critic.gamma, self.critic.lam)

            for t in range(len(states)):
                self.memory.push(states[t], actions[t], logprobs[t], advantages[t], returns[t])

            if len(self.memory) >= self.update_frequency:
                self.update()

            train_scores.append(episode_score)
        return train_scores, self.policy

    def test(self):
        test_scores = []
        for e in range(self.testing_episodes):
            s, _ = self.env.reset()
            s_tensor = torch.Tensor(s)
            done = False
            episode_score = 0
            while not done:
                a, logprob = self.act(s_tensor)
                s_, r, terminated, truncated, _ = self.env.step(a)
                done = terminated or truncated
                episode_score += r
                s = s_
            test_scores.append(episode_score)
        return test_scores

