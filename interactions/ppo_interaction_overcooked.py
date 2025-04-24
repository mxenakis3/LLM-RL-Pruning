from tkinter import YES
import torch as torch
import gymnasium as gym
from agents.ppo_agents import PPOActorNetwork, PPOCriticNetwork
from tqdm import tqdm
from configs.agent_configs.a_ppo_agents_overcooked import critic_configs, actor_configs
import torch.distributions as dist
from replay_buffer import PPOReplayBuffer
import numpy as np
from collections import deque
import random
from utils import get_environment, get_render_mode
from overcooked_ai_py.agents.agent import RandomAgent
from agents.llm_chain_of_thought_agent import Chain_of_Thought as LLMAgent


class PPO_interaction:
    def __init__(self, interaction_config, actor_configs, critic_configs, llm_configs):
        self.env = get_environment(interaction_config, train=True)
        self.test_env = get_environment(interaction_config, train=False)
        self.obs_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
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
        
        # Initialize params for llm decay
        self.kap_start = interaction_config.kap_start
        self.kap_end = interaction_config.kap_end
        self.kap_decay_episodes = interaction_config.kap_decay_episodes
        self.kap_decay_rate = interaction_config.kap_decay_rate
        self.kap_decay_type = interaction_config.kap_decay_type

        # intialize kap
        self.kap = self.kap_start

        # Store llm configs
        self.llm_configs = llm_configs


    def act(self, state):
        with torch.no_grad():
            action_probs = self.old_policy.model(state)
        distribution = dist.Categorical(action_probs)
        action = distribution.sample()
        logprob = distribution.log_prob(action)
        return action.item(), logprob.item(), distribution
    
    def update(self):
        # First shuffle the data in memory
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
                #print(self.obs_size)
                batch_s = s_vector[idx: idx+ self.batch_size] #shape: [batch_size,]
                #print("batch_s shape:", batch_s.shape)
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
    
    def train(self):
        """
        Uses PPO to train agent
        """
        train_scores = []
        # step = 0
        mlp_teammate = RandomAgent()
        llm_agent = LLMAgent(self.llm_configs)
        for e in tqdm(range(self.num_episodes)):

            states_agent1, states_agent2 = [], []
            actions_agent1, actions_agent2 = [], []
            rewards_agent1, rewards_agent2 = [], []
            logprobs_agent1, logprobs_agent2 = [], []
            dones_agent1, dones_agent2 = [], []
            values_agent1, values_agent2 = [], []


            s1, s2 = self.env.multi_reset()

            done = False
            episode_score = 0
            while not done:
                s1_tensor, s2_tensor = torch.Tensor(s1), torch.Tensor(s2)

                a1, a1_logprob, a1_dist = self.act(s1_tensor)
                a2, a2_logprob, a2_dist = self.act(s2_tensor)

                # Get new values for a1, a1_logprob, a2, a2_logprob if p
                if np.random.rand() < self.kap:
                    print(f"Entered random loop")
                    self.llm_configs.system_message["content"] = self.llm_configs.system_message["content"].format({"state": self.env.state_to_json(s1)})
                    a1 = llm_agent() # Existing messages get cleared here
                    print(f"Agent 1 choice: {a1}")
                    a1_logprob = a1_dist.log_prob(torch.Tensor(a1))

                    self.llm_config.system_message["content"] = self.llm_config.system_message["content"].format({"state": self.env.state_to_json(s2)})
                    a2 = llm_agent()
                    print(f"Agent 2 choice: {a2}")
                    a2_logprob = a2_dist.log_prob(torch.Tensor(a2))


                v1, v2 = self.critic(s1_tensor).item(), self.critic(s2_tensor).item()
                s1_next, s2_next, r1, r2, done, _ = self.env.multi_step(a1, a2)


                # Append to states
                states_agent1.append(s1)
                states_agent2.append(s2)
                actions_agent1.append(a1)
                actions_agent2.append(a2)
                rewards_agent1.append(r1)
                rewards_agent2.append(r2)
                logprobs_agent1.append(a1_logprob)
                logprobs_agent2.append(a2_logprob)
                dones_agent1.append(done)
                dones_agent2.append(done)
                values_agent1.append(v1)
                values_agent2.append(v2)

                # self.memory.push(s, a, r, s_, done)
                s1, s2 = s1_next, s2_next
                episode_score += r1
                # step += 1

            # End of trajectory
            # Append the value of the current state for the GAE calculation
            values_agent1.append(torch.tensor(0.0))
            values_agent2.append(torch.Tensor(0.0))

            # Compute Generalized Advantage Estimation
            advantages_agent1, returns_agent1 = self.compute_gae(rewards_agent1, values_agent1, dones_agent1, self.critic.gamma, self.critic.lam)
            advantages_agent2, returns_agent2 = self.compute_gae(rewards_agent2, values_agent2, dones_agent2, self.critic.gamma, self.critic.lam)

            for t in range(len(rewards_agent1)):
                self.memory.push(states_agent1[t], actions_agent1[t], logprobs_agent1[t], advantages_agent1[t], returns_agent1[t])
                self.memory.push(states_agent2[t], actions_agent2[t], logprobs_agent2[t], advantages_agent2[t], returns_agent2[t])

            if len(self.memory) >= self.update_frequency:
                self.update()
            # print(episode_score)
            train_scores.append(episode_score)

            if self.kap_decay_type.lower() == "linear":
                self.kap = max(self.kap_end, self.kap - (self.kap_start - self.kap_end / self.kap_decay_episodes))
            else:
                self.kap = self.kap * self.kap_decay_rate

        return train_scores, self.policy
    
    def test(self):
        test_scores = []
        for e in range(self.testing_episodes):
            s, _ = self.env.multi_reset()
            s_tensor = torch.Tensor(s)
            done = False
            episode_score = 0
            while not done:
                a, logprob = self.act(s_tensor)
                s_, r, terminated, truncated = self.env.multi_step(a, a)
                done = np.any(terminated) or np.any(truncated)
                episode_score += r[0]
                s = s_[0]
            test_scores.append(episode_score)
        return test_scores

