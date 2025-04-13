import torch as torch
import gymnasium as gym
from agents.ppo_agents import PPOActorNetwork, PPOCriticNetwork
from tqdm import tqdm
from configs.agent_configs.a_ppo_agents import critic_configs, actor_configs
import torch.distributions as dist
from replay_buffer import ReplayBuffer
import numpy as np
from collections import deque
import random


class PPO_interaction:
    def __init__(self, interaction_config, actor_configs, critic_configs):
        self.env = self._get_environment(interaction_config, train=True)
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
        self.memory = ReplayBuffer(capacity=interaction_config.capacity, 
                                   batch_size=self.batch_size)
        
        # # Initialize optimizer
        # self.optimizer = torch.optim.Adam(self.policy.parameters(),
        #                                   lr = interaction_config.learning_rate)

    def act(self, state):
        s = torch.Tensor(state)
        with torch.no_grad():
            action_probs = self.old_policy.model(s)
        distribution = dist.Categorical(action_probs)
        a = distribution.sample().item()
        return a
    
    def update(self):
        print(f"Updating...")
        # First shuffle the data in memory
        mem_list = list(self.memory.buffer)
        random.shuffle(mem_list) # shuffled list of tuples. s, s_ are numpy ndarrays, a, r, dones are floats or bools.

        # Get vectors for states, actions, rewards, s_,  dones etc. 
        s_vector = torch.Tensor(np.array([t[0] for t in mem_list])) #shape: [capacity, obs_size]
        a_vector = torch.Tensor([t[1] for t in mem_list]) #shape: [capacity, 1]
        r_vector = torch.Tensor([t[2] for t in mem_list]) #shape: [capacity, 1]
        s__vector = torch.Tensor(np.array([t[3] for t in mem_list])) #shape: [capacity, obs_size]
        dones_vector = torch.Tensor([t[4] for t in mem_list]) #shape: [capacity, 1]

        # calculate advantages
        with torch.no_grad():
            vs_vector = self.critic.model(s_vector)
            vs__vector = self.critic.model(s__vector)
            advantages_vector = r_vector + ((self.critic.gamma*vs__vector))*(1-dones_vector) - vs_vector
            returns_vector = advantages_vector + vs_vector

        # Update gradients
        for i in range(self.k):
            for idx in range(0, s_vector.shape[0], self.batch_size):
                batch_s = s_vector[idx: idx+ self.batch_size] #shape: [batch_size,]
                batch_a = a_vector[idx: idx+ self.batch_size]
                batch_r = r_vector[idx: idx+ self.batch_size]
                batch_s_ = s__vector[idx: idx+ self.batch_size]
                batch_dones = dones_vector[idx: idx+ self.batch_size]
                batch_advantages = advantages_vector[idx: idx+self.batch_size]
                batch_returns = returns_vector[idx: idx+self.batch_size]

                # Get old policy probabilities
                with torch.no_grad():
                    # old_probs = self.old_policy.actor(batch_s) #shape: [batch_size, action_space.n]
                    old_probs = self.old_policy.model(batch_s) #shape: [batch_size, action_space.n]
                    old_prob_distribution = dist.Categorical(old_probs)#shape: [batch_size, action_space.n]
                    old_logprobs = old_prob_distribution.log_prob(batch_a)
        
                # Get new probabilities. We will backpropagate on these. 
                new_probs = self.policy.model(batch_s)
                new_prob_distribution = dist.Categorical(new_probs) # Do this so we can get the entropy of the prob dist
                new_logprobs = new_prob_distribution.log_prob(batch_a) # gets the logprobs of the actions that were actually taken
                entropy = new_prob_distribution.entropy().mean()

                ratios = torch.exp(new_logprobs - old_logprobs.detach())

                # Actor loss
                surr1 = ratios.unsqueeze(-1) * batch_advantages
                surr2 = torch.clamp(ratios.unsqueeze(-1), 1-self.c, 1+self.c) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

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


    def train(self):
        """
        Uses PPO to train agent
        """
        train_scores = []
        step = 0
        for e in tqdm(range(self.num_episodes)):
            s, _ = self.env.reset()
            done = False
            episode_score = 0
            while not done:
                a = self.act(s)
                s_, r, terminated, truncated, _ = self.env.step(a)
                done = terminated or truncated
                self.memory.push(s, a, r, s_, done)
                s = s_
                episode_score += r
                step += 1

                if step % self.update_frequency == 0:
                    self.update()

            train_scores.append(episode_score)
        return train_scores, self.policy
    
    def test(self):
        test_scores = []
        for e in range(self.testing_episodes):
            s, _ = self.env.reset()
            done = False
            episode_score = 0
            while not done:
                a = self.act(s)
                s_, r, terminated, truncated, _ = self.env.step(a)
                done = terminated or truncated
                episode_score += r
                s = s_
            test_scores.append(episode_score)
        return test_scores


    def _get_render_mode(self, render_mode):
        """
        Helper function to get the correct render mode from config.
        """
        render_modes = {
            "human": "human",
            "none": None
        }
        return render_modes.get(render_mode.lower(), None)

    def _get_environment(self, config, train):
        # Define some variables that might make sense given the environment
        if train:
            render_mode = self._get_render_mode(config.render_mode_train)
        else:
            render_mode = self._get_render_mode(config.render_mode_train)

        # Still defining some variables in context
        continuous = config.continuous

        # LunarLander
        if config.env.lower() == "lunarlander-v3":
            return gym.make("LunarLander-v3", continuous=continuous, render_mode=render_mode)