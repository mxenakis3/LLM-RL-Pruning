import gymnasium as gym
import torch as torch
import numpy as np
import random
from agents.dqn_agent import DeepQNetwork
# from agents import llm_chain_of_thought_agent
from agents import tool_calling_agent_ex
from tqdm import tqdm
from utils import get_environment, get_render_mode


class DQNInteraction:
	"""
	Generic Class for agent/environment interaction.
	Now supports both LunarLander and a PettingZoo
	"""
	def __init__(self, dqn_ll_config, cot_ll_config): # Hard code this as the config file.
		self.config = dqn_ll_config # Instance of config class.
		self.llm_config = cot_ll_config

	def train(self):
		"""Dispatch training to the correct method based on the environment."""
		if self.config.env.lower() == "lunarlander-v3":
			return self.train_lunarlander()
		else:
			return self.train_pettingzoo()

	def train_lunarlander(self):
		"""
		Uses Deep Q Learning to train agent
		"""
		# INIT ENVIRONMENT
		env = get_environment(self.config, train=True)

		# INIT EPSILON
		# Initialize decay values
		epsilon = self.config.eps_start
		kappa = self.config.kap_start

		# If linear decay, calculate decrement for epsilon, kappa decay
		if self.config.decay_type.lower() == "linear":
			eps_decrement = (self.config.eps_start - self.config.eps_end) / self.config.eps_decay_episodes

		if self.config.kap_decay_type.lower() == "linear":
			kap_decrement = (self.config.kap_start - self.config.kap_end) / self.config.kap_decay_episodes

		# INIT AGENTS
		agent = DeepQNetwork(config = self.config)
		# llm_agent = llm_chain_of_thought_agent.Chain_of_Thought(config = self.llm_config)
		llm_agent = tool_calling_agent_ex.Chain_of_Thought(config = self.llm_config)

		# INIT SCORE TUPLES
		episode_scores = []

		# TRAINING LOOP
		# Outer loop = Training episode loop
		for e in tqdm(range(self.config.training_episodes)):

			# Reset the environment to generate the first state 's'
			s, info = env.reset()
			done = False
			score = 0

			# Run episode until terminal state is reached
			while not done:
				# Using epsilon greedy, decide between greedy and non-greedy action
				if np.random.rand() < epsilon:
					# using kappa-greedy, decide between LLM and random
					if np.random.rand() < kappa:
						# Fill the values of the current state into a dictionary readable by the LLM agent
						s_as_dict =	self._get_state_dict(s)
						self.llm_config.system_message["content"] = self.llm_config.system_message["content"].format(**s_as_dict)
						llm_agent.messages.append(self.llm_config.system_message)
						a = llm_agent()
					else:
						# Random action
						a = env.action_space.sample()
				else:
					# Choose greedy action
					# Ensure the state is correctly formatted (e.g., tensor, reshaped)
					s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)	# Add batch dimension if needed
					with torch.no_grad():	# No need to track gradients for action selection
							q_values = agent.model(s_tensor)	# Get Q-values for each action
					a = torch.argmax(q_values).item()	# Choose action with the highest Q-value

				# Take action
				s_, r, terminated, truncated, info = env.step(a)
				done = terminated or truncated

				# Increment score
				score += r

				# Append experience to agent's memory buffer
				agent.replay_buffer.push(s, a, r, s_, done)

				# Close-out step
				s = s_

				# Increment step counter for agent
				agent.step_counter += 1

				# Learn and update target network if needed:
				if agent.step_counter % agent.learning_frequency == 0:
					agent.learn()

				if agent.step_counter % agent.target_update_frequency == 0:
					agent.update_target_net()


			# Append episode score (for plots)
			episode_scores.append(score)

			# Recompute epsilon
			if self.config.decay_type.lower() == "linear":
				epsilon = max(self.config.eps_end, epsilon - eps_decrement)
			else:
				epsilon = epsilon * self.config.eps_decay_rate

			# Recompute kappa
			if self.config.kap_decay_type.lower() == "linear":
				kappa = max(self.config.kap_end, kappa - kap_decrement)
			else:
				kappa = kappa * self.config.kap_decay_rate


		# Close pygame window
		env.close()

		# Episode scores: score for each training episode
		# Agent: trained agent
		return episode_scores, agent

	def train_pettingzoo(self, env_constructor, fixed_agent = ""):
		env = env_constructor()
		env.reset()
		agents = env.agents

		llm_agent = llm_chain_of_thought_agent.Chain_of_Thought(config=self.llm_config)

		# Create one DQN agent per environment agent
		dqn_agents = {agent: DeepQNetwork(config=self.config) for agent in agents}
		epsilons = {agent: self.config.eps_start for agent in agents}
		if fixed_agent:
			epsilons[fixed_agent] = 0.05
		kappa = self.config.kap_start

		if self.config.decay_type.lower() == "linear":
			eps_decrement = (self.config.eps_start - self.config.eps_end) / self.config.eps_decay_episodes
		if self.config.kap_decay_type.lower() == "linear":
			kap_decrement = (self.config.kap_start - self.config.kap_end) / self.config.kap_decay_episodes

		episode_rewards_all = []

		for episode in tqdm(range(self.config.training_episodes)):
			env.reset()
			episode_rewards = {agent: 0 for agent in agents}

			for agent in env.agent_iter():
				observation, reward, terminated, truncated, info = env.last()
				done = terminated or truncated

				if not done:
					agent_epsilon = epsilons[agent]
					# if np.random.rand() < agent_epsilon:
					if np.random.rand() < agent_epsilon:
						if np.random.rand() < kappa:
							a = llm_agent()
						else:
							a = self.get_random_available_action(observation)
					else:
						a = dqn_agents[agent].select_action(observation)
				else:
					a = None


				# # Break out of current iteration if no legal moves possible
				# if a == None:
				# 	break

				env.step(a)

				episode_rewards[agent] += reward

			for agent in agents:
				if agent != fixed_agent:
					if self.config.decay_type.lower() == "linear":
						epsilons[agent] = max(self.config.eps_end, epsilons[agent] - eps_decrement)
					else:
						epsilons[agent] *= self.config.eps_decay_rate

			if self.config.kap_decay_type.lower() == "linear":
				kappa = max(self.config.kap_end, kappa - kap_decrement)
			else:
				kappa *= self.config.kap_decay_rate

			# Update only the learning agent's network:
			for agent in agents:
				if agent != fixed_agent:
					dqn_agents[agent].train()
			episode_rewards_all.append(episode_rewards)

			print(f"Episode {episode+1} - Rewards: {episode_rewards} - Epsilon: {epsilons}")

		return episode_rewards_all, dqn_agents

	def get_random_available_action(self, observation):
		# If observation is a dict, try to use the action mask first.
		if isinstance(observation, dict):
			if "action_mask" in observation:
				mask = np.array(observation["action_mask"])
				return random.choice(list(np.where(mask == 1)[0]))
			else:
				raise ValueError("Observation dict must contain an 'action_mask' or 'observation' key.")

	def test_pettingzoo(self, dqn_agents, env_constructor, num_episodes=10):
		"""
		Test function for a PettingZoo environment.
		Runs num_episodes of the game with trained agents, using greedy actions.
		Returns the rewards per episode.
		"""
		episode_rewards_all = []

		for episode in range(num_episodes):
			env = env_constructor()
			env.reset()
			episode_rewards = {agent: 0 for agent in env.agents}

			for agent in env.agent_iter():
				observation, reward, terminated, truncated, info = env.last()
				done = terminated or truncated

				a = dqn_agents[agent].select_action(observation) if not done else None
				env.step(a)
				episode_rewards[agent] += reward

			episode_rewards_all.append(episode_rewards)
			env.close()

		return episode_rewards_all

	def test(self, agent):
		"""
		Uses trained model to run training episodes
		"""

		# INIT SCORES
		episode_scores = []

		# INIT ENVIRONMENT
		env = get_environment(self.config, train=False) # train == False means we are testing

		# TEST LOOP
		# Outer loop = Test episode loop
		for e in tqdm(range(self.config.testing_episodes)):

			# Reset the environment to generate the first state 's'
			s, info = env.reset()
			done = False
			score = 0

			# Run episode until terminal state is reached
			while not done:

				# Choose greedy action
				s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)	# Add batch dimension if needed
				with torch.no_grad():	# No need to track gradients for action selection
						q_values = agent.model(s_tensor)	# Get Q-values for each action
				a = torch.argmax(q_values).item()	# Choose action with the highest Q-value

				# Take action
				s_, r, terminated, truncated, info = env.step(a)

				# Increment score
				score += r

				# Close-out step
				s = s_
				done = terminated or truncated

			# Append episode score (for plots)
			episode_scores.append(score)

		env.close()
		return episode_scores


	def _get_state_dict(self, s):
		# RIGHT NOW THIS IS HARDCODED FOR LUNARLANDER
		s_as_dict=	{
			"x_pos": s[0],
			"y_pos": s[1],
			"x_vel": s[2],
			"y_vel": s[3],
			"angle": s[4],
			"angular_vel": s[5],
			"left_contact": s[6],
			"right_contact": s[7]
		}
		return s_as_dict
