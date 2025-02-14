import gymnasium as gym
import yaml
import torch as torch
import numpy as np
from agents.dqn_agent import DeepQNetwork
from tqdm import tqdm

from interactions.dqn_interaction import DQNInteraction

# LOAD CONFIG
# Contains params for interaction loop (ie. number of training episodes)
with open("dqn_interaction_config.yaml", 'r') as file:
  interaction_config = yaml.safe_load(file)

# Contains params for deep-learning agent (ie. learning rates, hidden layer sizes, etc.)
with open("dqn_agent_config.yaml", 'r') as file:
  agent_config = yaml.safe_load(file)

# Run interaction
dqn_interaction = DQNInteraction(interaction_config, agent_config)
episodes, agent = dqn_interaction.train()