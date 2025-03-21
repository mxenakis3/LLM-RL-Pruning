from ctypes.wintypes import tagRECT
from math import gamma
from re import A
import torch
import torch.nn as nn
import copy

# Pass in config from interaction

class PPONetwork(nn.Module):
  def __init__(self, obs_size, action_size, actor_config, critic_config):
    super().__init__()
    self.obs_size = obs_size
    self.action_size = action_size

    self.actor  = self._make_network(actor_config)
    self.critic = self._make_network(critic_config)

  def _make_network(self, config):
    in_layer = self.obs_size
    layers = []
    for h_size in config.hidden_layer_sizes:
      layers.append(nn.Linear(in_layer, h_size)) 
      layers.append(nn.ReLU())
      in_layer = h_size
    if config.is_actor:
      layers.append(nn.Linear(h_size, self.action_size))
      layers.append(nn.Softmax(dim = -1))
    else:
      layers.append(nn.Linear(h_size, 1))
    
    return nn.Sequential(*layers)




