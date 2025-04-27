from math import gamma
from re import A
import torch
import torch.nn as nn
import copy

# Pass in config from interaction
class PPOActorNetwork(nn.Module):
  def __init__(self, obs_size, action_size, actor_config):
    super().__init__()
    self.obs_size = obs_size
    self.action_size = action_size

    self.model = self._make_network(actor_config)
    # self.critic = self._make_network(critic_config)

    self.gamma = actor_config.gamma

    self.gradient_clipping = actor_config.gradient_clipping

    self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr = actor_config.learning_rate)

  def _make_network(self, config):
    in_layer = self.obs_size
    layers = []
    for h_size in config.hidden_layer_sizes:
      layers.append(nn.Linear(in_layer, h_size)) 
      layers.append(nn.ReLU())
      in_layer = h_size
    layers.append(nn.Linear(h_size, self.action_size))
    layers.append(nn.Softmax(dim = -1))
    
    return nn.Sequential(*layers)



class PPOCriticNetwork(nn.Module):
  def __init__(self, obs_size, action_size, critic_config):
    super().__init__()
    self.obs_size = obs_size
    self.action_size = action_size

    # self.actor  = self._make_network(actor_config)
    self.model = self._make_network(critic_config)

    self.gamma = critic_config.gamma
    self.lam = critic_config.lam

    self.gradient_clipping = critic_config.gradient_clipping

    self.optimizer = torch.optim.Adam(self.model.parameters(),
                                      lr = critic_config.learning_rate)

  def _make_network(self, config):
    in_layer = self.obs_size
    layers = []
    for h_size in config.hidden_layer_sizes:
      layers.append(nn.Linear(in_layer, h_size)) 
      layers.append(nn.ReLU())
      in_layer = h_size
    layers.append(nn.Linear(h_size, 1))
    
    return nn.Sequential(*layers)


