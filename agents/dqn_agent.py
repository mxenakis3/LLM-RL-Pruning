import torch as torch
import torch.nn as nn
from replay_buffer import PPOReplayBuffer
import numpy as np


class DeepQNetwork(nn.Module):
  """
  Deep Q Network:
  - Approximates value of Q(s, a) given (s, a)
  - s.shape: [8,] (x, y, x_velocity, y_velocity, angle, angular_velocity, left_leg_touch, right_leg_touch)
  - a.shape: [4,] (do nothing, fire left orientation engine, fire main engine, fire right orientation engine)
  """
  def __init__(self, config):
    super().__init__() # Call nn.Module superclass

    # INITIALIZE REPLAY BUFFER
    self.replay_buffer = PPOReplayBuffer(capacity = config.replay_buffer_size, batch_size = config.batch_size)

    # Store discount rate
    self.gamma = config.gamma

    # Initialize model
    self.model = self.create_model(config.hidden_layer_sizes, config.activations, config.obs_size, config.action_space_size)

    # Get loss function
    self.loss_function = self._get_loss_function(config.loss_function)

    # Get optimizer
    self.optimizer = self._get_optimizer(config.optimizer, config.learning_rate)

    # INITIALIZE Target Network (if target == True)
    self.target_network_exists = config.target_network
    if self.target_network_exists:
      self.target_network = self.create_model(config.hidden_layer_sizes, config.activations, config.obs_size, config.action_space_size)
      # Copy initial weights
      self.target_network.load_state_dict(self.model.state_dict())
    
    # This step counter is used to check later when we need to update the target network
    self.step_counter = 0

    # This is the number of steps we take before we update the target network
    self.target_update_frequency = config.target_update_frequency

    # Number of steps we take before we learn again
    self.learning_frequency = config.learning_frequency



  def create_model(self, hidden_layer_sizes, activations, obs_size, action_space_size):
    """
    Initializes the neural network from config.
    """
    hidden_layer_sizes = hidden_layer_sizes
    layers = []

    # Error handling: If only one activation is provided in config, use it for all layers
    if isinstance(activations, str):
      activations = [activations] * len(hidden_layer_sizes)

    # Dynamically create hidden layers from config.
    # PyTorch automatically sets reasonable random initial values for the hidden layers.
    prev_size = obs_size
    for i, h in enumerate(hidden_layer_sizes):
      layers.append(nn.Linear(prev_size, h))
      layers.append(self._get_activations(activations[i]))
      prev_size = h
    
    # Output layer
    layers.append(nn.Linear(prev_size, action_space_size))

    return nn.Sequential(*layers)


  def _get_activations(self, activation_name):
    """
    Helper function:
    returns the activation function specified by the user in config
    """
    activations = {
      "relu": nn.ReLU(),
      "tanh": nn.Tanh(),
      "sigmoid": nn.Sigmoid(),
      "leaky_relu": nn.LeakyReLU(),
      "none": nn.Identity()
    }
    return activations.get(activation_name.lower(), nn.ReLU())

  
  def _get_loss_function(self, loss_name):
    """
    Helper function:
    returns the loss function specified by the user in config
    """
    loss_functions = {
      "mseloss": nn.MSELoss(),
      "crossentropyloss": nn.CrossEntropyLoss(),
      "bceloss": nn.BCELoss(), # Binary cross entropy loss
      "nllloss": nn.NLLLoss(), # negative log-likelihood loss
      "hingemebeddingloss": nn.HingeEmbeddingLoss(),
      "smoothl1loss": nn.SmoothL1Loss(),
      "cosinesimilarityloss": nn.CosineEmbeddingLoss(),
      "l1loss": nn.L1Loss(),
      "poissonloss": nn.PoissonNLLLoss()
    }
    return loss_functions.get(loss_name.lower(), nn.MSELoss())


  def _get_optimizer(self, optimizer_name, learning_rate):
    """
    Helper function:
    returns the optimizer specified by the user in config
    """
    optimizers = {
      "sgd": torch.optim.SGD(self.model.parameters(), lr=learning_rate),
      "adam": torch.optim.Adam(self.model.parameters(), lr=learning_rate),
    }
    return optimizers.get(optimizer_name.lower(), torch.optim.Adam(self.model.parameters(), lr = 0.001))


  def forward(self, state):
    """
    Create estimate for Q(s,a) values
    Inputs: State information (tuple)
    Outputs: NN approx. of value for each action (Q(s,a) for each a)
    """
    return self.model(state)


  def learn(self):
    """
    Perform gradient descent and update weights. 
    """
    """
    We want to update our estimate of the "value", or "Expected Reward" of taking action 'a' from state 's', whose true value is denoted Q(s,a). 
    Currently, our approximation for the value of state 's' is parameterized by a set of parameters which we will call 'w'. ie) we have Q_{w} (s, a)
    
    We'll update Q_{w} (s, a) towards the value of the next state we reach from (s, a).
    This value is our bootstrapped estimate of the value of the best action that can be taken at its following state, Q_{w} (s_,a_).

    In particular, we will add the mean squared error of the difference between Q(s, a) and (Q(s, a) + reward_a + gamma * (Q(s_, a_))) as a term in our loss function.

    The greater the difference between the current value of Q(s, a) and this updated estimate for Q(s, a), the greater the loss.

    We will then find the gradient of this loss function (taken over many examples) with respect to the weights, w. 

    Finally, we will update the weights in the direction that minimizes this loss function.

    The boostrapped estimates are initially biased, but as the number of training episodes increase, the bias will decrease. But bootstrapping early on like this will prevent overfitting. 
    """
    # Return if there are not enough samples
    if len(self.replay_buffer) < self.replay_buffer.batch_size:
      return 
  
    # Randomly sample a batch of experiences
    # self.replay_buffer.sample: list of tuples
    s, a, r, s_, done = zip(*self.replay_buffer.sample())

    s = torch.tensor(np.array(s), dtype=torch.float32)
    a = torch.tensor(a, dtype=torch.long)
    r = torch.tensor(r, dtype=torch.float32)
    s_ = torch.tensor(np.array(s_), dtype=torch.float32)
    done = torch.tensor(done, dtype=torch.float32)

    # Feed states forward
    Q_s = self.model(s)

    # Use target network to estimate Q value of next state if target network exists.
    # Read about how target networks stabilize estimates
    if self.target_network_exists:
      Q_s_ = self.target_network(s_)
    else:
      Q_s_ = self.model(s_)

    # Find greedy action in Q_s_
    Q_s_max = Q_s_.max(1)[0] # Max of a tensor object along dimension 1 (the action dimension) returns (max_val, max_val_idx). Return [0] from this tuple.

    # Calculate target
    target = r + self.gamma * Q_s_max.detach() * (1 - done)

    # Compute loss
    Q_sgreedy = Q_s[range(Q_s_.shape[0]), a]

    loss = self.loss_function(Q_sgreedy, target) # Assumes a is the index of the action to take

    # Clear old gradients from previous iteration - we want to start with a fresh gradient
    self.optimizer.zero_grad()

    # Performs backpropogation - 
    # Here, loss is a scalar tensor whose value is derived from the forward passes
    loss.backward()

    # Updates the model parameters
    self.optimizer.step()


  def update_target_net(self):
    # Update target network if needed
    if self.target_network_exists:
      self.target_network.load_state_dict(self.model.state_dict())

    
    
    
    
   