from collections import deque
import random
import numpy as np

class ReplayBuffer:
  """
  Replay Buffer: Stores past experiences and samples it during training.
  """
  def __init__(self, capacity, batch_size):
    self.buffer = deque(maxlen = capacity)
    self.batch_size = batch_size
  
  def push(self, state, action, reward, next_state, done):
    # state, action reward, next_state are already numpy arrays
    self.buffer.append((state, action, reward, next_state, done))
  
  def sample(self):
    return random.sample(self.buffer, self.batch_size)
  
  def clear(self):
    self.buffer.clear()

  def __len__(self):
    return len(self.buffer)
  

class PPOReplayBuffer(ReplayBuffer):
    """
    Replay Buffer for PPO or other algorithms that need to store values and logprobs.
    """
    def __init__(self, capacity, batch_size):
        super().__init__(capacity, batch_size)
    
    def push(self, state, action, reward, done, value, logprob, advantage, returns):
        """
        Push an experience into the buffer, including value and logprob.
        """
        self.buffer.append((state, action, reward, done, value, logprob, advantage, returns))
    
    def sample(self):
        """
        Sample a random batch of experiences from the buffer.
        """
        return random.sample(self.buffer, self.batch_size)
