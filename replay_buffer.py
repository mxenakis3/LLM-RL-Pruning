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

  def __len__(self):
    return len(self.buffer)
