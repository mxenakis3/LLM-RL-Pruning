import gymnasium as gym
import torch as torch
import numpy as np
from agents.dqn_agent import DeepQNetwork
from agents.llm_chain_of_though_agent import Chain_of_Thought
from tqdm import tqdm
from configs.i_dqn_LL_default import dqn_lunar_lander_default_configs_dict as dqn_ll_config
from configs.a_lunarlander_cot_agent import lunarlander_cot_agent_configs as cot_ll_config
class DQNInteraction:
  """
  Generic Class for agent/environment interaction.
  Currently, env is set to lunar_lander
  Ideally this will be customizable
  """
  def __init__(self, dqn_ll_config): # Hard code this as the config file.
    self.config = dqn_ll_config # Instance of config class. 
    self.llm_config = cot_ll_config

  def train(self):
    """
    Uses Deep Q Learning to train agent
    """
    # INIT ENVIRONMENT
    env = self._get_environment(self.config, train=True)
    
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
    llm_agent = Chain_of_Thought(config = self.llm_config)

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
            s_as_dict =  self._get_state_dict(s)
            cot_ll_config.system_message["content"] = cot_ll_config.system_message["content"].format(**s_as_dict)
            llm_agent.messages.append(cot_ll_config.system_message)
            a = llm_agent()
          else:
            # Random action
            a = env.action_space.sample()
        else:
          # Choose greedy action
          # Ensure the state is correctly formatted (e.g., tensor, reshaped)
          s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)  # Add batch dimension if needed
          with torch.no_grad():  # No need to track gradients for action selection
              q_values = agent.model(s_tensor)  # Get Q-values for each action
          a = torch.argmax(q_values).item()  # Choose action with the highest Q-value

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


  def test(self, agent):
    """
    Uses trained model to run training episodes
    """

    # INIT SCORES
    episode_scores = []

    # INIT ENVIRONMENT
    env = self._get_environment(self.config, train=False) # train == False means we are testing

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
        s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)  # Add batch dimension if needed
        with torch.no_grad():  # No need to track gradients for action selection
            q_values = agent.model(s_tensor)  # Get Q-values for each action
        a = torch.argmax(q_values).item()  # Choose action with the highest Q-value

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
    s_as_dict=  {
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
    continuous = self.config.continuous

    # LunarLander
    if config.env.lower() == "lunarlander-v3":
      return gym.make("LunarLander-v3", continuous = continuous, render_mode = render_mode)


