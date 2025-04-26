
from configs.config_class import Config

env_lunar_lander_config = Config({
  # INTERACTION CONFIGURATIONS:
"env": "lunarlander_v3",
"render_mode_train": "None", # "human" plays an animation, "None" skips the animation
"render_mode_test": "None", # "human" plays an animation, "None" skips the animation
"continuous": False, # Discrete, or continuous simulation -- OpenAI parameter})
})

interaction_example_lunarlander = Config({
  # INTERACTION CONFIGURATIONS:
"episode_length": 64, # Max length of a single episode
"training_episodes": 50,
"testing_episodes": 50,
"eps_start": 1.0,
"eps_end": 0.001,
"eps_decay_episodes": 750, # Number of episodes of decay (if linear decay)
"eps_decay_rate": 0.99, # rate of exponential decay (if exponential decay)
"eps_decay_type": "exponential", # linear or exponential
"kap_start": 1.0, # Set kappa to zero if no LLM agent. 
"kap_end": 0.001,
"kap_decay_episodes": 5,
"kap_decay_rate": 0.98,
"kap_decay_type": "linear", # linear or exponential
})

agent_example_lunarlander = Config({
  # These parameters seem to work well for lunarlander
"learning_rate": .0001,
"replay_buffer_size": 10000,
"loss_function": "mseloss",
"optimizer": "adam",
"gamma": 0.99,
"batch": True, # Keep as true. Should toggle batch gradient descent
"batch_size": 64,
"obs_size": 8, # dimension of observation space. Used to fit layers
"action_space_size": 4, # dimension of action. Used to fit layers
"hidden_layer_sizes": [128, 128],
"activations": ["ReLU", "ReLU"],
"target_network": True, # Keep as true by default, creates another network to create estiamtes for Q(s,a), stabilizes training
"target_update_frequency": 64, # The number of steps we take before we reset the target network to have the values as the current network
"learning_frequency": 4 # Number of steps before training
})
