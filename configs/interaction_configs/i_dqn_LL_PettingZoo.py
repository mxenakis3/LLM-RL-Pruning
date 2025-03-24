from configs.config_class import Config

# THE DEFAULT CONFIGURATION SETTINGS FOR THE LUNAR LANDER ENVIRONMENT USING DEEP Q-NETWORK AGENTS

configs = {
# INTERACTION CONFIGURATIONS:
"env": "tictactoe_v3",
"episode_length": 1000,
"training_episodes": 800,
"render_mode_train": "None", # "human" plays an animation, "None" skips the animation
"render_mode_test": "human", # "human" plays an animation, "None" skips the animation
"continuous": False, # Discrete, or continuous simulation -- OpenAI parameter
"testing_episodes": 50,
# Eps decay: decay for greedy vs. nongreedy exploration
"eps_start": 1.0,

"eps_end": 0.01,
"eps_decay_episodes": 750, # Number of episodes of decay (if linear decay)
"eps_decay_rate": 0.9995, # rate of exponential decay (if exponential decay)
"decay_type": "exponential", # linear or exponential

# kap decay - decay for llm vs. random selection (given nongreedy exploration)
# Set kappa to zero if no LLM agent.
"kap_start": 0,
"kap_end": 0,
"kap_decay_episodes": 500,
"kap_decay_rate": 0.98,
"kap_decay_type": "exponential", # linear or exponential

# AGENT CONFIGURATIONS
"learning_rate": .0001,
"replay_buffer_size": 100000,
"loss_function": "mseloss",
"optimizer": "adam",
"gamma": 0.99,
"batch": True,
"batch_size": 64,
"obs_size": 18, # dimension of observation
"action_space_size": 9, # dimension of action
"hidden_layer_sizes": [128, 128],
"activations": ["ReLU", "ReLU"],
"target_network": True,
"target_update_frequency": 64,
"learning_frequency": 4
}

dqn_tictactoe_default_configs_dict = Config(configs)



chess_configs = {
# INTERACTION CONFIGURATIONS:
"env": "tictactoe_v3",
"episode_length": 1000,
"training_episodes": 800,
"render_mode_train": "None", # "human" plays an animation, "None" skips the animation
"render_mode_test": "human", # "human" plays an animation, "None" skips the animation
"continuous": False, # Discrete, or continuous simulation -- OpenAI parameter
"testing_episodes": 50,
# Eps decay: decay for greedy vs. nongreedy exploration
"eps_start": 1.0,

"eps_end": 0.01,
"eps_decay_episodes": 750, # Number of episodes of decay (if linear decay)
"eps_decay_rate": 0.99, # rate of exponential decay (if exponential decay)
"decay_type": "exponential", # linear or exponential

# kap decay - decay for llm vs. random selection (given nongreedy exploration)
# Set kappa to zero if no LLM agent.
"kap_start": 0,
"kap_end": 0,
"kap_decay_episodes": 500,
"kap_decay_rate": 0.98,
"kap_decay_type": "exponential", # linear or exponential

# AGENT CONFIGURATIONS
"learning_rate": .0001,
"replay_buffer_size": 100000,
"loss_function": "mseloss",
"optimizer": "adam",
"gamma": 0.99,
"batch": True,
"batch_size": 64,
"obs_size": 7104, # dimension of observation
"action_space_size": 4672, # dimension of action
"hidden_layer_sizes": [128, 128],
"activations": ["ReLU", "ReLU"],
"target_network": True,
"target_update_frequency": 64,
"learning_frequency": 4
}

dqn_chess_default_configs_dict = Config(chess_configs)