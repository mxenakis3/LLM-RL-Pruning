from configs.config_class import Config

# THE DEFAULT CONFIGURATION SETTINGS FOR THE LUNAR LANDER ENVIRONMENT USING DEEP Q-NETWORK AGENTS

configs = {
# INTERACTION CONFIGURATIONS:
"env": "lunarlander-v3",
"episode_length": 1000,
"training_episodes": 500,
"render_mode_train": "None", # "human" plays an animation, "None" skips the animation
"render_mode_test": "human", # "human" plays an animation, "None" skips the animation 
"continuous": False, # Discrete, or continuous simulation -- OpenAI parameter
"eps_start": 1.0,
"eps_end": 0.001,
"eps_decay": 750, # Number of episodes of decay (if linear decay)
"eps_decay_rate": 0.99, # rate of exponential decay (if exponential decay)
"decay_type": "exponential", # linear or exponential
"testing_episodes": 50,

# AGENT CONFIGURATIONS
"learning_rate": .0001,
"replay_buffer_size": 100000,
"loss_function": "mseloss",
"optimizer": "adam",
"gamma": 0.99,
"batch": True,
"batch_size": 64,
"obs_size": 8, # dimension of observation
"action_space_size": 4, # dimension of action
"hidden_layer_sizes": [128, 128],
"activations": ["ReLU", "ReLU"],
"target_network": True,
"target_update_frequency": 64,
"learning_frequency": 4
}

dqn_lunar_lander_default_configs_dict = Config(configs)