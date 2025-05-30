
texas_holdem_config = {
  # INTERACTION CONFIGURATIONS:
"env": "texas_holdem_v4",
"render_mode_train": "None", # "human" plays an animation, "None" skips the animation
"render_mode_test": "None", # "human" plays an animation, "None" skips the animation
"continuous": False, # Discrete, or continuous simulation -- OpenAI parameter})
}

ppo_interaction_config_texas = {
  # INTERACTION CONFIGURATIONS:
"training_episodes": 70000,
"testing_episodes": 3000,
"bot_type": "heuristic", # "heuristic" or "random"
"c": 0.1, # Loss function clipping coefficient
"batch_size":512, # Size of batch for SGD in update method.
"capacity": 20000, # Size of experience tuple (s, a, r, s_, ...) memory
"num_epochs": 6, # number of batches in each iteration of learning
"update_frequency": 4000, # how many steps are taken before an update occurs
"kap_start" : 1.0, # The probability of samping an action from the LLM agent at the first timestep
"kap_end": 0.001, # The min probability of sampling an action from the LLM agent
"kap_decay_episodes": 3, # For linear decay: The number of steps we are taking to decay kap. Each episode, kap decays by (kap_start - kap_finish)/ kap_decay_episodes.
"kap_decay_rate": 0.98, # For exponential decay: The rate of decay for kap, the probability that we sample from llm. kap = kap_start* kap_decay_rate^(episode_number)
"kap_decay_type": "linear", # Exponential or linear
}

actor_configs_texas = {
# AGENT CONFIGURATIONS
"learning_rate": .0002,
"loss_function": "mseloss",
"optimizer": "adam",
"gamma": 0.99, # Discount factor
"obs_size": 76, # dimension of observation
"output_size": 4, # dimension of action space
"hidden_layer_sizes": [256, 256],
"activations": ["ReLU", "ReLU"],
"is_actor": True, # Leave as True: distinguishes actor and critic networks
"gradient_clipping": 0.5
}

critic_configs_texas = {
# INTERACTION CONFIGURATIONS:

# AGENT CONFIGURATIONS
"learning_rate": 5e-4,
"loss_function": "mseloss",
"optimizer": "adam",
"gamma": 1.0, # Discount factor
"lam":0.95, # Exponential weight for GAE. Probably leave at 0.95
"obs_size": 76, # dimension of observation
"output_size": 1, # scalar output
"hidden_layer_sizes": [256, 256],
"activations": ["ReLU", "ReLU"],
"is_actor": False, # Leave as false: distinguishes actor and critic networks,
"gradient_clipping": 0.5
}

ppo_interaction_config_texas_opt = {
  # INTERACTION CONFIGURATIONS:
"training_episodes": 100000,
"testing_episodes": 5000,
"bot_type": "heuristic", # "heuristic" or "random"
"c": 0.05, # Loss function clipping coefficient
"batch_size":256, # Size of batch for SGD in update method.
"capacity": 20000, # Size of experience tuple (s, a, r, s_, ...) memory
"num_epochs": 8, # number of batches in each iteration of learning
"update_frequency": 4000, # how many steps are taken before an update occurs
"kap_start" : 1.0, # The probability of samping an action from the LLM agent at the first timestep
"kap_end": 0.001, # The min probability of sampling an action from the LLM agent
"kap_decay_episodes": 3, # For linear decay: The number of steps we are taking to decay kap. Each episode, kap decays by (kap_start - kap_finish)/ kap_decay_episodes.
"kap_decay_rate": 0.98, # For exponential decay: The rate of decay for kap, the probability that we sample from llm. kap = kap_start* kap_decay_rate^(episode_number)
"kap_decay_type": "linear", # Exponential or linear
}

actor_configs_texas_opt = {
# AGENT CONFIGURATIONS
"learning_rate": 1e-3,
"loss_function": "mseloss",
"optimizer": "adam",
"gamma": 0.99, # Discount factor
"obs_size": 76, # dimension of observation
"output_size": 4, # dimension of action space
"hidden_layer_sizes": [256, 256],
"activations": ["ReLU", "ReLU"],
"is_actor": True, # Leave as True: distinguishes actor and critic networks
"gradient_clipping": 0.5
}

critic_configs_texas_opt = {
# INTERACTION CONFIGURATIONS:

# AGENT CONFIGURATIONS
"learning_rate": 3e-4,
"loss_function": "mseloss",
"optimizer": "adam",
"gamma": 1.0, # Discount factor
"lam":0.95, # Exponential weight for GAE. Probably leave at 0.95
"obs_size": 76, # dimension of observation
"output_size": 1, # scalar output
"hidden_layer_sizes": [256, 256],
"activations": ["ReLU", "ReLU"],
"is_actor": False, # Leave as false: distinguishes actor and critic networks,
"gradient_clipping": 0.5
}
ppo_interaction_config_texas_random_opt = {
  # INTERACTION CONFIGURATIONS:
"training_episodes": 100000,
"testing_episodes": 5000,
"bot_type": "random", # "heuristic" or "random"
"c": 0.15, # Loss function clipping coefficient
"batch_size":128, # Size of batch for SGD in update method.
"capacity": 20000, # Size of experience tuple (s, a, r, s_, ...) memory
"num_epochs": 4, # number of batches in each iteration of learning
"update_frequency": 1000, # how many steps are taken before an update occurs
"kap_start" : 1.0, # The probability of samping an action from the LLM agent at the first timestep
"kap_end": 0.001, # The min probability of sampling an action from the LLM agent
"kap_decay_episodes": 3, # For linear decay: The number of steps we are taking to decay kap. Each episode, kap decays by (kap_start - kap_finish)/ kap_decay_episodes.
"kap_decay_rate": 0.98, # For exponential decay: The rate of decay for kap, the probability that we sample from llm. kap = kap_start* kap_decay_rate^(episode_number)
"kap_decay_type": "linear", # Exponential or linear
}

actor_configs_texas_random_opt = {
# AGENT CONFIGURATIONS
"learning_rate": 3e-5,
"loss_function": "mseloss",
"optimizer": "adam",
"gamma": 0.99, # Discount factor
"obs_size": 76, # dimension of observation
"output_size": 4, # dimension of action space
"hidden_layer_sizes": [256, 256],
"activations": ["ReLU", "ReLU"],
"is_actor": True, # Leave as True: distinguishes actor and critic networks
"gradient_clipping": 0.5
}

critic_configs_texas_random_opt = {
# INTERACTION CONFIGURATIONS:

# AGENT CONFIGURATIONS
"learning_rate": 3e-4,
"loss_function": "mseloss",
"optimizer": "adam",
"gamma": 1.0, # Discount factor
"lam":0.95, # Exponential weight for GAE. Probably leave at 0.95
"obs_size": 76, # dimension of observation
"output_size": 1, # scalar output
"hidden_layer_sizes": [256, 256],
"activations": ["ReLU", "ReLU"],
"is_actor": False, # Leave as false: distinguishes actor and critic networks,
"gradient_clipping": 0.5
}