
from configs.config_class import Config

texas_holdem_config = Config({
  # INTERACTION CONFIGURATIONS:
"env": "texas_holdem_v4",
"render_mode_train": "none", # "human" plays an animation, "None" skips the animation
"render_mode_test": "none", # "human" plays an animation, "None" skips the animation
"continuous": False, # Discrete, or continuous simulation -- OpenAI parameter})
})

ppo_interaction_config_texas = Config({
  # INTERACTION CONFIGURATIONS:
"training_episodes": 10,
"testing_episodes": 5,
"bot_type": "random", # "heuristic" or "random"
"c": 0.15, # Loss function clipping coefficient
"batch_size":128
, # Size of batch for SGD in update method.
"capacity": 20000, # Size of experience tuple (s, a, r, s_, ...) memory
"num_epochs": 4, # number of batches in each iteration of learning
"update_frequency": 4000, # how many steps are taken before an update occurs
"kap_start" : 0.5, # The probability of samping an action from the LLM agent at the first timestep
"kap_end": 0.001, # The min probability of sampling an action from the LLM agent
"kap_decay_episodes": 8, # For linear decay: The number of steps we are taking to decay kap. Each episode, kap decays by (kap_start - kap_finish)/ kap_decay_episodes.
"kap_decay_rate": 0.98, # For exponential decay: The rate of decay for kap, the probability that we sample from llm. kap = kap_start* kap_decay_rate^(episode_number)
"kap_decay_type": "linear", # Exponential or linear
})

actor_configs_texas = Config({
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
})

critic_configs_texas = Config({
# INTERACTION CONFIGURATIONS:

# AGENT CONFIGURATIONS
"learning_rate": 3e-4 ,
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
})

# c: a range over [0.75, 0.0]
# batch_size [any]
# capacity [any]
# "update_frequency": 128, # how many steps are taken before an update occurs
# learning rate (powers of 10 that we test)
# hidden layer sizes

# Linear kap decay
# 0.8 --> 0

# Optional - dropout (add in methods for agent get_model() or whatever)
# Activations - leaky ReLU

# First step - tune without llm
# second step - add in llm
