from configs.config_class import Config
ppo_interaction_config = Config({
  # INTERACTION CONFIGURATIONS:
"env": "lunarlander-v3",
"learning_rate": .0003,
"c": 0.2,
"training_episodes": 100,
"render_mode_train": "None", # "human" plays an animation, "None" skips the animation
"render_mode_test": "human", # "human" plays an animation, "None" skips the animation
"continuous": False, # Discrete, or continuous simulation -- OpenAI parameter
"testing_episodes": 100,
"batch_size":64,
"capacity": 2048,
"gamma": 0.99,
"k": 4, # number of batches in each iteration of learning
"num_epochs": 4,
"update_frequency": 2048
# # Eps decay: decay for greedy vs. nongreedy exploration
# "eps_start": 1.0,
# "eps_end": 0.001,
# "eps_decay_episodes": 750, # Number of episodes of decay (if linear decay)
# "eps_decay_rate": 0.99, # rate of exponential decay (if exponential decay)
# "decay_type": "exponential", # linear or exponential
})

ppo_interaction_conf = Config({
  # INTERACTION CONFIGURATIONS:
# "env": "texas_holdem_v4",
"learning_rate": .0003,
"c": 0.2, 
"training_episodes": 100,
# "render_mode_train": "None", # "human" plays an animation, "None" skips the animation
# "render_mode_test": "human", # "human" plays an animation, "None" skips the animation
"continuous": False, # Discrete, or continuous simulation -- OpenAI parameter
"testing_episodes": 100,
"batch_size":64,
"capacity": 2048,
"gamma": 0.99,
"k": 4, # number of batches in each iteration of learning
"num_epochs": 4,
"update_frequency": 128
# # Eps decay: decay for greedy vs. nongreedy exploration
# "eps_start": 1.0,
# "eps_end": 0.001,
# "eps_decay_episodes": 750, # Number of episodes of decay (if linear decay)
# "eps_decay_rate": 0.99, # rate of exponential decay (if exponential decay)
# "decay_type": "exponential", # linear or exponential
})

texas_holdem_config = Config({
  # INTERACTION CONFIGURATIONS:
"env": "texas_holdem_v4",
"render_mode_train": "None", # "human" plays an animation, "None" skips the animation
"render_mode_test": "human", # "human" plays an animation, "None" skips the animation
})





ppo_interaction_texasholdem_config = Config({
  # INTERACTION CONFIGURATIONS:
"env": "texas_holdem_v4",
"learning_rate": .0003,
"c": 0.2,
"training_episodes": 100,
"render_mode_train": "None", # "human" plays an animation, "None" skips the animation
"render_mode_test": "human", # "human" plays an animation, "None" skips the animation
"continuous": False, # Discrete, or continuous simulation -- OpenAI parameter
"testing_episodes": 100,
"batch_size":64,
"capacity": 2048,
"gamma": 0.99,
"k": 4, # number of batches in each iteration of learning
"num_epochs": 4,
"update_frequency": 128
# # Eps decay: decay for greedy vs. nongreedy exploration
# "eps_start": 1.0,
# "eps_end": 0.001,
# "eps_decay_episodes": 750, # Number of episodes of decay (if linear decay)
# "eps_decay_rate": 0.99, # rate of exponential decay (if exponential decay)
# "decay_type": "exponential", # linear or exponential
})


