from configs.config_class import Config

critic_configs = Config({
# INTERACTION CONFIGURATIONS:

# AGENT CONFIGURATIONS
"learning_rate": 1e-5,
"replay_buffer_size": 100000,
"loss_function": "mseloss",
"optimizer": "adam",
"gamma": 0.99,
"lam":0.95,
"batch": True,
"batch_size": 64,
"obs_size": 1040, # dimension of observation
"output_size": 1, # scalar output
"hidden_layer_sizes": [1024, 1024],
"activations": ["ReLU", "ReLU"],
"learning_frequency": 2,
"is_actor": False,
"entropy_coef": 0.01,
"use_shaped_reward": True
})

actor_configs = Config({
# AGENT CONFIGURATIONS
"learning_rate": 1e-5,
"replay_buffer_size": 100000,
#"loss_function": "mseloss", # We are performing gradient ascent, so we will need to fix this
"optimizer": "adam",
"gamma": 0.99,
"batch": True,
"batch_size": 64,
"obs_size": 1040, # dimension of observation
"output_size": 6, # dimension of action space
"hidden_layer_sizes": [1024, 1024],
"activations": ["ReLU", "ReLU"],
"learning_frequency": 2,
"c": 0.3, # clipping range
"is_actor": True,
"entropy_coef": 0.01,
"use_shaped_reward": True
})