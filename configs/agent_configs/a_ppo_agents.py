from configs.config_class import Config

critic_configs = Config({
# INTERACTION CONFIGURATIONS:

# AGENT CONFIGURATIONS
"learning_rate": .0001,
"replay_buffer_size": 100000,
"loss_function": "mseloss",
"optimizer": "adam",
"gamma": 0.99,
"batch": True,
"batch_size": 64,
"obs_size": 8, # dimension of observation
"output_size": 1, # scalar output
"hidden_layer_sizes": [128, 128],
"activations": ["ReLU", "ReLU"],
"target_network": True,
"target_update_frequency": 64,
"learning_frequency": 4,
"is_actor": False
})

actor_configs = Config({
# AGENT CONFIGURATIONS
"learning_rate": .0001,
"replay_buffer_size": 100000,
"loss_function": "mseloss", # We are performing gradient ascent, so we will need to fix this
"optimizer": "adam",
"gamma": 0.99,
"batch": True,
"batch_size": 64,
"obs_size": 8, # dimension of observation
"output_size": 4, # dimension of action space
"hidden_layer_sizes": [128, 128],
"activations": ["ReLU", "ReLU"],
"target_network": True,
"target_update_frequency": 64,
"learning_frequency": 4,
"c": 0.2, # clipping range
"is_actor": True
})