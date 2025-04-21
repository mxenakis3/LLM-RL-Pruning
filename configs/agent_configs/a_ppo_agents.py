from configs.config_class import Config

critic_configs = Config({
# INTERACTION CONFIGURATIONS:

# AGENT CONFIGURATIONS
"learning_rate": 5e-6,
"replay_buffer_size": 100000,
"loss_function": "mseloss",
"optimizer": "adam",
"gamma": 0.99,
"lam":0.95,
"batch": True,
"batch_size": 64,
"obs_size": 128, # dimension of observation
"output_size": 1, # scalar output
"hidden_layer_sizes": [512, 512],
"activations": ["ReLU", "ReLU"],
"learning_frequency": 4,
"is_actor": False
"entropy_coef": 0.01
})

actor_configs = Config({
# AGENT CONFIGURATIONS
"learning_rate": 5e-6,
"replay_buffer_size": 100000,
#"loss_function": "mseloss", # We are performing gradient ascent, so we will need to fix this
"optimizer": "adam",
"gamma": 0.99,
"batch": True,
"batch_size": 64,
"obs_size": 128, # dimension of observation
"output_size": 6, # dimension of action space
"hidden_layer_sizes": [512, 512],
"activations": ["ReLU", "ReLU"],
"learning_frequency": 4,
"c": 0.2, # clipping range
"is_actor": True
"entropy_coef": 0.01
})