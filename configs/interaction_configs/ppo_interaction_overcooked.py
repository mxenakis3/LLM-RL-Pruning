from configs.config_class import Config

ppo_interaction_config = Config({
  "env": "overcooked",
  "training_episodes": 150,
  "testing_episodes": 20,
  "render_mode_train": "None",
  "render_mode_test": "human",
  "batch_size": 64,
  "capacity": 8192,
  "gamma": 0.99,
  "k": 4,
  "num_epochs": 2500,
  "update_frequency": 5,
  "learning_rate": 1e-5,
  "c": 0.3,
  "kap_start": 0.7,
  "kap_end": 0.001,
  "kap_decay_episodes": 7,
  "kap_decay_rate": 0.98,
  "kap_decay_type": "linear", # linear or exponential
})