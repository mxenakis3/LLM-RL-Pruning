from configs.config_class import Config

ppo_interaction_config = Config({
  "env": "overcooked",
  "training_episodes": 5000,
  "testing_episodes": 5000,
  "render_mode_train": "None",
  "render_mode_test": "human",
  "batch_size": 64,
  "capacity": 4096,
  "gamma": 0.99,
  "k": 4,
  "num_epochs": 4,
  "update_frequency": 4096,
  "learning_rate": 5e-6,
  "c": 0.2
})
