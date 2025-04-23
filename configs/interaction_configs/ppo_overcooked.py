from configs.config_class import Config

ppo_interaction_config = Config({
  "env": "overcooked",
  "training_episodes": 500,
  "testing_episodes": 500,
  "render_mode_train": "None",
  "render_mode_test": "human",
  "batch_size": 64,
  "capacity": 4096,
  "gamma": 0.99,
  "k": 1,
  "num_epochs": 250,
  "update_frequency": 5,
  "learning_rate": 5e-6,
  "c": 0.1
})
