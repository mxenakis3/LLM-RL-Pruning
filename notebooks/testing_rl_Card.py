# IMPORT DESIRED INTERACTION CLASS AND CONFIGURATION
import sys
import os

# Get the parent directory (where `configs/` and `interactions/` are located)
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Add it to sys.path
sys.path.append(parent_dir)

# Now you should be able to import
from interactions import ppo_interaction as ppo
# from configs.interaction_configs.ppo_interaction_configs import ppo_interaction_config
# from configs.interaction_configs.ppo_interaction_configs import ppo_interaction_texasholdem_config
from configs.ppo_configs import ppo_interaction_config_texas, texas_holdem_config, actor_configs_texas, critic_configs_texas
from configs.llm_configs import texas_holdem_llm_agent_configs
# from configs.agent_configs.a_ppo_agents import actor_configs, critic_configs

ppo_interaction = ppo.PPO_interaction(interaction_configs=ppo_interaction_config_texas,
                    env_configs = texas_holdem_config,
                      actor_configs = actor_configs_texas,
                      critic_configs = critic_configs_texas,
                      llm_configs = texas_holdem_llm_agent_configs,
                  )

train_scores, trained_agents = ppo_interaction.train_multiagent()
print(train_scores)

ppo_interaction.testing_episodes = 100
test_scores = ppo_interaction.test_multiagent(trained_agents)
print(test_scores)