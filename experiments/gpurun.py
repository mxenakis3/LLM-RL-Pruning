# IMPORT DESIRED INTERACTION CLASS AND CONFIGURATION
import sys
import os

# Get the parent directory (where `configs/` and `interactions/` are located)
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Add it to sys.path
sys.path.append(parent_dir)

# Now you should be able to import
from interactions import ppo_interaction as ppo
from configs.interaction_configs.ppo_interaction_overcooked import ppo_interaction_config
from configs.agent_configs.a_ppo_agents_overcooked import actor_configs, critic_configs


trainer = ppo.PPO_interaction(ppo_interaction_config, actor_configs, critic_configs)

train_scores, policy = trainer.train()
test_scores = trainer.test()
#test_scores = trainer.test()
import matplotlib.pyplot as plt

fig, (ax1, ax2)  = plt.subplots(1, 2)
ax1.set_title("Training Scores")
ax1.set_xlabel("Episodes")
ax1.set_ylabel("Score")
ax1.plot(train_scores)
ax1.grid(True)

ax2.set_title("Testing Scores")
ax2.set_xlabel("Episodes")
ax2.set_ylabel("Score")
ax2.plot(test_scores)
ax2.grid(True)
plt.tight_layout()
plt.savefig("PPO_Train_Test")
plt.show()
