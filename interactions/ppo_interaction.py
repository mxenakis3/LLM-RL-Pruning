import torch as torch
from agents.ppo_agents import PPOActorNetwork, PPOCriticNetwork
from agents.llm_chain_of_thought_agent import Chain_of_Thought
from tqdm import tqdm
import torch.distributions as dist
from replay_buffer import PPOReplayBuffer
import numpy as np
import random
from utils import get_environment, get_render_mode, decay_prob, texas_holdem_state_to_json
from torch.distributions import Categorical

class PPO_interaction:
    def __init__(self, interaction_configs, env_configs, actor_configs, critic_configs, llm_configs):
        # load parallel TexasHoldem or LunarLander, etc.
        self.env      = get_environment(env_configs, train=True)
        self.test_env = get_environment(env_configs, train=False)

        # This variabe will be used in train, test to dispatch training to the appropriate method.
        self.env_name = env_configs.env

        # Set up additional parameters for texas holdem
        if self.env_name.lower() == "texas_holdem_v4":
            self.env.reset()
            self.agents = list(self.env.possible_agents)

            # Only these are the “real” players
            self.player_agents = [
                a for a in self.env.possible_agents
                if a.startswith("player")
            ]
            first_obs, _, _, _, _ = self.env.last()
            first_agent = self.env.agent_selection

            obs_vec = first_obs["observation"]

            act_space = self.env.action_space(first_agent)
            if hasattr(act_space, "n"):
                self.action_size = act_space.n
            elif hasattr(act_space, "nvec"):
                self.action_size = int(act_space.nvec[0])
            else:
                raise ValueError("Unsupported action space")
        
            self.obs_size = obs_vec.shape[0] + self.action_size
            print(f"self obs size: {self.obs_size}")



        self.num_episodes = interaction_configs.training_episodes
        self.testing_episodes = interaction_configs.testing_episodes
        self.update_frequency = interaction_configs.update_frequency
        self.critic = PPOCriticNetwork(obs_size = self.obs_size,
                                 action_size=self.action_size,
                                 critic_config=critic_configs)
        self.policy = PPOActorNetwork(obs_size = self.obs_size,
                                 action_size=self.action_size,
                                 actor_config=actor_configs)
        self.old_policy = PPOActorNetwork(obs_size = self.obs_size,
                                 action_size=self.action_size,
                                 actor_config=actor_configs)
        self.old_policy.load_state_dict(self.policy.state_dict())

        # Initialize hyperparemeters
        self.c = interaction_configs.c # clipping factor (0 <= x <= 1) for PPO
        self.batch_size = interaction_configs.batch_size
        self.num_epochs = interaction_configs.num_epochs # number of training epochs

        # Initialize memory
        self.memory = PPOReplayBuffer(capacity=interaction_configs.capacity,
                                   batch_size=self.batch_size)
        
        # Init LLM
        self.llm_configs = llm_configs
        self.llm_agent = Chain_of_Thought(config=llm_configs)
        self.llm_system_message = self.llm_configs.system_message["content"] # This string needs to provide information about the state

        # LLM decay parameters
        self.kap_start, self.kap_end  = interaction_configs.kap_start, interaction_configs.kap_end
        self.kap_decay_episodes, self.kap_decay_rate = interaction_configs.kap_decay_episodes, interaction_configs.kap_decay_rate
        self.kap_decay_type = interaction_configs.kap_decay_type

        # init kap
        self.kap = self.kap_start

    def act(self, state):
        with torch.no_grad():
            action_probs = self.old_policy.model(state)
        distribution = dist.Categorical(action_probs)
        action = distribution.sample()
        logprob = distribution.log_prob(action)
        return action.item(), logprob.item()

    def act_multiagent(self, state_vec: np.ndarray, action_mask: np.ndarray):
        """
        Multi-agent action for PettingZoo environments
        """
        with torch.no_grad():
            # old_policy.model returns probabilities [batch=1, action_size]
            probs = self.old_policy.model(torch.Tensor(state_vec).unsqueeze(0))
            probs = probs.squeeze(0)  # -> [action_size]

        mask = torch.Tensor(action_mask)  # float tensor of 0/1
        masked = probs * mask             # zero out illegal
        total = masked.sum().item()

        if total == 0.0:
            # fallback: uniform over legal moves
            legal_count = mask.sum().item()
            masked = mask / legal_count
        else:
            masked = masked / total

        dist = Categorical(masked)
        a = dist.sample().item()
        return a, dist.log_prob(torch.tensor(a)).item()

    def update(self):
        """
        Update Actor and Critic networks in PPO.
        """
        # First shuffle the data in memory
        actor_losses, critic_losses = [], []
        mem_list = list(self.memory.buffer)
        random.shuffle(mem_list) # shuffled list of tuples. s, s_ are numpy ndarrays, a, r, dones are floats or bools.

        # Get vectors for states, actions, rewards, s_,  dones etc.
        s_vector = torch.Tensor(np.array([t[0] for t in mem_list])) #shape: [capacity, obs_size]
        a_vector = torch.Tensor([t[1] for t in mem_list]) #shape: [capacity, 1]
        logprobs_vector = torch.Tensor([t[2] for t in mem_list]) #shape: [capacity, 1]
        advantages_vector = torch.Tensor([t[3] for t in mem_list]) #shape: [capacity, 1]
        returns_vector = torch.Tensor([t[4] for t in mem_list]) #shape: [capacity, 1]

        # Normalize advantages vector
        advantages_vector = (advantages_vector - advantages_vector.mean()) / (advantages_vector.std() + 1e-8)

        # Update gradients
        for i in range(self.num_epochs):
            for idx in range(0, s_vector.shape[0], self.batch_size):

                batch_s = s_vector[idx: idx+ self.batch_size] #shape: [batch_size,]
                batch_a = a_vector[idx: idx+ self.batch_size]
                batch_logprobs = logprobs_vector[idx: idx+ self.batch_size]
                batch_advantages = advantages_vector[idx: idx+self.batch_size]
                batch_returns = returns_vector[idx: idx+self.batch_size]

                # Get new probabilities. We will backpropagate on these.
                new_probs = self.policy.model(batch_s)
                new_prob_distribution = dist.Categorical(new_probs) # Do this so we can get the entropy of the prob dist
                new_logprobs = new_prob_distribution.log_prob(batch_a) # gets the logprobs of the actions that were actually taken
                entropy = new_prob_distribution.entropy().mean()

                ratios = torch.exp(new_logprobs - batch_logprobs.detach())

                # Actor loss
                surr1 = ratios.unsqueeze(-1) * batch_advantages
                surr2 = torch.clamp(ratios.unsqueeze(-1), 1-self.c, 1+self.c) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() + 0.01 * entropy

                # Critic loss
                values_pred = self.critic.model(batch_s)
                critic_loss = 0.5 * (values_pred - batch_returns).pow(2).mean()

                # Update critic
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

                # Update actor
                self.policy.optimizer.zero_grad()
                actor_loss.backward()
                self.policy.optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        print(f"[PPO UPDATE] avg_actor_loss={np.mean(actor_losses):.4f}, "
              f"avg_critic_loss={np.mean(critic_losses):.4f}")

        # clear the buffer and fix old policy
        self.memory.clear()
        self.old_policy.load_state_dict(self.policy.state_dict())

    def compute_gae(self, rewards, values, dones, gamma, lam):
        """
        Computes the Generalized Advantage Estimate for the TD targets used in update. 
        The "advantage" for a given action selection is the difference between the expected value of taking that action given a particular state, and the expected value of selecting a random action for that state.

        The GAE computes this as a geometrically weighted average over the expected rewards from all future states. This is a common practice in PPO implementations.

        len(values) = len(rewards) + 1
        """
        T = len(rewards)
        advantages  = torch.zeros(T)
        gae = 0
        for t in reversed(range(T)):
            delta = rewards[t] + (gamma*values[t+1] * (1 - dones[t])) - values[t]
            gae = delta + gamma*lam*(1- dones[t])*gae
            advantages[t] = gae
        returns = advantages + torch.Tensor(values[:-1])
        return advantages, returns

    def train(self):
        """
        Read environment name, and dispatch training to correct method.
        Inputs: None
        Outputs: None
        """
        if self.env.lower() == "texas_holdem_v4":
            self.train_multiagent()
        elif self.env.lower() == "lunarlander-v3":
            self.train_single_agent()

    def get_llm_action(self, state_vec, action_mask=None, moves_dict=None):
        """
        Returns action and logprob given state and legal actions
        
        """
        with torch.no_grad():
            action_logits = self.policy.model(torch.Tensor(state_vec).unsqueeze(0))
            action_probs = torch.softmax(action_logits, dim=-1).squeeze().numpy()

            # Update state message
            if action_mask is not None:
                # Get set of legal moves
                print(type(action_mask))
                legal_actions = [moves_dict[i] for i, mask in enumerate(action_mask) if mask == 1]

                # Apply action mask (set invalid actions to 0 probability)
                masked_probs = action_probs * action_mask
                masked_probs = masked_probs / masked_probs.sum()  # Renormalize           

                # Sample action from llm. (Assumes Texas Holdem is the current environment)
                state_message = self.llm_system_message.format(**{"state" : texas_holdem_state_to_json(state_vec), "legal_actions": legal_actions})
                self.llm_agent.messages.append(state_message)
                action = self.llm_agent() # Automatically clears messages
                lp = np.log(masked_probs[action] + 1e-10)  # Small epsilon to avoid log(0)

            else:
                state_message = self.llm_system_message.format(**{"state" : texas_holdem_state_to_json(state_vec), "legal_actions": legal_actions})
                action = np.random.randint(0, self.action_size - 1) # Dummy for now
                lp = np.log(action_probs[action] + 1e-10)


    def train_multiagent(self):
        """
        Training PPO agent against one random agent (for now)
        Returns:
            - train_scores: The PPO agent's reward after each hand
            - policy: The trained network type: (PPONetwork) (see agents/PPONetwork)
        """
        train_scores = []
        # # for debugging:
        moves_dict = {0: 'call', 1: 'raise', 2: 'fold', 3: 'check'}
        for ep in tqdm(range(self.num_episodes)):
            # print(f"NEW EPISODE")
            self.env.reset()

            # Store trajectories for each agent in dictionary (needs to be serialized for GAE calculation)
            agent_trajectories = {agent: {"states": [], "actions":[], "logps":[], "rewards": [], "dones":[], "values":[]} for agent in self.player_agents}

            # We will one-hot encode the action of the last agent. At start, no one has acted, so last_action = 0
            last_action = np.array([0]*self.action_size)

            for agent in self.env.agent_iter(): # Function that iterates through agents dynamically based on state information. 
                # NOTE: If an agent folds, then self.env.agent_iter() iterates from agent_0 to agent_1

                obs, reward, termination, truncation, info = self.env.last() # info from last step. 
                # print(f"{agent} self.env.last(): Terminated: {termination}, Reward: {reward}, last ation: {last_action}")

                if termination or truncation:
                    # print(f"{agent}: last reward: {reward}")
                    agent_trajectories[agent]["values"].append(0.0) # A terminal state is reached, and v=0
                    agent_trajectories[agent]["rewards"].append(reward) # Append reward for the round
                    action = None 
                    last_action = np.array([0]*self.action_size)
                    self.env.step(action)
                    
                    # summary = {agent: {key: len(value) for key, value in data.items()} for agent, data in agent_trajectories.items()}
                    # print(f"Last round someone folded. {summary}")
                    continue # Need this line to iterate to player_1 after game ends. If no continue, only player_0 receives rewards

                else:
                    state_vec   = obs["observation"]
                    state_vec = np.concatenate((state_vec, last_action), axis=0)
                    action_mask = obs["action_mask"]
                    v = self.critic.model(torch.Tensor(state_vec).unsqueeze(0)).item()

                    if agent == 'player_0':
                        # With probability kap, select action with llm agent
                        p = np.random.uniform(0, 1)
                        if p <= self.kap:
                            action, lp = self.get_llm_action(state_vec, action_mask, moves_dict)
                        else:
                            action, lp = self.act_multiagent(state_vec, action_mask)

                    else: # Random agent
                        # Get action probabilities from the policy network (even if choosing randomly)
                        with torch.no_grad():
                            action_logits = self.policy.model(torch.Tensor(state_vec).unsqueeze(0))
                            action_probs = torch.softmax(action_logits, dim=-1).squeeze().numpy()
                        
                        # Apply action mask (set invalid actions to 0 probability)
                        masked_probs = action_probs * action_mask
                        masked_probs = masked_probs / masked_probs.sum()  # Renormalize
                        
                        # Randomly sample an action from valid ones
                        valid_indices = np.where(action_mask == 1)[0]
                        action = np.random.choice(valid_indices, p=masked_probs[valid_indices])
                        
                        # Compute log probability of the chosen action
                        lp = np.log(masked_probs[action] + 1e-10)  # Small epsilon to avoid log(0)
                    
                    last_action = np.eye(self.action_size)[action]
                    agent_trajectories[agent]["states"].append(state_vec)
                    agent_trajectories[agent]["actions"].append(action)
                    agent_trajectories[agent]["logps"].append(lp)
                    agent_trajectories[agent]["values"].append(v)
                    agent_trajectories[agent]["rewards"].append(reward) # Append reward for the round
                    agent_trajectories[agent]["dones"].append(False)

                    # print(f"{agent} takes action {moves_dict[action]}. New 'last_action' was {last_action}")

                # summary = {agent: {key: len(value) for key, value in data.items()} for agent, data in agent_trajectories.items()}
                # print(f"After action: {summary}")
                self.env.step(action) # Automatically terminates when an agent folds

            # Truncate the dones and rewards from the start state (we need to stagger rewards for PPO)
            for agent, traj in agent_trajectories.items():
                if len(traj["actions"]) > 0:
                    traj["rewards"] = traj["rewards"][1:] # If the agent got to act, rewards are staggered. 
                    # ie. if player_0 folds on turn 1, agent_1 does not act. Therefore they do not learn.
                    # player_0's rewards[0] is the reward conferred on reset, which==0. Clip this reward so that the rewards come 'after' the action (needs to work with GAE)
                    # if player_0 did not fold on first turn, player_1 had a chance to act, but his first reward comes from 
                    advantages, returns = self.compute_gae(
                        traj["rewards"], traj["values"], traj["dones"],
                        self.critic.gamma, self.critic.lam
                    )
                    # Now add this to memory
                    for s, a, lp, adv, ret in zip(traj["states"], traj["actions"], traj["logps"], advantages, returns):
                        self.memory.push(s, a, lp, adv.item(), ret.item())
              
            
            train_scores.append(agent_trajectories["player_0"]["rewards"][-1]) # Append the payoff of the current hand to train scores

            self.kap = decay_prob(self.kap, self.kap_decay_type, self.kap_start, self.kap_end, self.kap_decay_episodes, self.kap_decay_rate) # decay kap

            if len(self.memory) >= self.update_frequency:
                self.update()

        return train_scores, self.policy

    def test_multiagent(self, policy):
        """
        Testing PPO agent against one random agent (for now)
        Returns:
            - test_scores: The PPO agent's reward after each hand
            - policy: The trained network type: (PPONetwork) (see agents/PPONetwork)
        """
        test_scores = []
        for ep in tqdm(range(self.testing_episodes)):
            # print(f"NEW EPISODE")
            self.env.reset()

            # Store trajectories for each agent in dictionary (needs to be serialized for GAE calculation)
            agent_trajectories = {agent: {"states": [], "actions":[], "logps":[], "rewards": [], "dones":[], "values":[]} for agent in self.player_agents}

            # We will one-hot encode the action of the last agent. At start, no one has acted, so last_action = 0
            last_action = np.array([0]*self.action_size)

            for agent in self.env.agent_iter(): # Function that iterates through agents dynamically based on state information. 
                # NOTE: If an agent folds, then self.env.agent_iter() iterates from agent_0 to agent_1

                obs, reward, termination, truncation, info = self.env.last() # info from last step. 
                # print(f"{agent} self.env.last(): Terminated: {termination}, Reward: {reward}, last ation: {last_action}")

                if termination or truncation:
                    # print(f"{agent}: last reward: {reward}")
                    agent_trajectories[agent]["values"].append(0.0) # A terminal state is reached, and v=0
                    agent_trajectories[agent]["rewards"].append(reward) # Append reward for the round
                    action = None 
                    last_action = np.array([0]*self.action_size)
                    self.env.step(action)
                    continue # Need this line to iterate to player_1 after game ends. If no continue, only player_0 receives rewards

                else:
                    state_vec   = obs["observation"]
                    state_vec = np.concatenate((state_vec, last_action), axis=0)
                    action_mask = obs["action_mask"]

                    if agent == 'player_0':
                        action, lp = self.act_multiagent(state_vec, action_mask)
                    else:
                        # Get action probabilities from the policy network (even if choosing randomly)
                        with torch.no_grad():
                            action_logits = policy.model(torch.Tensor(state_vec).unsqueeze(0))
                            action_probs = torch.softmax(action_logits, dim=-1).squeeze().numpy()
                        
                        # Apply action mask (set invalid actions to 0 probability)
                        masked_probs = action_probs * action_mask
                        masked_probs = masked_probs / masked_probs.sum()  # Renormalize
                        
                        # Randomly sample an action from valid ones
                        valid_indices = np.where(action_mask == 1)[0]
                        action = np.random.choice(valid_indices, p=masked_probs[valid_indices])
                    
                    last_action = np.eye(self.action_size)[action]
                    agent_trajectories[agent]["rewards"].append(reward) # Append reward for the round

                self.env.step(action) # Automatically terminates when an agent folds            

            test_scores.append(agent_trajectories["player_0"]["rewards"][-1]) # Append the payoff of the current hand to train scores

        return test_scores

    def train_single_agent(self):
        """
        Uses PPO to train gymnasium agent (LunarLander for now)
        """
        train_scores = []
        step = 0
        for e in tqdm(range(self.num_episodes)):
            states, actions, rewards, logprobs, dones, values = [], [], [], [], [], []

            s, _ = self.env.reset()
            s_tensor = torch.Tensor(s)
            done = False
            episode_score = 0
            while not done:
                # Get action and logprob associated with action (old policy)
                a, logprob = self.act(s_tensor)

                # Get value associated with state
                v = self.critic.model(s_tensor)

                # Step in environment
                s_, r, terminated, truncated, _ = self.env.step(a)
                done = terminated or truncated

                # Append to states
                states.append(s)
                actions.append(a)
                rewards.append(r)
                logprobs.append(logprob)
                dones.append(done)
                values.append(v)

                # self.memory.push(s, a, r, s_, done)
                s = s_
                episode_score += r
                step += 1

            # End of trajectory
            # Append the value of the current state for the GAE calculation
            values.append(torch.tensor(0.0))

            # Compute Generalized Advantage Estimation
            advantages, returns = self.compute_gae(rewards, values, dones, self.critic.gamma, self.critic.lam)

            for t in range(len(states)):
                self.memory.push(states[t], actions[t], logprobs[t], advantages[t], returns[t])

            if len(self.memory) >= self.update_frequency:
                self.update()

            train_scores.append(episode_score)
        return train_scores, self.policy

    def test_single_agent(self):
        test_scores = []
        for e in range(self.testing_episodes):
            s, _ = self.env.reset()
            s_tensor = torch.Tensor(s)
            done = False
            episode_score = 0
            while not done:
                a, logprob = self.act(s_tensor)
                s_, r, terminated, truncated, _ = self.env.step(a)
                done = terminated or truncated
                episode_score += r
                s = s_
            test_scores.append(episode_score)
        return test_scores

