import gymnasium as gym
import numpy as np
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS


import json
from typing import Dict, Any


# Source: https://github.com/Stanford-ILIAD/PantheonRL/blob/master/overcookedgym/overcooked.py
class OvercookedGymWrapper(gym.Env):
    def __init__(self, layout_name, ego_agent_idx=0, baselines=False):
        """
        base_env: OvercookedEnv
        featurize_fn: what function is used to featurize states returned in the 'both_agent_obs' field
        """
        super(gym.Env, self).__init__()

        DEFAULT_ENV_PARAMS = {
            "horizon": 500
        }
        params_to_overwrite = {
            "rew_shaping_params": {
                "PLACEMENT_IN_POT_REW": 3,
                "DISH_PICKUP_REWARD": 3,
                "SOUP_PICKUP_REWARD": 5,
                "DISH_DISP_DISTANCE_REW": 0, 
                "POT_DISTANCE_REW": 0,
                "SOUP_DISTANCE_REW": 0,
            }
        }



        self.mdp = OvercookedGridworld.from_layout_name(layout_name=layout_name, params_to_overwrite=params_to_overwrite)
        mlp = MediumLevelActionManager.from_pickle_or_compute(self.mdp, NO_COUNTERS_PARAMS, force_compute=False)

        self.featurize_fn = lambda x: self.mdp.featurize_state(x, mlp)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, **DEFAULT_ENV_PARAMS)

        if baselines: np.random.seed(0)

        self.observation_space = self._setup_observation_space()
        self.lA = len(Action.ACTION_TO_CHAR)
        self.action_space  = gym.spaces.Discrete( self.lA )
        self.ego_agent_idx = ego_agent_idx
        self.multi_reset()
        
    def _setup_observation_space(self):
        dummy_state = self.mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_state)[0].shape
        high = np.ones(obs_shape, dtype=np.float32)  # max(self.mdp.soup_cooking_time, self.mdp.num_items_for_soup, 5)

        return gym.spaces.Box(-high, high, dtype=np.float64)

    def multi_step(self, ego_action, alt_action):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format
            encoded as an int

        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        ego_action, alt_action = Action.INDEX_TO_ACTION[ego_action], Action.INDEX_TO_ACTION[alt_action]
        if self.ego_agent_idx == 0:
            joint_action = (ego_action, alt_action)
        else:
            joint_action = (alt_action, ego_action)

        next_state, reward, done, info = self.base_env.step(joint_action)
        # try:
        #     if(info['episode'] is not None):
        #         print(info)
        # except:
        #     j = 1
        # reward shaping
        rew_shape = info['shaped_r_by_agent']
        reward = np.sum(rew_shape) + reward
        #print(info)
        #print(self.base_env.mdp.state_string(next_state))
        ob_p0, ob_p1 = self.featurize_fn(next_state)
        if self.ego_agent_idx == 0:
            ego_obs, alt_obs = ob_p0, ob_p1
        else:
            ego_obs, alt_obs = ob_p1, ob_p0

        return ego_obs, alt_obs, reward, reward, done, {}#info#info

    def multi_reset(self):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset()
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)
        if self.ego_agent_idx == 0:
            ego_obs, alt_obs = ob_p0, ob_p1
        else:
            ego_obs, alt_obs = ob_p1, ob_p0

        return (ego_obs, alt_obs)

    def render(self, mode='human', close=False):
        pass

    def state_to_json(self, state: np.ndarray) -> str:
        """
        Convert a 96-element Overcooked observation NumPy array into a structured JSON string.
        """
        state = np.asarray(state).flatten()
        if state.size != 96:
            raise ValueError(f"Expected state of length 96, got {state.size}")

        # Feature definitions
        item_names = ["onion", "tomato", "dish", "soup", "serving_area", "empty_counter"]
        orientations = ["up", "right", "down", "left"]
        objects = ["onion", "soup", "dish", "tomato"]

        def parse_player_feats(feat: np.ndarray) -> Dict[str, Any]:
            idx = 0

            # Orientation one-hot
            orient_vec = feat[idx:idx+4]
            orientation = orientations[int(np.argmax(orient_vec))] if np.any(orient_vec == 1) else None
            idx += 4

            # Object held one-hot
            obj_vec = feat[idx:idx+4]
            obj = objects[int(np.argmax(obj_vec))] if np.any(obj_vec == 1) else None
            idx += 4

            # Distances to items
            dists = {}
            for name in item_names:
                dx, dy = int(feat[idx]), int(feat[idx+1])
                dists[name] = {"dx": dx, "dy": dy}
                idx += 2

            # Contents of closest soup
            num_onions = int(feat[idx]); num_tomatoes = int(feat[idx+1])
            idx += 2

            # Pot features (for two pots)
            pots = []
            for _ in range(2):
                exists = bool(feat[idx]); idx += 1

                status_keys = ["empty", "full", "cooking", "ready"]
                status_vals = feat[idx:idx+4]; idx += 4
                status = {k: bool(v) for k, v in zip(status_keys, status_vals)}

                num_onions_p = int(feat[idx]); num_tomatoes_p = int(feat[idx+1])
                idx += 2

                cook_time = int(feat[idx]); idx += 1

                dx_p = int(feat[idx]); dy_p = int(feat[idx+1]); idx += 2

                pots.append({
                    "exists": exists,
                    "status": status,
                    "num_onions": num_onions_p,
                    "num_tomatoes": num_tomatoes_p,
                    "cook_time": cook_time,
                    "dx": dx_p,
                    "dy": dy_p
                })

            # Wall indicators
            wall_dirs = ["up", "right", "down", "left"]
            walls = {d: bool(feat[idx + i]) for i, d in enumerate(wall_dirs)}
            idx += 4

            return {
                "orientation": orientation,
                "holding": obj,
                "distance_to_items": dists,
                "closest_soup_contents": {"num_onions": num_onions, "num_tomatoes": num_tomatoes},
                "pots": pots,
                "walls": walls
            }

        # Split state into components
        p0_feats = state[0:46]
        p1_feats = state[46:92]
        dist_to_other = {"dx": int(state[92]), "dy": int(state[93])}
        position = {"x": int(state[94]), "y": int(state[95])}

        # Build JSON
        data = {
            "player_0": parse_player_feats(p0_feats),
            "player_1": parse_player_feats(p1_feats),
            "distance_between_players": dist_to_other,
            "player_position": position
        }

        return json.dumps(data, indent=2)
