import gymnasium as gym
from pettingzoo.classic import texas_holdem_v4
from pettingzoo.classic.rlcard_envs import texas_holdem
from pettingzoo.utils.conversions import aec_to_parallel
import json
import numpy as np

def decay_prob(prob, decay_type, prob_start, prob_end, prob_decay_episodes, prob_decay_rate):
    if decay_type.lower() == "linear":
        prob = max(prob_end, (prob - (prob_start - prob_end)/(prob_decay_episodes)))
    else:
        prob = max(prob_end, (prob * prob_decay_rate))
    return prob

def get_render_mode(render_mode):
    """
    Helper function to get the correct render mode from config.
    """
    render_modes = {
        "human": "human",
        "none": None
    }
    return render_modes.get(render_mode.lower(), None)

def get_environment(config, train):
    """
    Helper function to load in the environment from config.
    The point of having this function is so that you can specify the environment in a config file without having to hard-code it into the Interaction file/notebook.
    Inputs:
    - config (dict): The config for the interaction class. This is usually a Python dictionary
    - train (Bool): This indicates whether we are calling the environment from a training or testing loop. We may wish to specify different training modes for each.
    """
    # Get the training mode specified by config
    if train:
        render_mode = get_render_mode(config.render_mode_train)
    else:
        render_mode = get_render_mode(config.render_mode_train)

    # Decide whether the environment is discrete or continuous
    continuous = config.continuous
    # Use elif statements to select the environment
    if config.env.lower() == "lunarlander-v3":

        return gym.make("LunarLander-v3", continuous=continuous, render_mode=render_mode)
    
    elif config.env.lower() == "texas_holdem_v4":
        return texas_holdem_v4.env(
            render_mode=render_mode
        )
    elif config.env.lower() == 'texas_holdem':
        return texas_holdem

def texas_holdem_state_to_json(arr):
    """
    Convert a (76,) numpy array into a JSON-ready dict for an LLM.

    Structure of arr:
      0–12   : Spades (one-hot over ranks A,2,…,K)
      13–25  : Hearts
      26–38  : Diamonds
      39–51  : Clubs
      52–56  : Chips raised in Round 1 (one-hot over 0–4)
      57–61  : Chips raised in Round 2
      62–66  : Chips raised in Round 3
      67–71  : Chips raised in Round 4
      72–75  : Opponent action (one-hot over Call,Raise,Fold,Check)
    """
    if arr.shape != (76,):
        raise ValueError(f"Expected shape (76,), got {arr.shape}")
    
    # rank labels
    ranks = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
    # suit blocks
    suit_info = {
        'spades':   (0, 13),
        'hearts':   (13, 13),
        'diamonds': (26, 13),
        'clubs':    (39, 13),
    }
    
    cards = {}
    for suit, (start, length) in suit_info.items():
        block = arr[start:start+length]
        # find all non-zero positions (in case of multiple cards)
        idxs = np.where(block > 0)[0]
        cards[suit] = [ranks[i] for i in idxs]
    
    # chips raised per round
    chips = []
    for r in range(4):
        start = 52 + r*5
        block = arr[start:start+5]
        # argmax gives the raised amount (0–4)
        chips.append(int(np.argmax(block)))
    
    # opponent action
    action_block = arr[72:76]
    action_list = ['Call', 'Raise', 'Fold', 'Check']
    action_idx = int(np.argmax(action_block))
    opponent_action = action_list[action_idx]
    
    payload = {
        'cards': cards,
        'chips_raised': {
            'round_1': chips[0],
            'round_2': chips[1],
            'round_3': chips[2],
            'round_4': chips[3],
        },
        'opponent_action': opponent_action
    }
    
    return json.dumps(payload)