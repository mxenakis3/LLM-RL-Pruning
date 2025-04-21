import gymnasium as gym
from overcooked_wrapper import OvercookedGymWrapper

def get_render_mode(render_mode):
    """
    Helper function to get the correct render mode from config.
    """
    render_modes = {
        "human": "human",
        "none": None
    }
    return render_modes.get(render_mode.lower(), None)

def get_environment(config, train=True):
    render = config.render_mode_train if train else config.render_mode_test
    render = False if render == "None" else True
    return OvercookedGymWrapper(layout_name="cramped_room")
