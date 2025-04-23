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

def get_environment(config, train):
    """
    Helper function to load in the environment from config.
    The point of having this function is so that you can specify the environment in a config file without having to hard-code it into the Interaction file/notebook.
    Inputs:
    - config (dict): The config for the interaction class. This is usually a Python dictionary
    - train (Bool): This indicates whether we are calling the environment from a training or testing loop. We may wish to specify different training modes for each.
    """
    if config.env.lower() == "overcooked":
        render = config.render_mode_train if train else config.render_mode_test
        render = False if render == "None" else True
        return OvercookedGymWrapper(layout_name="cramped_room")
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
    