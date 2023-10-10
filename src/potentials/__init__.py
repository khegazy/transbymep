import os
import json

from .wolfe_schlegel import wolfe_schlegel
from .muller_brown import muller_brown

potential_dict = {
    "wolfe_schlegel" : wolfe_schlegel,
    "muller_brown" : muller_brown,
}

def get_potential(
        potential, config_path="./src/potentials/configs/", expect_config=False
    ):
    print(potential)
    assert potential.lower() in potential_dict
    config_fileName = os.path.join(config_path, potential.lower() + ".json")
    if os.path.exists(config_fileName):
        with open(config_fileName) as file:
            config = json.load(file)
    elif expect_config:
        raise ImportError(
            "Cannot find the requred config file {config_fileName}"
        )
    else:
        raise ImportWarning(
            "Cannot find the config {config_fileName}, running without it."
        )
        config = None 
    return potential_dict[potential], config