import os
import yaml

from .wolfe_schlegel import WolfeSchlegel
from .muller_brown import MullerBrown
from .constant import Constant
from .lennard_jones import LennardJones

potential_dict = {
    "wolfe_schlegel" : WolfeSchlegel,
    "muller_brown" : MullerBrown,
    "constant" : Constant,
    "lennard_jones" : LennardJones
}

def import_potential_config(
        name,
        tag="",
        dir="./src/potentials/configs/",
        is_expected=False
    ):
    filename = name
    filename += f"_{tag}.yaml" if tag != "" else ".yaml"
    address = os.path.join(dir, filename)

    if os.path.exists(address):
        with open(address, 'r') as file:
            loaded_yaml = yaml.safe_load(file)
        return loaded_yaml
    elif is_expected:
        raise ImportError(f"Cannot find required file {address}")
    else:
        ImportWarning(f"Cannot find file {address}, running without it")
        return {}

def get_potential(
        potential,
        tag="",
        config_dir="./src/potentials/configs/",
        expect_config=False,
        **kwargs
    ):
    assert potential.lower() in potential_dict
    config_filename = potential
    config_filename += f"_{tag}.yaml" if tag != "" else ".yaml"
    print(potential)
    config = import_potential_config(
        potential, tag, dir=config_dir, is_expected=expect_config
    )
    return potential_dict[potential](**config, **kwargs)