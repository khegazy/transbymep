import os
import yaml
import optax  # https://github.com/deepmind/optax
import equinox as eqx

from .losses import get_loss
from .integrator import ODEintegrator
from .update_minima import MinimaUpdate
from .initialize_path import randomly_initialize_path

optimizer_dict = {
    "sgd" : optax.sgd,
    "adabelief" : optax.adabelief,
    "adam" : optax.adam
}

def import_optimizer_config(
        name, 
        path_type,
        potential_type,
        tag="",
        dir="./src/optimizations/configs/",
        is_expected=True
    ):
    filename = f"{name}_{potential_type}_{path_type}"
    filename += f"_{tag}.yaml" if tag != "" else ".yaml"
    address = os.path.join(dir, filename)
    print(f"Importing optimizer config {address}")
    if os.path.exists(address):
        with open(address, 'r') as file:
            loaded_yaml = yaml.safe_load(file)
        return loaded_yaml
    elif is_expected:
        raise ImportError(f"Cannot find required file {address}")
    else:
        ImportWarning(f"Cannot find file {address}, still running")
        return {}

def get_optimizer(
        name,
        config,
        path,
        config_path=None,
        path_type=None,
        potential_type=None,
        tag="",
        config_dir="./src/optimizations/configs/",
        expect_config=False):
    
    name = name.lower()
    if name not in optimizer_dict:
        raise ValueError(f"Cannot handle optimizer type {name}, either add it to optimizer_dict or use {list(optimizer_dict.keys())}")

    if config_path is None and (path_type is None and potential_type is None):
        raise ValueError(f"get_optimizer requires either config_path")

    # Import saved optimizer config and combine with input config
    config_path_vars = path_type is not None and potential_type is not None 
    if config_path is None and not config_path_vars and not expect_config:
        print("Skipping optimizer config import")
    elif config_path is not None or config_path_vars:
        config.update(
            import_optimizer_config(
                name,
                path_type,
                potential_type,
                tag,
                config_dir,
                is_expected=expect_config
            )
        )
    else:
        raise ValueError("get_optimizer requres either config_path or both path_type and potential_type to be specified to import the config file.")
    
    optimizer = optimizer_dict[name](**config)
    opt_state = optimizer.init(eqx.filter(path, eqx.is_inexact_array))
    return optimizer, opt_state
