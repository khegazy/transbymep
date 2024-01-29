import os
import yaml
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class RunConfig:
    name : str
    initial_point : tuple[float, float]
    final_point : Tuple[float, float]
    potential : str
    potential_params : Dict
    path : str
    path_config : str
    loss_functions : Dict
    optimizer : str
    optimizer_params : Dict
    tag : str = ""
    potential_tag : str = ""

@dataclass
class PathConfig:
    name : str
    path_params : Dict
    tag : str = ""


def import_yaml(address, is_expected=False):
    if os.path.exists(address):
        with open(address, 'r') as file:
            loaded_yaml = yaml.safe_load(file)
        return loaded_yaml
    elif is_expected:
        raise ImportError(f"Cannot find required file {address}")
    else:
        ImportWarning(f"Cannot find file {address}, still running")
        return {}

def import_run_config(name, path_tag="", tag="", potential_tag="", dir="./configs/", is_expected=True):
    #filename = f"{name}_{path_type}"
    filename = name
    filename += f"_{tag}" if tag != "" else ""
    filename += f"_{path_tag}" if path_tag != "" else ""
    filename += f"_{potential_tag}" if tag != "" else ""
    filename += ".yaml"
    yaml_config = import_yaml(os.path.join(dir, filename), is_expected)
    
    print("run yaml inp", yaml_config)
    if "potential_params" not in yaml_config:
        yaml_config["potential_params"] = {}

    config = RunConfig(name=name, **yaml_config, tag=tag)
    print("config", config.loss_functions)
    for fxn in config.loss_functions.keys():
        config.loss_functions[fxn] = tuple(config.loss_functions[fxn])

    return config

def import_path_config(run_config, path_tag="", dir="./src/paths/configs/", is_expected=True):
    filename = f"{run_config.path}_{run_config.path_config}"
    filename += f"_{path_tag}" if path_tag != "" else ""
    filename += ".yaml"
    yaml_config = import_yaml(os.path.join(dir, filename), is_expected)
    
    print("path yaml inp", yaml_config)
    config = PathConfig(name=run_config.path, **yaml_config, tag=path_tag)

    return config


    

