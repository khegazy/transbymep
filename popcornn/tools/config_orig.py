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
    path_config_tag : str
    integral_params : Dict
    loss_function : str
    optimizer : str
    optimizer_config_tag : str
    optimizer_params : Dict
    tag : str = ""
    potential_tag : str = ""
    device : str = 'cuda'

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


def import_run_config(
        name,
        path_tag="",
        tag="",
        potential_tag="",
        dir="./configs/runs",
        is_expected=True,
        flags=None,
        device='cuda'
    ):
    #filename = f"{name}_{path_type}"
    filename = name
    filename += f"_{tag}" if tag != "" else ""
    filename += f"_{path_tag}" if path_tag != "" else ""
    filename += f"_{potential_tag}" if tag != "" else ""
    filename += ".yaml"
    yaml_config = import_yaml(os.path.join(dir, filename), is_expected)
    
    print("run yaml inp", yaml_config)
    if 'potential_params' not in yaml_config:
        yaml_config['potential_params'] = {}

    if 'integral_params' not in yaml_config:
        yaml_config['integral_params'] = {
            'method' : 'dopri5',
            'rtol' : 1e-7,
            'atol' : 1e-9,
            'computation' : 'parallel'
        }
    else:
        if 'rtol' in yaml_config['integral_params']:
            yaml_config['integral_params']['rtol'] =\
                float(yaml_config['integral_params']['rtol'])
        else:
            yaml_config['integral_params']['rtol'] = 1e-7
        if 'atol' in yaml_config['integral_params']:
            yaml_config['integral_params']['atol'] =\
                float(yaml_config['integral_params']['atol'])
        else:
            yaml_config['integral_params']['atol'] = 1e-9
        if 'computation' not in yaml_config['integral_params']:
            yaml_config['integral_params']['computation'] = 'parallel'
        


    config = RunConfig(name=name, **yaml_config, tag=tag)
    print("config", config.loss_function)
    """
    for fxn in config.loss_functions.keys():
        config.loss_functions[fxn] = tuple(config.loss_functions[fxn])
    """

    if flags is not None:
        if flags.max_batch is not None or 'max_batch' not in config.integral_params:
            config.integral_params['max_batch'] = flags.max_batch
        if flags.add_azimuthal_dof is not None:
            config.initial_point[0] += flags.add_azimuthal_dof 
            # Rotate by pi/2
            #config.final_point[-1] = config.final_point[0] + flags.add_azimuthal_dof
            #config.final_point[0] = 0
            # Rotate by pi
            config.final_point[0] = -1*(config.final_point[0] + flags.add_azimuthal_dof)
        print(config.device)
        print(flags.device)
        config.device = flags.device

    return config


def import_path_config(
        run_config,
        path_tag="",
        dir="./configs/paths",
        is_expected=True,
    ):
    filename = f"{run_config.path}_{run_config.path_config_tag}"
    filename += f"_{path_tag}" if path_tag != "" else ""
    filename += ".yaml"
    yaml_config = import_yaml(os.path.join(dir, filename), is_expected)

    print(os.path.join(dir, filename)) 
    print("path yaml inp", yaml_config)
    config = PathConfig(
        name=run_config.path, 
        path_params=yaml_config,
        tag=path_tag,
    )
    
    return config
