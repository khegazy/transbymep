import os
import yaml
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
from ase.io import read, write


def import_yaml(address):
    with open(address, 'r') as file:
        loaded_yaml = yaml.safe_load(file)
    return loaded_yaml


def import_run_config(name):
    yaml_config = import_yaml(name)
    print("run yaml inp", yaml_config)
    if 'images' in yaml_config and isinstance(yaml_config['images'], str) and yaml_config['images'].endswith('.xyz'):
        yaml_config['images'] = read(yaml_config['images'], index=':')
    return yaml_config
