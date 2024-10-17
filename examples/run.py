import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import NamedTuple
from matplotlib import pyplot as plt
import time as timer
from tqdm import tqdm
import wandb
import ase, ase.io

from transbymep import tools, optimize_MEP
from transbymep.tools import visualize


if __name__ == "__main__":
    ###############################
    #####  Setup environment  #####
    ###############################

    torch.manual_seed(42)
    # wandb.init(project="Geodesic_Sella")

    # Import configuration files
    config = tools.import_run_config('configs/wolfe.yaml')

    paths_time, paths_geometry, paths_energy, paths_velocity, paths_force, paths_loss, paths_integral, paths_neval = optimize_MEP(**config)
