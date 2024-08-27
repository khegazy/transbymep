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

    arg_parser = tools.build_default_arg_parser()
    args = arg_parser.parse_args()
    logger = tools.logging()
    torch.manual_seed(42)
    # wandb.init(project="Geodesic_Sella")

    # Import configuration files
    print(args.name, args.path_tag, args.tag)
    config = tools.import_run_config(
        args.name, path_tag=args.path_tag, tag=args.tag, flags=args
    )

    path_config = tools.import_path_config(
        config, path_tag=args.path_tag
    )

    path_integral = optimize_MEP(args, config, path_config, logger)
