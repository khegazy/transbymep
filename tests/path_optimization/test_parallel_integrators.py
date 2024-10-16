import os
import sys
import torch
import pytest
import argparse
import numpy as np
from typing import NamedTuple
from matplotlib import pyplot as plt
import time as timer

from torchpathdiffeq import UNIFORM_METHODS, VARIABLE_METHODS
from transbymep import tools, optimize_MEP
from transbymep.potentials import get_potential


@pytest.mark.parametrize(
    # "name, run_config_dir, path_config_dir, path_tag, tag, seed, output_dir, path, optimizer, num_optimizer_iterations, expected_path_integral",
    "config, expected_path_integral",
    [
        # ('Epvre_test', '../configs/runs', '../configs/paths', "", "", 123, "./output", "mlp", "gradient_descent", 5, 131.2415),
        ('../configs/Epvre_test.yaml', 131.2415),
        # Add more test cases with different input parameters as needed
    ]
)
# def test_parallel_integrators(tmp_path, monkeypatch, name, run_config_dir, path_config_dir, path_tag, tag, seed, output_dir, path, optimizer, num_optimizer_iterations, expected_path_integral):
def test_parallel_integrators(tmp_path, monkeypatch, config, expected_path_integral):
    input_dict = {
        # "name": name,
        # "run_config_dir": run_config_dir,
        # "potential_tag": None,
        # "path_tag": path_tag,
        # "tag": tag,
        # "seed": seed,
        # "output_dir": output_dir,
        # "path": path,
        # "path_config_dir": path_config_dir,
        # 'randomly_initialize_path': None,
        # "potential": None,
        # "optimizer": optimizer,
        # 'minimize_end_points': False,
        # 'num_optimizer_iterations': num_optimizer_iterations,
        # 'make_animation': False,
        # 'make_opt_plots': 0,
        # 'debug': False,
        # 'add_azimuthal_dof': None,
        # 'add_translation_dof': None,
        # 'device': 'cpu'
        'config': config,
    }

    # Create arguments and setup environment
    args = argparse.Namespace(**input_dict)
    file_dir = os.path.dirname(os.path.abspath(__file__))
    monkeypatch.chdir(tmp_path)
    # logger = tools.logging()

    # Get run config
    config = tools.import_run_config(
        # args.name,
        # path_tag=args.path_tag,
        # tag=args.tag,
        # dir=os.path.join(file_dir, args.run_config_dir),
        # flags=args
        os.path.join(file_dir, args.config)
    )
    # config.integral_params['computation'] = 'parallel'
    config['integrator_params']['computation'] = 'parallel'

    # Get path config
    # path_config = tools.import_path_config(
    #     config,
    #     path_tag=args.path_tag,
    #     dir=os.path.join(file_dir, args.path_config_dir)
    # )

    # Loop over uniform and variable parallel integrators, loop over all
    # methods within each sampling type 
    loop = zip(['uniform', 'variable'], [UNIFORM_METHODS, VARIABLE_METHODS])
    for sample_type, sampling_methods in loop:
        # config.integral_params['sample_type'] = sample_type
        config['integrator_params']['sample_type'] = sample_type
        for method in sampling_methods.keys():
            # config.integral_params['method'] = method
            config['integrator_params']['method'] = method
            # path_integral = optimize_MEP(args, config, path_config, logger)
            _, _, _, _, _, _, paths_integral, _ = optimize_MEP(**config)
            path_integral = paths_integral[-1]
            # error_message = f"Failed {sample_type} parallel integrator test for method {method}, expected {expected_path_integral} but got {path_integral.integral.item()}"
            error_message = f"Failed {sample_type} parallel integrator test for method {method}, expected {expected_path_integral} but got {path_integral}"
            # assert path_integral.integral.item() == pytest.approx(expected_path_integral, abs=1, rel=1), error_message
            assert path_integral == pytest.approx(expected_path_integral, abs=1, rel=1), error_message
