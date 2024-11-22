import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import NamedTuple, Any
from matplotlib import pyplot as plt
import time as time
from tqdm import tqdm
import wandb
from ase import Atoms
from ase.io import read, write
from typing import NamedTuple
from dataclasses import dataclass

from transbymep import tools
from transbymep import paths
from transbymep import optimization
from transbymep.optimization import initialize_path
from transbymep.tools import visualize
from transbymep.potentials import get_potential


@dataclass
class OptimizationOutput():
    paths_time: list[np.ndarray]
    paths_geometry: list[np.ndarray]
    paths_energy: list[np.ndarray]
    paths_velocity: list[np.ndarray]
    paths_force: list[np.ndarray]
    paths_loss: list[np.ndarray]
    paths_integral: list[float]
    paths_neval: list[int]


def optimize_MEP(
        images: list[Atoms],
        output_dir: str | None = None,
        potential_params: dict[str, Any] = {},
        path_params: dict[str, Any] = {},
        integrator_params: dict[str, Any] = {},
        optimizer_params: dict[str, Any] = {},
        scheduler_params: dict[str, Any] = {},
        loss_scheduler_params: dict[str, Any] = {},
        num_optimizer_iterations: int = 1000,
        device: str = 'cuda',
):
    """
    Run optimization process.

    Args:
        args (NamedTuple): Command line arguments.
        config (NamedTuple): Configuration settings.
        path_config (NamedTuple): Path configuration.
        logger (NamedTuple): Logger settings.
    """
    print("Images", images)
    print("Potential Params", potential_params)
    print("Path Params", path_params)
    print("Integrator Params", integrator_params)
    print("Optimizer Params", optimizer_params)

    torch.manual_seed(42)

    # Create output directories
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_dir = os.path.join(output_dir, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        plot_dir = os.path.join(output_dir, "plots")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
    
    #####  Get chemical potential  #####
    potential = get_potential(**potential_params, device=device)

    #####  Get path prediction method  #####
    path = paths.get_path(potential=potential, initial_point=images[0], final_point=images[-1], **path_params, device=device)

    # Randomly initialize the path, otherwise a straight line
    if len(images) > 2:
        path = initialize_path(
            path=path, 
            times=torch.linspace(0, 1, len(images), device=device), 
            init_points=torch.tensor([image.positions.flatten() for image in images], device=device),
            )

    #####  Path optimization tools  #####
    integrator = tools.ODEintegrator(**integrator_params, device=device)

    # potential.trainer.model.molecular_graph_cfg.max_num_nodes_per_batch = path.n_atoms
    # potential.trainer.model.global_cfg.batch_size = integrator._integrator.max_batch
    # potential.trainer.model.global_cfg.use_export = False
    # potential.trainer.model.global_cfg.use_compile = False

    # Gradient descent path optimizer
    optimizer = optimization.PathOptimizer(path=path, **optimizer_params, device=device)
    # if scheduler_params:
    #     optimizer.set_scheduler(**scheduler_params)
    if loss_scheduler_params:
        optimizer.set_loss_scheduler(optimizer=optimizer, integrator=integrator, **loss_scheduler_params)
        # metric_parameters = {key: loss_scheduler.get_value() for key, loss_scheduler in optimizer.loss_scheduler.items()}
        # integrator.update_metric_parameters(metric_parameters)

    ##########################################
    #####  Optimize minimum energy path  ##### 
    ##########################################
    paths_time = []
    paths_geometry = []
    paths_energy = []
    paths_velocity = []
    paths_force = []
    paths_loss = []
    paths_integral = []
    paths_neval = []
    for optim_idx in tqdm(range(num_optimizer_iterations)):
        path.neval = 0
        try:
            path_integral = optimizer.optimization_step(path, integrator)
            neval = path.neval
            print(f"Step {optim_idx}, Integral: {path_integral.integral.item()}, Neval: {neval}")
        except ValueError as e:
            print("ValueError", e)
            raise e

        paths_integral.append(path_integral.integral.item())
        paths_neval.append(neval)
        time = path_integral.t.detach()
        # time = torch.linspace(0, 1, 101, device=device).reshape(1, -1)
        paths_time.append(time.flatten().to('cpu').numpy())
        loss = path_integral.y.detach()
        paths_loss.append(loss.flatten().to('cpu').numpy())

        del path_integral
        
        # path_time = []
        # path_geometry, path_energy, path_velocity, path_force = [], [], [], []
        # for t in time:
        #     t = t.flatten()
        #     t = torch.linspace(t[0], t[-1], len(t), device=device).reshape(-1, 1)
        #     path_time.append(t.detach().to('cpu').numpy())
        #     path_output = path(t, return_velocity=True, return_energy=True, return_force=True)
        #     path_geometry.append(path_output.path_geometry.detach().to('cpu').numpy())
        #     path_energy.append(path_output.path_energy.detach().to('cpu').numpy())
        #     path_velocity.append(path_output.path_velocity.detach().to('cpu').numpy())
        #     path_force.append(path_output.path_force.detach().to('cpu').numpy())
        #     del path_output
        # paths_time.append(np.concatenate(path_time))
        # paths_geometry.append(np.concatenate(path_geometry))
        # paths_energy.append(np.concatenate(path_energy))
        # paths_velocity.append(np.concatenate(path_velocity))
        # paths_force.append(np.concatenate(path_force))
        # for ind_t, t in enumerate(time):
        #     t = t.flatten()
        #     t = torch.linspace(t[0], t[-1], len(t), device=device).reshape(-1, 1)
        #     time[ind_t] = t
        time = time.reshape(-1, 1)
        path_output = path(time, return_velocity=True, return_energy=True, return_force=True)
        # paths_time.append(time.detach().to('cpu').numpy())
        paths_geometry.append(path_output.path_geometry.detach().to('cpu').numpy())
        paths_energy.append(path_output.path_energy.detach().to('cpu').numpy())
        paths_velocity.append(path_output.path_velocity.detach().to('cpu').numpy())
        paths_force.append(path_output.path_force.detach().to('cpu').numpy())
        del path_output

        if optim_idx % 50 == 0:
            if output_dir is not None:
                log_filename = os.path.join(log_dir, f"output_{optim_idx}.npz")
                np.savez(
                    log_filename, 
                    path_times=paths_time[-1],
                    path_geometry=paths_geometry[-1],
                    path_energy=paths_energy[-1],
                    path_velocity=paths_velocity[-1],
                    path_force=paths_force[-1],
                    path_loss=paths_loss[-1],
                    path_integral=paths_integral[-1],
                    path_neval=paths_neval[-1],
                )
            if output_dir is not None:
                plot_filename = os.path.join(plot_dir, f"output_{optim_idx}.png")
                visualize.plot_path(
                    plot_filename,
                    time=paths_time[-1],
                    geometry=paths_geometry[-1],
                    energy=paths_energy[-1],
                    velocity=paths_velocity[-1],
                    force=paths_force[-1],
                    loss=paths_loss[-1],
                )

        if optimizer.converged:
            print(f"Converged at step {optim_idx}")
            break

    # path_geometry, path_energy, path_velocity, path_force = [], [], [], []
    # for t in time:
    #     t = t.flatten()
    #     t = torch.linspace(t[0], t[-1], 97, device=device).reshape(-1, 1)
    #     path_output = path(t, return_velocity=True, return_energy=True, return_force=True)
    #     path_geometry.append(path_output.path_geometry.detach().to('cpu').numpy())
    #     path_energy.append(path_output.path_energy.detach().to('cpu').numpy())
    #     path_velocity.append(path_output.path_velocity.detach().to('cpu').numpy())
    #     path_force.append(path_output.path_force.detach().to('cpu').numpy())
    #     del path_output
    # paths_geometry.append(np.concatenate(path_geometry))
    # paths_energy.append(np.concatenate(path_energy))
    # paths_velocity.append(np.concatenate(path_velocity))
    # paths_force.append(np.concatenate(path_force))

    output = OptimizationOutput(
        paths_time=paths_time,
        paths_geometry=paths_geometry,
        paths_energy=paths_energy,
        paths_velocity=paths_velocity,
        paths_force=paths_force,
        paths_loss=paths_loss,
        paths_integral=paths_integral,
        paths_neval=paths_neval,
    )
    return output
