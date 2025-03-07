import os
import torch
import numpy as np
from typing import Any
import time as time
from tqdm import tqdm
from ase import Atoms
from dataclasses import dataclass
import json

from popcornn.paths import get_path
from popcornn.optimization import initialize_path
from popcornn.optimization import PathOptimizer
from popcornn.tools import process_images, output_to_atoms
from popcornn.tools import ODEintegrator
from popcornn.potentials import get_potential


@dataclass
class OptimizationOutput():
    path_time: list
    path_geometry: list
    path_energy: list
    path_velocity: list
    path_force: list
    path_loss: list
    path_integral: float
    # path_neval: int
    path_ts_time: list
    path_ts_geometry: list
    path_ts_energy: list
    path_ts_velocity: list
    path_ts_force: list

    def save(self, file):
        with open(file, 'w') as f:
            json.dump(self.__dict__, f)


def optimize_MEP(
        images: list[Atoms],
        output_dir: str | None = None,
        potential_params: dict[str, Any] = {},
        path_params: dict[str, Any] = {},
        integrator_params: dict[str, Any] = {},
        optimizer_params: dict[str, Any] = {},
        scheduler_params: dict[str, Any] = {},
        loss_scheduler_params: dict[str, Any] = {},
        num_optimizer_iterations: int = 1001,
        num_record_points: int = 101,
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

    #####  Process images  #####
    images = process_images(images)
    
    #####  Get chemical potential  #####
    #     add_azimuthal_dof=args.add_azimuthal_dof,
    #     add_translation_dof=args.add_translation_dof,
    potential = get_potential(**potential_params, images=images, device=device)

    #####  Get path prediction method  #####
    # Minimize initial points with the given potential
    # if args.minimize_end_points:
    # if minimize_end_points:
    #     minima_finder = optimization.MinimaUpdate(potential)
    #     minima = minima_finder.find_minima(
    #         [config.initial_point, config.final_point]
    #     )
    #     print(f"Optimized Initial Point: {minima[0]}")
    #     print(f"Optimized Final Point: {minima[1]}")
    #     sys.exit(0)
    path = get_path(potential=potential, images=images, **path_params, device=device)

    # Randomly initialize the path, otherwise a straight line
    # if args.randomly_initialize_path is not None:
    #     path = optimization.randomly_initialize_path(
    #         path, args.randomly_initialize_path
    #     )
    if len(images) > 2:
        path = initialize_path(
            path=path, 
            times=torch.linspace(0, 1, len(images), device=device), 
            init_points=images.points.to(device),
        )

    #####  Path optimization tools  #####
    integrator = ODEintegrator(**integrator_params, device=device)

    # potential.trainer.model.molecular_graph_cfg.max_num_nodes_per_batch = path.n_atoms
    # potential.trainer.model.global_cfg.batch_size = integrator._integrator.max_batch
    # potential.trainer.model.global_cfg.use_export = False
    # potential.trainer.model.global_cfg.use_compile = False

    # Gradient descent path optimizer
    optimizer = PathOptimizer(path=path, **optimizer_params, device=device)
    """
    if scheduler_params:
        optimizer.set_scheduler(**scheduler_params)
    if loss_scheduler_params:
        optimizer.set_loss_scheduler(**loss_scheduler_params)
        metric_parameters = {key: loss_scheduler.get_value() for key, loss_scheduler in optimizer.loss_scheduler.items()}
        integrator.update_metric_parameters(metric_parameters)
    """

    ##########################################
    #####  Optimize minimum energy path  ##### 
    ##########################################
    for optim_idx in tqdm(range(num_optimizer_iterations)):
        path.neval = 0
        try:
            path_integral = optimizer.optimization_step(path, integrator)
            neval = path.neval
        except ValueError as e:
            print("ValueError", e)
            raise e

        if output_dir is not None:
            time = path_integral.t.flatten()
            ts_time = path.TS_time
            path_output = path(time, return_velocity=True, return_energy=True, return_force=True)
            ts_output = path(ts_time, return_velocity=True, return_energy=True, return_force=True)

            output = OptimizationOutput(
                path_time=time.tolist(),
                path_geometry=path_output.path_geometry.tolist(),
                path_energy=path_output.path_energy.tolist(),
                path_velocity=path_output.path_velocity.tolist(),
                path_force=path_output.path_force.tolist(),
                path_loss=path_integral.y.tolist(),
                path_integral=path_integral.integral.item(),
                # path_neval=neval,
                path_ts_time=ts_time.tolist(),
                path_ts_geometry=ts_output.path_geometry.tolist(),
                path_ts_energy=ts_output.path_energy.tolist(),
                path_ts_velocity=ts_output.path_velocity.tolist(),
                path_ts_force=ts_output.path_force.tolist(),
            )
            output.save(os.path.join(log_dir, f"output_{optim_idx}.json"))            

            # if optim_idx % 1 == 0:
            #     #     path_output = logger.optimization_step(	
            #     #         optim_idx,	
            #     #         path,	
            #     #         potential,	
            #     #         path_integral.integral,	
            #     #         plot=args.make_opt_plots,	
            #     #         plot_dir=plot_dir,	
            #     #         geo_paths=geo_paths,	
            #     #         pes_paths=pes_paths,	
            #     #         add_azimuthal_dof=args.add_azimuthal_dof,	
            #     #         add_translation_dof=args.add_translation_dof	
            #     #     )
            #     log_filename = os.path.join(log_dir, f"output_{optim_idx}.npz")
            #     np.savez(
            #         log_filename, 
            #         path_times=paths_time[-1],
            #         path_geometry=paths_geometry[-1],
            #         path_energy=paths_energy[-1],
            #         path_velocity=paths_velocity[-1],
            #         path_force=paths_force[-1],
            #         path_loss=paths_loss[-1],
            #         path_integral=paths_integral[-1],
            #         path_neval=paths_neval[-1],
            #         ts_times=paths_ts_time[-1],
            #         ts_geometry=paths_ts_geometry[-1],
            #         ts_energy=paths_ts_energy[-1],
            #         ts_velocity=paths_ts_velocity[-1],
            #         ts_force=paths_ts_force[-1],
            #     )
            #     # plot_filename = os.path.join(plot_dir, f"output_{optim_idx}.png")
            #     # visualize.plot_path(
            #     #     plot_filename,
            #     #     time=paths_time[-1],
            #     #     geometry=paths_geometry[-1],
            #     #     energy=paths_energy[-1],
            #     #     velocity=paths_velocity[-1],
            #     #     force=paths_force[-1],
            #     #     loss=paths_loss[-1],
            #     #     ts_times=paths_ts_time[-1],
            #     #     ts_geometry=paths_ts_geometry[-1],
            #     #     ts_energy=paths_ts_energy[-1],
            #     #     ts_velocity=paths_ts_velocity[-1],
            #     #     ts_force=paths_ts_force[-1],
            #     # )

        if optimizer.converged:
            print(f"Converged at step {optim_idx}")
            break

    # del optimizer
    # del integrator
    # del potential
    torch.cuda.empty_cache()

    #####  Save optimization output  #####
    time = torch.linspace(path.t_init.item(), path.t_final.item(), num_record_points)
    ts_time = path.TS_time
    path_output = path(time, return_velocity=True, return_energy=True, return_force=True)
    ts_output = path(ts_time, return_velocity=True, return_energy=True, return_force=True)
    if images.dtype == Atoms:
        images, ts_images = output_to_atoms(path_output, images), output_to_atoms(ts_output, images)
        return images, ts_images[0]
    else:
        return OptimizationOutput(
            path_time=time.tolist(),
            path_geometry=path_output.path_geometry.tolist(),
            path_energy=path_output.path_energy.tolist(),
            path_velocity=path_output.path_velocity.tolist(),
            path_force=path_output.path_force.tolist(),
            path_loss=path_integral.y.tolist(),
            path_integral=path_integral.integral.item(),
            # path_neval=neval,
            path_ts_time=ts_time.tolist(),
            path_ts_geometry=ts_output.path_geometry.tolist(),
            path_ts_energy=ts_output.path_energy.tolist(),
            path_ts_velocity=ts_output.path_velocity.tolist(),
            path_ts_force=ts_output.path_force.tolist(),
        )

