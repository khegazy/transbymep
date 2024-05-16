import os
import sys
import torch
import numpy as np
from typing import NamedTuple
from matplotlib import pyplot as plt
import time as timer

from src import mechanics
from src import tools
from src import paths
from src import optimization
from src import ddp
from src.tools import visualize
from src.potentials import get_potential


if __name__ == "__main__":
    ###############################
    #####  Setup environment  #####
    ###############################
    
    arg_parser = tools.build_default_arg_parser()
    args = arg_parser.parse_args()
    logger = tools.logging()
    process = ddp.DistributedEnvironment(
        device_type=args.device, is_local=args.is_local, is_slurm=args.is_slurm
    )
    print(process)

    # Import configuration files
    config = tools.import_run_config(
        args.name, path_tag=args.path_tag, tag=args.tag, flags=args
    )
    path_config = tools.import_path_config(
        config, path_tag=args.path_tag
    )

    # Create output directories
    output_dir = os.path.join(args.output_dir, config.potential, config.optimizer)
    log_dir = os.path.join(output_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    #####  Get chemical potential  #####
    potential = get_potential(
        config.potential,
        tag=config.potential_tag,
        expect_config=config.potential!="constant",
        add_azimuthal_dof=args.add_azimuthal_dof,
        add_translation_dof=args.add_translation_dof
    )

    # Minimize initial points with the given potential
    if args.minimize_end_points:
        minima_finder = optimization.MinimaUpdate(potential)
        minima = minima_finder.find_minima(
            [config.initial_point, config.final_point]
        )
        print(f"Optimized Initial Point: {minima[0]}")
        print(f"Optimized Final Point: {minima[1]}")
        sys.exit(0)

    #####  Get path prediction method  #####
    path = paths.get_path(
        config.path,
        potential,
        config.initial_point,
        config.final_point,
        process=process,
        #add_azimuthal_dof=args.add_azimuthal_dof,
        #add_translation_dof=args.add_translation_dof,
        **path_config.path_params
    )

    # Randomly initialize the path, otherwise a straight line
    if args.randomly_initialize_path is not None:
        path = optimization.randomly_initialize_path(
            path, args.randomly_initialize_path
        )

    #####  Path optimization tools  #####
    # Path integrating function
    integrator = tools.ODEintegrator(
        potential,
        solver=config.integral_params['solver'],
        rtol=config.integral_params['rtol'],
        atol=config.integral_params['atol'],
        process=process,
        is_multiprocess=config.is_multiprocess,
        is_load_balance=config.is_load_balance
    )
    #print("test integrate", integrator.path_integral(path, 'E_pvre'))

    # Gradient descent path optimizer
    optimizer = optimization.PathOptimizer(
        config.optimizer,
        config.optimizer_params,
        path,
        config.loss_function,
        process,
        path_type=config.path,
        potential_type=config.potential,
        config_tag=config.optimizer_config_tag
    ) 
    
    # Loss
    #print(config.loss_functions)
    #loss_grad_fxn, loss_fxn = optimization.get_loss(config.loss_functions)
    
    ##########################################
    #####  Optimize minimum energy path  ##### 
    ##########################################
    geo_paths = []
    pes_paths = []
    t0 = timer.time()
    prev_t = t0
    for optim_idx in range(args.num_optimizer_iterations):
        path_integral = optimizer.optimization_step(path, integrator)
        if optim_idx%250 == 0 and process.is_master:
            print(f"EVAL TIME: {(timer.time()-prev_t)/60:0.3f} / {(timer.time()-t0)/60:0.3f} min ")
            prev_t = timer.time()
            path_output = logger.optimization_step(
                optim_idx,
                path,
                potential,
                path_integral,
                plot=args.make_opt_plots,
                geo_paths=geo_paths,
                pes_paths=pes_paths,
                add_azimuthal_dof=args.add_azimuthal_dof,
                add_translation_dof=args.add_translation_dof
            )

    print("EVAL TIME", (timer.time()-t0)/60)
    # Plot gif animation of the MEP optimization (only for 2d potentials)
    if args.make_animation:
        geo_paths = potential.point_transform(torch.tensor(geo_paths))
        ani_name = f"{config.potential}_W{path_config.path_params['n_embed']}_D{path_config.path_params['depth']}_LR{config.optimizer_params['lr']}"
        visualize.animate_optimization_2d(
            geo_paths, ani_name, ani_name,
            potential, plot_min_max=(-2, 2, -2, 2),
            levels=np.arange(-100,100,5),
            add_translation_dof=args.add_translation_dof,
            add_azimuthal_dof=args.add_azimuthal_dof
        )
    
    process.end_process()
