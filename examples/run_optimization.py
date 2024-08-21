import os
import sys
import torch
import numpy as np
from typing import NamedTuple
from matplotlib import pyplot as plt
import time as timer

from transbymep import tools
from transbymep import paths
from transbymep import optimization
from transbymep.tools import visualize
from transbymep.potentials import get_potential


def run_opt(
        args: NamedTuple,
        config: NamedTuple,
        path_config: NamedTuple,
        logger: NamedTuple
):
    """
    Run optimization process.

    Args:
        args (NamedTuple): Command line arguments.
        config (NamedTuple): Configuration settings.
        path_config (NamedTuple): Path configuration.
        logger (NamedTuple): Logger settings.
    """
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
        add_translation_dof=args.add_translation_dof,
        **config.potential_params,
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
    print("PATH PARAMS", path_config.path_params)
    path = paths.get_path(
        config.path,
        potential,
        config.initial_point,
        config.final_point,
        device=config.device,
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
    print("int params", config.integral_params)
    integrator = tools.ODEintegrator(**config.integral_params)
    #print("test integrate", integrator.path_integral(path, 'E_pvre'))

    # Gradient descent path optimizer
    optimizer = optimization.PathOptimizer(
        config.optimizer,
        config.optimizer_params,
        path,
        config.loss_function,
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
    loss_curve = []
    for optim_idx in range(args.num_optimizer_iterations):
        path_integral = optimizer.optimization_step(path, integrator)
        #print(f'optim_idx:, {optim_idx}, {path_integral.integral}')
        loss_curve.append(path_integral.integral.item())
        if optim_idx%50 == 0:
            print("EVAL TIME", (timer.time()-t0)/60)
            path_output = logger.optimization_step(
                optim_idx,
                path,
                potential,
                path_integral.integral,
                plot=args.make_opt_plots,
                geo_paths=geo_paths,
                pes_paths=pes_paths,
                add_azimuthal_dof=args.add_azimuthal_dof,
                add_translation_dof=args.add_translation_dof
            )
            fig, ax = plt.subplots()
            ax.plot(loss_curve)
            if not os.path.exists("./plots"):
                os.makedirs("./plots")
            fig.savefig("./plots/loss_curve.png")

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
    return path_integral


if __name__ == "__main__":
    ###############################
    #####  Setup environment  #####
    ###############################

    arg_parser = tools.build_default_arg_parser()
    args = arg_parser.parse_args()
    logger = tools.logging()

    # Import configuration files
    print(args.name, args.path_tag, args.tag)
    config = tools.import_run_config(
        args.name, path_tag=args.path_tag, tag=args.tag, flags=args
    )

    path_config = tools.import_path_config(
        config, path_tag=args.path_tag
    )

    path_integral = run_opt(args, config, path_config, logger)
