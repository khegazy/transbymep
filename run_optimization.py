import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import NamedTuple
from matplotlib import pyplot as plt
from time import time

import equinox as eqx
import optax  # https://github.com/deepmind/optax

from src import mechanics
from src import tools
from src import paths
from src import optimization
from src.tools import visualize
from src.potentials import get_potential


if __name__ == "__main__":
    ###############################
    #####  Setup environment  #####
    ###############################
    
    arg_parser = tools.build_default_arg_parser()
    args = arg_parser.parse_args()
    logger = tools.logging()

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
    if not os.path.exists(output_dir):
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
    path, filter_spec = paths.get_path(
        config.path,
        potential,
        config.initial_point,
        config.final_point,
        #add_azimuthal_dof=args.add_azimuthal_dof,
        #add_translation_dof=args.add_translation_dof,
        **path_config.path_params
    )

    # Randomly initialize the path, otherwise a straight line
    if args.randomly_initialize_path:
        path = optimization.randomly_initialize_path(path, filter_spec, 1)

    #####  Path optimization tools  #####
    # Path integrating function
    integrator = optimization.ODEintegrator(potential)
    print("test integrate", integrator.path_integral(path.vre_residual))

    # Gradient descent path optimizer
    optim, opt_state = optimization.get_optimizer(
        config.optimizer,
        config.optimizer_params,
        path,
        path_type=config.path,
        potential_type=config.potential
    ) 
    
    # Loss
    print(config.loss_functions)
    loss_grad_fxn, loss_fxn = optimization.get_loss(config.loss_functions)
    
    ##########################################
    #####  Optimize minimum energy path  ##### 
    ##########################################
    geo_paths = []
    pes_paths = []
    for optim_idx in range(args.num_optimizer_iterations):
        diff_path, static_path = eqx.partition(path, filter_spec)
        #print("diff_path", diff_path)
        #print("static_path", static_path)
        loss, grads = loss_grad_fxn(diff_path, static_path, integrator)
        if optim_idx%250 == 0:
            logger.optimization_step(
                optim_idx,
                path,
                potential,
                loss,
                grads,
                plot=True,
                geo_paths=geo_paths,
                pes_paths=pes_paths,
                add_azimuthal_dof=args.add_azimuthal_dof,
                add_translation_dof=args.add_translation_dof
            )
        updates, opt_state = optim.update(grads, opt_state)
        #print("grads", grads.initial_point)
        #print("Updtads", updates.initial_point)
        path = eqx.apply_updates(path, updates)
        #plot_times = jnp.expand_dims(jnp.arange(100, dtype=float), 1)/99

    # Plot gif animation of the MEP optimization (only for 2d potentials)
    if args.make_animation:
        geo_paths = jax.vmap(potential.point_transform, 0)(jnp.array(geo_paths))
        ani_name = f"{config.potential}_W{path_config.path_params['n_embed']}_D{path_config.path_params['depth']}_LR{config.optimizer_params['learning_rate']}"
        visualize.animate_optimization_2d(
            geo_paths, ani_name, ani_name,
            potential, plot_min_max=(-2, 2, -2, 2),
            levels=np.arange(-100,100,5),
            add_translation_dof=args.add_translation_dof,
            add_azimuthal_dof=args.add_azimuthal_dof
        )
