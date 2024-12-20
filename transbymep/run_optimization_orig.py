import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import NamedTuple
from matplotlib import pyplot as plt
import time as time
from tqdm import tqdm
import wandb
import ase, ase.io
from typing import NamedTuple

from transbymep import tools
from transbymep import paths
from transbymep import optimization
from transbymep.tools import visualize
from transbymep.potentials import get_potential


def optimize_MEP(
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
    plot_dir = os.path.join(output_dir, "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    #####  Get chemical potential  #####
    potential = get_potential(
        config.potential,
        tag=config.potential_tag,
        expect_config=config.potential!="constant",
        add_azimuthal_dof=args.add_azimuthal_dof,
        add_translation_dof=args.add_translation_dof,
        device=config.device,
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
    integrator = tools.ODEintegrator(**config.integral_params, device=config.device)
    #print("test integrate", integrator.path_integral(path, 'E_pvre'))

    # Gradient descent path optimizer
    optimizer = optimization.PathOptimizer(
        config.optimizer,
        config.optimizer_params,
        path,
        config.loss_function,
        path_type=config.path,
        potential_type=config.potential,
        config_tag=config.optimizer_config_tag,
        device=config.device
    )

    # Loss
    #print(config.loss_functions)
    #loss_grad_fxn, loss_fxn = optimization.get_loss(config.loss_functions)

    ##########################################
    #####  Optimize minimum energy path  ##### 
    ##########################################
    geo_paths = []
    pes_paths = []
    t0 = time.time()
    df = pd.DataFrame(columns=["optim_idx", "neval", "loss"])
    for optim_idx in tqdm(range(args.num_optimizer_iterations)):
        #print(f"Optimization step {optim_idx}")
        path.neval = 0
        try:
            path_integral = optimizer.optimization_step(path, integrator)
            neval = path.neval
            #print(f'n_eval: {neval}, loss: {path_integral.integral.item()}')
            df.loc[optim_idx] = [optim_idx, neval, path_integral.integral.item()]
            # wandb.log({"optim_idx": optim_idx, "neval": neval, "loss": path_integral.integral.item()})
        except ValueError as e:
            print("ValueError", e)
            neval = path.neval
            # wandb.log({"optim_idx": optim_idx, "neval": neval, "loss": np.nan})
            raise e
        
        if optim_idx%250 == 0:
            print("EVAL TIME", (time.time()-t0)/60)
            path_output = logger.optimization_step(
                optim_idx,
                path,
                potential,
                path_integral.integral,
                plot=args.make_opt_plots,
                plot_dir=plot_dir,
                geo_paths=geo_paths,
                pes_paths=pes_paths,
                add_azimuthal_dof=args.add_azimuthal_dof,
                add_translation_dof=args.add_translation_dof
            )
            print("finished logging")
            fig, ax = plt.subplots()
            ax.plot(df["optim_idx"], df["loss"])
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_yscale("log")
            fig.savefig(os.path.join(plot_dir, "loss_curve.png"))
            plt.close()
            df.to_csv(os.path.join(plot_dir, "loss_curve.csv"), header=False)
            # visualize.plot_path(
            #     path.get_path(torch.linspace(0, 1, 32, device='cuda')).geometric_path.detach().to('cpu').numpy(),
            #     f"test_plot_{optim_idx:03d}",
            #     pes_fxn=potential,
            #     plot_min_max=(-2, 2, -2, 2),
            #     levels=16,
            #     plot_dir=plot_dir,
            # )
            # traj = [ase.Atoms(numbers=config.potential_params['numbers'], positions=pos.reshape(-1, 3)) for pos in path.get_path(torch.linspace(0, 1, 101, device='cuda')).geometric_path.detach().to('cpu').numpy()]
            # ase.io.write(os.path.join(plot_dir, f"traj_{optim_idx:03d}.xyz"), traj)

    print("EVAL TIME", (time.time()-t0)/60)
    # Plot gif animation of the MEP optimization (only for 2d potentials)
    # if args.make_animation:
    #     geo_paths = potential.point_transform(torch.tensor(geo_paths))
    #     ani_name = f"{config.potential}_W{path_config.path_params['n_embed']}_D{path_config.path_params['depth']}_LR{config.optimizer_params['lr']}"
    #     visualize.animate_optimization_2d(
    #         geo_paths, ani_name, ani_name,
    #         potential, plot_min_max=(-2, 2, -2, 2),
    #         levels=np.arange(-100,100,5),
    #         add_translation_dof=args.add_translation_dof,
    #         add_azimuthal_dof=args.add_azimuthal_dof
    #     )
    return path_integral


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
