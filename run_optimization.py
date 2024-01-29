import os
import jax.numpy as jnp
import numpy as np
from jax import random
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
from src.optimization import get_optimizer, losses
from src.paths import initialize as init

from src.potentials.wolfe_schlegel import ws 



if __name__ == "__main__":
    # Setup environment
    arg_parser = tools.build_default_arg_parser()
    args = arg_parser.parse_args()

    config = tools.import_run_config(args.name, args.path_tag, args.tag)
    path_config = tools.import_path_config(config, args.path_tag)
    print("fin config", config)

    # Create output directories
    output_dir = os.path.join(args.output_dir, config.potential, config.optimizer)
    log_dir = os.path.join(output_dir, "logs")
    if not os.path.exists(output_dir):
        os.makedirs(log_dir)
    
    # Get chemical potential
    potential = get_potential(
        config.potential,
        tag=config.potential_tag,
        expect_config=config.potential!="constant"
    )

    # Get path calculation method
    path, filter_spec = paths.get_path(
        config.path,
        potential,
        config.initial_point,
        config.final_point,
        **path_config.path_params
    )
    """
    fig, ax = plt.subplots()
    for t in np.arange(0,100):
        point = path.eval(t/99)
        print(point)
        ax.scatter(t/99., point[0])
    fig.savefig("testpath.png")
    """

    # Path integrator
    integrator = optimization.ODEintegrator(potential)
    print("test integrate", integrator.path_integral(path.vre_residual))

    # Optimizer
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
    # Optimize path
    """
    #optim = optax.adabelief(0.001)
    optim = optax.sgd(0.05)
    #optim = optax.adam(0.005)
    opt_state = optim.init(eqx.filter(path, eqx.is_inexact_array))
    """
    #print("opt_state", opt_state)
    geo_paths = []
    pes_paths = []
    for i in range(20000):
        diff_path, static_path = eqx.partition(path, filter_spec)
        #print("diff_path", diff_path)
        #print("static_path", static_path)
        loss, grads = loss_grad_fxn(diff_path, static_path, integrator)
        if i%500 == 0:
            print("Step", i)
            print("loss", loss)
            print(path.total_grad_path(0.55, 0.))
            for ii in range(len(path.mlp.layers)):
                print(f"W{ii} sum: {jnp.sum(path.mlp.layers[ii].weight)}")
                print(f"Wg{ii} sum: {jnp.sum(grads.mlp.layers[ii].weight)}")
            #print(path.mlp.layers[0].weight)
            #print("test grad", grads.mlp.layers[0].weight)
            geo_path, pes_path = path.get_path()
            geo_paths.append(geo_path)
            pes_paths.append(pes_path)
            visualize.plot_path(
                geo_path, f"test_plot_{i:03d}", pes_fxn=potential,
                x_min=-2, x_max=2, y_min=-2, y_max=2,
                levels=np.arange(-100,100,5))

        updates, opt_state = optim.update(grads, opt_state)
        #print("grads", grads.initial_point)
        #print("Updtads", updates.initial_point)
        path = eqx.apply_updates(path, updates)
        #plot_times = jnp.expand_dims(jnp.arange(100, dtype=float), 1)/99

    print("PATH LISTS", len(geo_paths), geo_paths[0].shape, geo_paths[0])
    ani_name = f"{config.potential}_W{path_config.path_params['n_embed']}_D{path_config.path_params['depth']}_LR{config.optimizer_params['learning_rate']}"
    visualize.animate_optimization_2d(
        geo_paths, ani_name, ani_name,
        potential, x_min=-2, x_max=2, y_min=-2, y_max=2,
        levels=np.arange(-100,100,5)
    )
    """
    learning_rate = 0.1
    def update(params):

        #integrate(params)
        grads = jax.grad(loss_fxn)(params)#grad_fxn(params)
        print("grads", grads)
        return jax.tree_map(
            lambda param, g: param - g*learning_rate, params, grads
    )

    for i in range(1000):
        print("step", i)
        path_points = predict(path_params, points, plot_times)
        print("path points", plot_times.shape, path_points.shape)
        visualize.plot_path(path_points, f"test_path_{i}")
        path_params = update(path_params)

    """




    """
    loss_fxn = None
    # Get optimizer
    optimizer = get_optimizer(
        config.optimizer,
        config.path,
        config.potential,
        potential,
        loss_fxn,
        #action=mechanics.action,
    )
    """

    #minima = optimizer.find_minima()

    """
    paths = optimizer.find_critical_path(path)

    # Plot results
    visualize.contour_2d(
        function=potential,
        x_min=-2.0,
        x_max=2.0,
        y_min=-2.0,
        y_max=2.0,
        levels=np.arange(-100,100,5),
        paths=paths,
        title="wolfe schlegel",
        contour_file = f'./plots/{args.potential}/contour_plot'
    )

    """


