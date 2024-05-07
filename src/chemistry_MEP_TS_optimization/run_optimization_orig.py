import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from functools import partial
from typing import NamedTuple
from matplotlib import pyplot as plt

from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController, Tsit5, DirectAdjoint
#from jaxopt import Bisection

from src import mechanics
from src import tools
from src import paths
from src.optimization import path_metrics as metrics
from src.tools import visualize
from src.potentials import get_potential
from src.optimization import get_optimizer
from src.paths import initialize as init

from src.potentials.wolfe_schlegel import ws 


class Points(NamedTuple):
    initial : jnp.array
    final : jnp.array

class LayerParams(NamedTuple):
    weight : jnp.ndarray
    bias : jnp.ndarray

def init_network_params(key, width, depth, points):
    sizes = [1] + [width]*depth + [len(points.final)]
    keys = random.split(key, len(sizes))
    return [
        LayerParams(*init.random_layer_params(n, m, k))\
        for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]

def predict_(params, time):
    activation = jnp.array(time)
    for p in params[:-1]:
        linear = jnp.matmul(activation, p.weight) + p.bias
        activation = jax.nn.gelu(linear)
    
    return jnp.matmul(activation, params[-1].weight) + params[-1].bias

#@jax.jit
def predict(params, points, time):
    print("params", params)
    return predict_(params, time)\
        - (1 - time)*(predict_(params, jnp.array([0])) - points.initial)\
        - time*(predict_(params, jnp.array([1.])) - points.final)
 
if __name__ == "__main__":
    # Setup environment
    arg_parser = tools.build_default_arg_parser()
    args = arg_parser.parse_args()

    config = tools.import_run_config(args.name, args.path, args.tag)
    print("fin config", config)

    # Create output directories
    output_dir = os.path.join(args.output_dir, config.potential, config.optimizer)
    log_dir = os.path.join(output_dir, "logs")
    if not os.path.exists(output_dir):
        os.makedirs(log_dir)
    

    # Get path calculation method
    points = Points(jnp.array(config.initial_point), jnp.array(config.final_point))
    path_params = init_network_params(random.PRNGKey(123), config.path_params["width"], config.path_params["depth"], points)
    test = predict(path_params, points, jnp.array([0.2]))
    print("testing path", test)
    test = predict(path_params, points, jnp.array([0.5]))
    print("testing path", test)
    """
    path = paths.get_path(
        config.path,
        config.initial_point,
        config.final_point,
        **config.path_params
    )
    """

    """
    fig, ax = plt.subplots()
    for t in np.arange(0,100):
        point = path.eval(t/99)
        print(point)
        ax.scatter(t/99., point[0])
    fig.savefig("testpath.png")
    """

    potential = ws
    solver = Dopri5()
    save_at = SaveAt(dense=True)
    stepsize_controller = PIDController(rtol=1e-1, atol=1e-1)
    
    
    def integrate(params, t_init = 0., t_final = 1.):
    
        term = ODETerm(
            lambda t, y, *args: jnp.linalg.norm(potential(predict(*args, points, jnp.array([t]))))
        )
        solution = diffeqsolve(
            term,
            solver = Tsit5(scan_kind="bounded"),
            t0=t_init,
            t1=t_final,
            dt0=None,
            y0=0,
            args=params,
            #saveat=save_at,
            stepsize_controller=stepsize_controller,
            max_steps=int(1e5),
            #adjoint=DirectAdjoint()
        )
        return solution.ys[0]

    print("test integrate", integrate(path_params))
    """
    # Get chemical potential
    potential = get_potential(
        config.potential,
        tag=config.potential_tag,
        expect_config=config.potential!="constant"
    )
    """

    """
    #print("potential", potential.eval(jnp.array([1])))
    odeInt = metrics.ODEintegrator(potential)
    odeInt.path = path
    #jac = jax.jacfwd(odeInt.fxn)
    #print("test jac", jac(0.), jac(0.5), jac(1.0))
    print("integral", odeInt.integrate(path))
    #print("Jacobian", odeInt.jacobian(0.5,9))
    sol = metrics.path_integral(potential, path)
    print(sol.evaluate(1.))
    sys.exit(0)
    """

    plot_times = jnp.expand_dims(jnp.arange(100, dtype=float), 1)/99
    def loss_fxn(params):
        #return jnp.sum(jnp.array(predict(params, points, plot_times)))
        return integrate(params)

    learning_rate = 0.1
    grad_fxn = jax.grad(loss_fxn)
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



