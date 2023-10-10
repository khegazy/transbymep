import os
import jax
import jax.numpy as jnp
import numpy as np

from src import mechanics
from src import tools
from src.tools import visualize
from src.potentials import get_potential
from src.optimizations import get_optimizer



if __name__ == "__main__":
    arg_parser = tools.build_default_arg_parser()
    args = arg_parser.parse_args()

    # Create output directories
    output_dir = os.path.join(args.output_dir, args.potential, args.optimizer)
    log_dir = os.path.join(output_dir, "logs")
    if not os.path.exists(output_dir):
        os.makedirs(log_dir)
    
    # Get chemical potential
    potential, config = get_potential(args.potential)

    # Get optimizer
    optimizer = get_optimizer(args.optimizer)(
        potential=potential,
        config=config,
        action=mechanics.action,
    )

    minima = optimizer.find_minima()

    special_point = jnp.array([0.1, 0.1])

    initial_points_1 = tools.compute_initial_points(minima[0], special_point, 25)
    initial_points_2 = tools.compute_initial_points(special_point, minima[-1], 25)
    initial_path = jnp.vstack([initial_points_1, initial_points_2])
    print(initial_path)

    paths = optimizer.find_critical_path(initial_path, minima[0], minima[-1])

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



