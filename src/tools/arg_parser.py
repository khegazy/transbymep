import argparse
from typing import Optional


def build_default_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Name, seed, directory
    parser.add_argument("--name", help="experiment name", required=True)
    parser.add_argument(
        "--potential_tag",
        help="potential name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--path_tag", 
        help="path tag", 
        type=str, 
        default="", 
    )
    parser.add_argument("--tag", help="run tag", type=str, default="")
    parser.add_argument("--seed", help="random seed", type=int, default=123)
    parser.add_argument(
        "--output_dir",
        help="top level output directory",
        type=str,
        default="./output"
    )

    # Path description
    parser.add_argument(
        "--path",
        help="name of the path calculation procedure",
        type=str,
        default="mlp",
        required=False
    )
    parser.add_argument(
        '--randomly_initialize_path',
        default=False,
        action='store_true',
        help="Will randomly initialize path"
    )

    # Chemical potential
    parser.add_argument(
        "--potential", 
        help="name of chemical potential", 
        type=str,
        default=None,
        required=False
    )

    # Optimizer
    parser.add_argument(
        "--optimizer", 
        help="name of MEP and TS optimizer", 
        type=str,
        default="gradient_descent",
    )
    parser.add_argument(
        '--minimize_end_points', default=False, action='store_true'
    )
    parser.add_argument(
        '--num_optimizer_iterations', default=10000, type=int
    )

    # Plotting
    parser.add_argument(
        '--make_animation', default=False, action='store_true'
    )

    parser.add_argument('--make_opt_plots', action='store_true',
                        help="Store plots of various optimization steps.")

    # Debuging and testing
    parser.add_argument(
        '--debug', default=False, action='store_true'
    )
    parser.add_argument(
        '--add_azimuthal_dof', default=None, type=float
    )
    parser.add_argument(
        '--add_translation_dof', default=None, type=float
    )
    return parser