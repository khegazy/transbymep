import argparse
from typing import Optional


def build_default_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Name, seed, directory
    parser.add_argument("--name", help="experiment name", required=True)
    parser.add_argument("--seed", help="random seed", type=int, default=123)
    parser.add_argument(
        "--output_dir",
        help="top level output directory",
        type=str,
        default="./output"
    )

    # Chemical potential
    parser.add_argument(
        "--potential", 
        help="name of chemical potential", 
        type=str,
        default=None,
        required=True
    )

    # Optimizer
    parser.add_argument(
        "--optimizer", 
        help="name of MEP and TS optimizer", 
        type=str,
        default="gradient_descent",
    )

    return parser