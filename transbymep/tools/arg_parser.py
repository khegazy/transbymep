import argparse
from typing import Optional


def build_default_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        help="Path to YAML configuration file.",
        type=str,
        default="config.yaml"
    )
    return parser