import os
import pytest
import argparse
from transbymep import tools
from transbymep.examples.sample import add
from run_optimization import run_opt


@pytest.mark.parametrize(
    "name, run_config_dir, path_config_dir, path_tag, tag, seed, output_dir, path, optimizer, num_optimizer_iterations, expected_path_integral",
    [
        ('Epvre_test', '../configs/runs', '../configs/paths', "", "", 123, "./output", "mlp", "gradient_descent", 5, 131.2415),
        # Add more test cases with different input parameters as needed
    ]
)
def test_run_opt(tmp_path, monkeypatch, name, run_config_dir, path_config_dir, path_tag, tag, seed, output_dir, path, optimizer, num_optimizer_iterations, expected_path_integral):
    input_dict = {
        "name": name,
        "run_config_dir": run_config_dir,
        "potential_tag": None,
        "path_tag": path_tag,
        "tag": tag,
        "seed": seed,
        "output_dir": output_dir,
        "path": path,
        "path_config_dir": path_config_dir,
        'randomly_initialize_path': None,
        "potential": None,
        "optimizer": optimizer,
        'minimize_end_points': False,
        'num_optimizer_iterations': num_optimizer_iterations,
        'make_animation': False,
        'make_opt_plots': 0,
        'debug': False,
        'add_azimuthal_dof': None,
        'add_translation_dof': None
    }

    args = argparse.Namespace(**input_dict)
    file_dir = os.path.dirname(os.path.abspath(__file__))

    monkeypatch.chdir(tmp_path)
    logger = tools.logging()
    config = tools.import_run_config(
        args.name,
        path_tag=args.path_tag,
        tag=args.tag,
        dir=os.path.join(file_dir, args.run_config_dir),
        flags=args
    )
    path_config = tools.import_path_config(
        config,
        path_tag=args.path_tag,
        dir=os.path.join(file_dir, args.path_config_dir)
    )
    path_integral = run_opt(args, config, path_config, logger)
    assert path_integral == pytest.approx(expected_path_integral, abs=1, rel=1)
