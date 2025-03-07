 import os
 import pytest
 import argparse

 from torchpathdiffeq import UNIFORM_METHODS
 from popcornn import tools, optimize_MEP


# @pytest.mark.parametrize(
#     "config, expected_path_integral",
#     [
#         ('../configs/Epvre_test.yaml', 131.2415),
#     ]
# )
# def test_serial_integrators(tmp_path, monkeypatch, config, expected_path_integral):
#     input_dict = {
#         'config': config,
#     }

#     # Create arguments and setup environment
#     args = argparse.Namespace(**input_dict)
#     file_dir = os.path.dirname(os.path.abspath(__file__))
#     monkeypatch.chdir(tmp_path)
    
#     # Get run config
#     config = tools.import_run_config(
#         os.path.join(file_dir, args.config)
#     )
#     config['integrator_params']['computation'] = 'serial'

#     # Loop over all serial integration methods
#     for method in UNIFORM_METHODS.keys():
#         config['integrator_params']['method'] = method
#         output = optimize_MEP(**config)
#         paths_integral = output.paths_integral
#         path_integral = paths_integral[-1]
#         error_message = f"Failed sequential integrator test for method {method}, expected {expected_path_integral} but got {path_integral}"
#         assert path_integral == pytest.approx(expected_path_integral, abs=1, rel=1), error_message
