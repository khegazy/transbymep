# transbymep

Before running one must install torchpathdiffeq. To do so you have to clone the repo and install in edit mode from the top level torchpathdiffeq directory.
```
pip install -e .[dev]
```

To run one must select the name of a config file
```
python3 run_optimization.py --name test_Epvre
```

Optionally, you can store the optimization process by passing the following arguments:

- `--make_opt_plots`: Plot PNG files to visualize various optimization steps.
- `--make_animation`: Create a GIF file to visualize the optimization process.

For example:

```
python3 run_optimization.py --name <config_file_name> --make_opt_plots --make_animation
```
