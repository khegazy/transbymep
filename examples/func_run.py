from ase.io import read, write
from transbymep import optimize_MEP


# Create configurations, or read them from a file
config = {
    "potential_params": {
        "potential": "morse",
        "alpha": 2.0,
    },
    "path_params": {
        "name": "mlp",
        "n_embed": 16,
        "depth": 3,
        "activation": "gelu",
    },
    "integrator_params": {
        "method": "dopri5",
        "rtol": 1.0e-5,
        "atol": 1.0e-5,
        "computation": "parallel",
        "sample_type": "uniform",
        "path_loss_name": "integral",
        "path_ode_names": "E_vre",
    },
    "optimizer_params": {
        "optimizer": {
            "name": "adam",
            "lr": 1.0e-3,
            "weight_decay": 100.0,
        },
        "lr_scheduler": {
            "name": "one_cycle",
            "max_lr": 1.0e-3,
            "total_steps": 1000,
        }
    },
    "num_optimizer_iterations": 1000,
}

# Read the initial images
initial_images = read('configs/44939.xyz', index=':')

# Run the optimization
final_images = optimize_MEP(initial_images, **config)

# Write the final images
write('configs/44939_popcornn.xyz', final_images)

