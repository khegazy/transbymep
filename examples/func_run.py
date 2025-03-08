from ase.io import read, write
from popcornn import optimize_MEP

if __name__ == "__main__":
    # Create configurations, or read them from a file
    config = {
        "potential_params": {
            "potential": "repel",
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
            "path_ode_names": "E_geo",
        },
        "optimizer_params": {
            "optimizer": {
                "name": "adam",
                "lr": 1.0e-3,
            },
            "lr_scheduler": {
                "name": "cosine",
                "T_max": 1000,
            }
        },
        "num_optimizer_iterations": 1000,
        "num_record_points": 20,
    }

    # Read the initial images
    initial_images = read('configs/6445.xyz', index=':')

    # Run the optimization
    final_images, ts_image = optimize_MEP(initial_images, **config)

    # Write the final images
    write('configs/6445_popcornn.xyz', final_images)

