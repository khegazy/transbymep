import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torchpathdiffeq
import transbymep
from transbymep import tools


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        help="Path to YAML configuration file.",
        type=str,
        default="visualize_growing_string.yaml"
    )
    parser.add_argument(
        "--make_plots",
        help="Make individual plots for each time step",
        action='store_true'
    )
    args = parser.parse_args()
    
    config = tools.import_yaml(f"configs/{args.config}")
    #config = transbymep.tools.import_run_config(
    #    "configs/visualize_growing_string.yaml"
    #)

    """
    integrator = tools.ODEintegrator(**config['integrator_params'])
    optimizer = transbymep.optimization.PathOptimizer(
        path=None,
        **config['optimizer_params']
    )
    """
    t_even = torch.linspace(0, 1, 50)
    t_right = torch.sqrt(torch.linspace(0, 1, 50))
    t_left = torch.linspace(0, 1, 50)**2

    for weight_type, sched_types in config.items():
        for sched_name, scheds in sched_types.items():
            print(f"Starting {weight_type} {sched_name}")
            loss_fxn = tools.metrics.get_loss_fxn('growing_string', weight_type=weight_type)
            schedulers = tools.scheduler.get_schedulers(scheds)
            history = []

            for t_idx in range(0, 101):
                if t_idx % 2 == 0:
                    loss_scales = {
                        name : schd.get_value() for name, schd in schedulers.items()
                    }
                    integral_output = torchpathdiffeq.IntegralOutput(
                        integral=torch.tensor([0]),
                        t_init=torch.tensor([0]),
                        t_final=torch.tensor([1]),
                        t=t_even[:,None,None],
                        t_pruned=t_even[:,None,None]
                    )
                    loss_fxn.update_parameters(
                        integral_output=integral_output, **loss_scales
                    )
                    weights = loss_fxn.get_weights(
                        t_even[:,None], integral_output.t_init, integral_output.t_final
                    )
                    if 'gauss' in weight_type:
                        history.append((t_idx, weights, loss_fxn.weight_scale, loss_fxn.variance_scale))
                    elif 'poly' in weight_type or 'butter' in weight_type:
                        history.append((t_idx, weights, loss_fxn.weight_scale, loss_fxn.order))
                    else:
                        history.append((t_idx, weights, loss_fxn.weight_scale, None))
                    
                    if args.make_plots:
                        fig, ax = plt.subplots()
                        ax.plot(weights)
                        if 'gauss' in weight_type:
                            history.append((t_idx, weights, loss_fxn.weight_scale, loss_fxn.variance_scale))
                            ax.text(0.05, 0.1, f"Variance: {loss_fxn.variance_scale:.3f}", fontsize=16, transform=ax.transAxes)
                        elif 'poly' in weight_type or 'butter' in weight_type:
                            history.append((t_idx, weights, loss_fxn.weight_scale, loss_fxn.order))
                            ax.text(0.05, 0.1, f"Order: {loss_fxn.order:.3f}", fontsize=16, transform=ax.transAxes)
                        ax.text(0.6, 0.1, f"Weight: {loss_fxn.weight_scale:.3f}", fontsize=16, transform=ax.transAxes)
                        ax.set_yscale('log')
                        ax.set_ylim(5e-4, 5)
                        fig.savefig(f"plots/{weight_type}_{sched_name}_{t_idx}.png")
                for _, sched in schedulers.items():
                    sched.step()
            
            fig, ax = plt.subplots()
            ax.set_yscale('log')
            ax.set_ylim(1e-4, 1.5)
            ax.set_title(f"{weight_type}_{sched_name}_{t_idx}")
            plot_artist = ax.plot([], [], color='red', linestyle='-')[0]
            text_artists = [ax.text(0.03, 0.14, "", fontsize=16, transform=ax.transAxes)]
            text_artists.append(ax.text(0.6, 0.05, "", fontsize=16, transform=ax.transAxes))
            if 'poly' in weight_type or 'butter' in weight_type or 'gauss' in weight_type:
                text_artists.append(ax.text(0.03, 0.05, "", fontsize=16, transform=ax.transAxes))
            


            def animation_function(weight_info):
                t, weight, scale, width = weight_info
                plot_artist.set_data(np.linspace(0, 1, len(weight)), weight)
                text_artists[0].set_text(f"Time: {t:.0f}")
                text_artists[1].set_text(f"Weight: {scale:.3f}")
                if 'gauss' in weight_type:
                    text_artists[2].set_text(f"Variance: {width:.3f}")
                elif 'poly' in weight_type or 'butter' in weight_type:
                    text_artists[2].set_text(f"Order: {width:.3f}")
                ax.set_xlim(0, 1)

                return plot_artist

            print("HIST", len(history), len(history[0]))
            ani = animation.FuncAnimation(fig, animation_function, frames=history)
            ani.save(f"plots/{weight_type}_{sched_name}.gif")
            
                    
