import jax.numpy as jnp
import numpy as np
from .visualize import plot_path
class logging():
    def __init__(self, *args, **kwargs):
        return
    
    def training_logger(self, step, val):
        step_string = ("step: " + str(step)).ljust(15)
        val_string = "val: " + str(val)
        print(step_string, val_string)
    
    def optimization_step(
            self,
            step,
            path,
            potential,
            loss,
            grads,
            plot=True,
            geo_paths=None,
            pes_paths=None,
            add_azimuthal_dof=False,
            add_translation_dof=False
        ):
        print(f"Step {step} | Loss: {loss}")
        #print(path.total_grad_path(0.55, 0.))
        """
        for ii in range(len(path.mlp.layers)):
            print(f"W{ii} sum: {jnp.sum(path.mlp.layers[ii].weight)}")
            print(f"Wg{ii} sum: {jnp.sum(grads.mlp.layers[ii].weight)}")
        """
        #print(path.mlp.layers[0].weight)
        #print("test grad", grads.mlp.layers[0].weight)
        geo_path, pes_path = path.get_path()
        if geo_paths is not None:
            geo_paths.append(geo_path)
        if pes_paths is not None:
            pes_paths.append(pes_path)
        if plot:
            plot_path(
                geo_path, f"test_plot_{step:03d}", pes_fxn=potential,
                plot_min_max=(-2, 2, -2, 2),
                levels=np.arange(-100, 100, 5),
                add_translation_dof=add_translation_dof,
                add_azimuthal_dof=add_azimuthal_dof
            )
        return geo_paths, pes_paths