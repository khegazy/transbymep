import jax
import jax.numpy as jnp
import equinox as eqx
from transbymep.tools import metrics


def pes_integral(path, integrator):
    return integrator.path_integral(path.pes_path)


def E_vre_integral(path, integrator):
    return integrator.path_integral(path.E_vre)


def E_pvre_integral(path, integrator):
    return integrator.path_integral(path.E_pvre)


def E_pvre_mag_integral(path, integrator):
    return integrator.path_integral(path.E_pvre_mag)


def vre_residual_integral(path, integrator):
    return integrator.path_integral(path.vre_residual)


loss_dict = {
    'pes' : pes_integral,
    'e_vre' : E_vre_integral,
    'e_pvre' : E_pvre_integral,
    'e_pvre_mag' : E_pvre_mag_integral,
    'vre_residual' : vre_residual_integral
}


def get_loss(loss_types):
    loss_subfxns = []
    for name, (weight, config) in loss_types.items():
        if name.lower() not in loss_dict:
            raise ValueError(f"get_loss does not recognize loss {name.lower()}, either add it to loss_dict or use {list(loss_dict.keys())}.")
        loss_subfxns.append((loss_dict[name], weight, config))
    
    def loss_fxn(diff_path, static_path, integrator):
        path = eqx.combine(diff_path, static_path)
        loss = 0
        for fxn, weight, config in loss_subfxns:
            loss = loss + weight*fxn(path, integrator, **config)
        return loss

    return eqx.filter_value_and_grad(loss_fxn), loss_fxn


"""
@eqx.filter_value_and_grad
def path_integral(diff_path, stat_path, integrator):
    path = eqx.combine(diff_path, stat_path)
    pes_path = integrator.path_integral(path)
    return pes_path
"""
    
"""
from . import integrator
class geodesic_loss(integrator.ODEintegrator):
    def __init__(self, potential):
        super().__init__(potential)
        self.length = None
"""
