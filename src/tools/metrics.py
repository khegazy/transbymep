import jax
import jax.numpy as jnp


def E_vre(geo_val, geo_grad, pes_val, pes_grad):
    return jnp.linalg.norm(pes_grad)*jnp.linalg.norm(geo_grad),\
        jnp.linalg.norm(geo_grad*pes_grad)

def vre_residual(geo_val, geo_grad, pes_val, pes_grad):
    e_vre, e_pvre = E_vre(geo_val, geo_grad, pes_val, pes_grad)
    #return e_vre - e_pvre
    return -1*e_pvre