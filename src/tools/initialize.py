import jax
import jax.numpy as jnp
import numpy as np

def compute_initial_points(start, end, number_of_points):
    ts = np.linspace(0.0, 1.0, number_of_points+1)[1:]
    points = [ start * ( 1 - t ) + end * t for t in ts ]
    return jnp.stack(points)