import jax
import jax.numpy as jnp
from functools import partial

# @partial(jax.jit, static_argnums=[0])
def lagrangian(potential, left_point, right_point, distance_factor=100):

    displacement = right_point - left_point
    squares = displacement * displacement
    graph_component = (potential(right_point) - potential(left_point)) ** 2
    return ( jnp.exp(distance_factor*squares.sum()) - 1.0 +
        graph_component )