import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=[0])
def lagrangian(
        potential,          # Function defining the potential energy
        left_point,         # Left point
        right_point,        # Right point
        distance_factor=100 # Scaling factor for the distance term
) -> float:
    """
    Calculate the Lagrangian for a given potential and two points.

    Parameters:
    -----------
    potential : function
        Function defining the potential energy.
    left_point : array_like
        Left point in the configuration space.
    right_point : array_like
        Right point in the configuration space.
    distance_factor : float, optional
        Scaling factor for the distance term (default is 100).

    Returns:
    --------
    float
        The Lagrangian for the given potential and points.
    """
    displacement = right_point - left_point
    squares = displacement * displacement
    graph_component = (potential(right_point) - potential(left_point)) ** 2
    return ( jnp.exp(distance_factor*squares.sum()) - 1.0 +
        graph_component )