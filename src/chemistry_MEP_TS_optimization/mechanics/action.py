import jax
import jax.numpy as jnp
from functools import partial
from .lagrangians import lagrangian


@partial(jax.jit, static_argnums=[0])
def action(
        potential,     # Function defining the potential energy
        points,        # Array of n points along the path
        start,         # Fixed start point
        end,           # Fixed end point
) -> float:
    """
    Calculate the action for a given path using the specified potential.

    Parameters:
    -----------
    potential : function
        Function defining the potential energy.
    points : array_like
        Array of n points along the path.
    start : array_like
        Fixed start point.
    end : array_like
        Fixed end point.

    Returns:
    --------
    float
        The action for the given path.
    """
    accumulator = lagrangian(potential, start, points[0])

    accumulator += sum(jnp.array(
        [lagrangian(potential, points[i], points[i+1])
         for i in range(0, points.shape[0] - 1)]))

    accumulator += lagrangian(potential, points[-1], end)

    return accumulator