import jax
import jax.numpy as jnp
from functools import partial
from .lagrangians import lagrangian


@partial(jax.jit, static_argnums=[0])
def action(
        potential,     # function defining graph
        points,        # n points
        start,         # start point. fixed
        end,           # end point. fixed
):

    accumulator = lagrangian(potential, start, points[0])

    accumulator += sum(jnp.array(
        [lagrangian(potential, points[i], points[i+1])
         for i in range(0, points.shape[0] - 1)]))

    accumulator += lagrangian(potential, points[-1], end)

    return accumulator