import jax
import jax.numpy as jnp

@jax.jit
def wolfe_schlegel(point):
    x, y = point
    return 10*(x**4 + y**4 - 2*x*x - 4*y*y +
        x*y + 0.2*x + 0.1*y)
