import jax
import jax.numpy as jnp

@jax.jit
def muller_brown(point):
    x, y = point

    ai = [-200.0, -100.0, -170.0, 15.0]
    bi = [-1.0, -1.0, -6.5, 0.7]
    ci = [0.0, 0.0, 11.0, 0.6]
    di = [-10.0, -10.0, -6.5, 0.7]

    xi = [1.0, 0.0, -0.5, -1.0]
    yi = [0.0, 0.5, 1.5, 1.0]

    total = 0.0
    for i in range(4):
        total += ai[i] * jnp.exp(bi[i] * (x - xi[i]) * (x - xi[i]) +
                                 ci[i] * (x - xi[i]) * (y - yi[i]) +
                                 di[i] * (y - yi[i]) * (y - yi[i]))

    return  total