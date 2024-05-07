import jax
import jax.numpy as jnp
from jax import random

def random_layer_params(m, n, key, scale=1):
    w_key, b_key = random.split(key)
    return scale*random.normal(w_key, (n, m)), scale*random.normal(b_key, (m,))