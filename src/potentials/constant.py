import jax
import jax.numpy as jnp
from functools import partial

class Constant():
    def __init__(self, scale=1., **kwargs):
        self.scale = scale

    @partial(jax.jit, static_argnums=(0,)) 
    def eval(self, point):
        return self.scale