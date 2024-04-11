import jax
import jax.numpy as jnp
from functools import partial

from .base_class import PotentialBase

class WolfeSchlegel(PotentialBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "minima" in kwargs:
            self.minima = kwargs["minima"]
        else:
            self.minima = None
    
    #@partial(jax.jit, static_argnums=(0,))
    def evaluate(self, point):
        x, y = self.point_transform(point)
        val = 10*(x**4 + y**4 - 2*x**2 - 4*y**2 + x*y + 0.2*x + 0.1*y)
        return val
    
    def gradient(self, point):
        x, y = self.point_transform(point)
        val = self.evaluate(point)
        # grad = jax.grad(self.evaluate)(point)
        grad = jnp.array([
            10*(4*x**3 - 4*x + y + 0.2),
            10*(4*y**3 - 8*y + x + 0.1)
        ])
        return val, grad
