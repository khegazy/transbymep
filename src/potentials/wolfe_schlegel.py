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
        return 10*(x**4 + y**4 - 2*x**2 - 4*y**2\
            + x*y + 0.2*x + 0.1*y)
