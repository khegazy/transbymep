import jax
import jax.numpy as jnp
from functools import partial

from .base_class import PotentialBase

def ws(point):
    x, y = point
    return 10*(x**4 + y**4 - 2*x*x - 4*y*y +
        x*y + 0.2*x + 0.1*y)

class wolfe_schlegel(PotentialBase):
    def __init__(self, **kwargs):
        if "minima" in kwargs:
            self.minima = kwargs["minima"]
        else:
            self.minima = None
    
    #@partial(jax.jit, static_argnums=(0,))
    def evaluate(self, point):
        #print("point", point)
        return 10*(point[0]**4 + point[1]**4 - 2*point[0]**2 - 4*point[1]**2 +
            point[0]*point[1] + 0.2*point[0] + 0.1*point[1])