import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from .base_class import PotentialBase

class LennardJones(PotentialBase):
    def __init__(self, epsilon=1.0, sigma=1.0, **kwargs):
        super().__init__(**kwargs)
        if "minima" in kwargs:
            self.minima = kwargs["minima"]
        else:
            self.minima = None
        self.epsilon = epsilon
        self.sigma = sigma
    
    #@partial(jax.jit, static_argnums=(0,))
    def energy(self, point):
        val = 0.0
        for i in range(len(point)):
            for j in range(i+1, len(point)):
                r = jnp.linalg.norm(point[i] - point[j])
                val += 4*self.epsilon*((self.sigma/r)**12 - (self.sigma/r)**6)
        return val
