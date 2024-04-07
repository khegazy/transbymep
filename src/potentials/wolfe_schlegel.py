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
        
class RotatedWolfeSchlegel(PotentialBase):
    def __init__(self, **kwargs):
        if "minima" in kwargs:
            self.minima = kwargs["minima"]
        else:
            self.minima = None
        if 'pes_rot_translation' in kwargs:
            self.pes_rot_translation = kwargs['pes_rot_translation']
        else:
            self.pes_rot_translation = 10
    
    def tree_filter_fxn(self, tree, get_len=False):
        if get_len:
            return 4
        return (tree.initial_point, tree.final_point, tree.potential, tree.pes_rot_translation)
    
    #@partial(jax.jit, static_argnums=(0,))
    def evaluate(self, point):
        x, y, z = point
        xz = jnp.sqrt(x**2 + z**2) - self.pes_rot_translation
        y = y - self.pes_rot_translation
        return 10*(xz**4 + y**4 - 2*xz**2 - 4*y**2\
            + xz*y + 0.2*xz + 0.1*y)