import jax
import jax.numpy as jnp
import numpy as np

from .base_path import BasePath

class ElasticBand(BasePath):
    points: jnp.array
    times: jnp.array

    def __init__(
            self,
            potential,
            initial_point,
            final_point,
            mid_point=None,
            n_images=50,
        ):
        super().__init__(
            potential=potential,
            initial_point=initial_point,
            final_point=final_point,
        )
        self.times = jnp.linspace(0, 1, n_images+2)[1:-1]
        if mid_point is None:
            self.points = jax.vmap(self.interpnd, in_axes=(0, None, None))(self.times, jnp.array([0.0, 1.0]), jnp.array([initial_point[None, ...], final_point[None, ...]]))
        else:
            mid_point = jnp.array(mid_point)
            self.points = jax.vmap(self.interpnd, in_axes=(0, None, None))(self.times, jnp.array([0.0, 0.5, 1.0]), jnp.array([initial_point[None, ...], mid_point[None, ...], final_point[None, ...]]))
            
    def interpnd(self, x, xp, fp):
        fp = fp.reshape(fp.shape[0], -1)
        f = jax.vmap(lambda fp: jnp.interp(x, xp, fp))(fp.T).T
        f = f.reshape(*self.final_point.shape)
        return f
    
    def geometric_path(self, t, y=None, *args):
        times = jnp.concatenate([jnp.array([0.]), self.times, jnp.array([1.])])
        points = jnp.concatenate([self.initial_point[None, ...], self.points, self.final_point[None, ...]])
        order = jnp.argsort(times)
        return self.interpnd(t, times[order], points[order])

    def get_path(self, times=None):
        if times is None:
            times = jnp.expand_dims(
                jnp.linspace(0, 1., 1000, endpoint=True), -1
            )
        elif len(times.shape) == 1:
            times = jnp.expand_dims(times, -1)
        
        geo_path = jax.vmap(self.geometric_path, in_axes=(0, None))(times, 0)
        print("geo_path", geo_path.shape)
        pot_path = jax.vmap(self.potential.energy, in_axes=(0))(geo_path)
        return geo_path, pot_path