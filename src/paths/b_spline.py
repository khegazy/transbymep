import jax
import jax.numpy as jnp
import equinox as eqx

from .base_path import BasePath


class BSpline(BasePath):
    degree: int
    points: jnp.array
    knots: jnp.array

    def __init__(
        self,
        potential,
        initial_point,
        final_point,
        degree=2,
        n_anchors=4,
        **kwargs
    ):
        super().__init__(
            potential=potential,
            initial_point=initial_point,
            final_point=final_point,
            **kwargs
        )

        self.degree = degree
        delta_geo = (self.final_point - self.initial_point)/float(n_anchors + 2) 
        self.points = jnp.array([
            self.initial_point + delta_geo*(i + 1) for i in range(n_anchors)
        ])
        delta_time = 1./(n_anchors + 1)
        self.knots = jnp.array([
            (i + 1)/float(n_anchors + 1) for i in range(n_anchors)
        ])
        print("This method is not finished")
        raise NotImplementedError
    
    def geometric_path(self, time, y, *args):
        idx = self.degree + int(time/self.delta_time)
        time_diffs = time - self.knots[idx-(self.degree-1):idx+self.degree]