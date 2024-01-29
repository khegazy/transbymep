import jax
import jax.numpy as jnp
import numpy as np


class ElasticBand():
    def __init__(
            self,
            initial_point,
            final_point,
            n_images=50,
            special_point=None
        ):
        self.initial_point = jnp.array(initial_point)
        self.final_point = jnp.array(final_point)
        if special_point is None:
            self.special_point = jnp.array(initial_point) + jnp.array(final_point)
            self.special_point = self.special_point/2
        else:
            self.special_point = jnp.array(special_point)
        self.n_images = n_images

        images_1 = self.compute_initial_points(
            self.initial_point, self.special_point, self.n_images//2
        )
        images_2 = self.compute_initial_points(
            self.special_point, self.final_point, self.n_images//2
        )
        self.path = jnp.vstack([images_1, images_2])

            
    def compute_initial_points(self, start, end, n_images):
        ts = np.linspace(0.0, 1.0, n_images+1)[1:]
        points = [start*(1 - t) + end*t for t in ts]
        return jnp.stack(points)