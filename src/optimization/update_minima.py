import jax
import jax.numpy as jnp
from functools import partial

class MinimaUpdate():
    def __init__(self, potential, n_steps=10000, step_size=1e-2):
        self.potential = potential
        self.step_size = step_size
        self.n_steps = n_steps
    
    @partial(jax.jit, static_argnums=[0])
    def update_minimum(self, point):
        """
        returns the new point, and the val / grad norm at the old point.
        """
        grad = jax.grad(self.potential.evaluate)(point)
        new_point = point - self.step_size*grad

        return new_point

    def find_minima(self, initial_points=[]):
        self.minima = [
            self.find_minimum(jnp.array(point)) for point in initial_points
        ]
        return self.minima

    def find_minimum(self, point, log_frequency=1000):
        """
        loop for finding minima
        """

        print(f"computing minima ... {point}")
        for step in range(self.n_steps):
            point = self.update_minimum(point)
            #if step % log_frequency == 0:
            #    self.training_logger(step, self.potential(point))

        return point