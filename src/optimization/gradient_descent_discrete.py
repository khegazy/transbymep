import jax
import jax.numpy as jnp
from functools import partial

from src.tools.logging import logging


class gradientDescent(logging):
    def __init__(
            self,
            potential,
            config,
            action,
            minima_step_factor=None,
            minima_num_steps=None,
            path_step_factor=None,
            path_num_steps=None,
    ):
        super().__init__()
        self.potential = potential
        self.config = config
        self.action = action

        if "minima_step_factor" in self.config and minima_step_factor is None:
            self.minima_step_factor = self.config["minima_step_factor"]
        else:
            assert minima_step_factor is not None
            self.minima_step_factor = minima_step_factor
        if "minima_num_steps" in self.config and minima_num_steps is None:
            self.minima_num_steps = self.config["minima_num_steps"]
        else:
            assert minima_num_steps is not None
            self.minima_step_factor = minima_num_steps
        
        if "path_step_factor" in self.config and path_step_factor is None:
            self.path_step_factor = self.config["path_step_factor"]
        else:
            assert path_step_factor is not None
            self.minima_step_factor = path_step_factor
        if "path_num_steps" in self.config and path_num_steps is None:
            self.path_num_steps = self.config["path_num_steps"]
        else:
            assert path_num_steps is not None
            self.path_num_steps = path_num_steps


    # @partial(jax.jit, static_argnums=[0])
    def update_minimum(self, point):
        """
        returns the new point, and the val / grad norm at the old point.
        """
        grad = jax.grad(self.potential)(point)
        new_point = point - self.minima_step_factor*grad

        return new_point

    
    
    def find_minima(self, initial_points=None):
        if "minima" in self.config:
            initial_points = self.config["minima"] if initial_points is None\
                else initial_points
        else:
            initial_points = [] if initial_points is None else initial_points
        
        self.minima = [
            self.find_minimum(jnp.array(point)) for point in initial_points
        ]
        return self.minima


    def find_minimum(self, point, log_frequency=1000):
        """
        loop for finding minima
        """

        print("computing minima...")

        for step in range(self.minima_num_steps):
            point = self.update_minimum(point)
            if step % log_frequency == 0:
                self.training_logger(step, self.potential(point))

        return point
    
    def find_critical_paths(
            self,
            initial_points,
            start,
            end,
            num_steps=None,
            log_frequency=1000
    ):

        print("computing critical_path...")
        result = []
        points = initial_points
        result.append(points)
        num_steps = num_steps if num_steps is not None else self.path_num_steps

        for step in range(num_steps):
            points = self.update_critical_path(points, start, end)
            if step % log_frequency == 0:
                result.append(points)
                self.training_logger(
                    step,
                    self.action(self.potential, points, start, end)
                )

        result.append(points)

        print("\n\n\n")
        return result
    
    # @partial(jax.jit, static_argnums=[0])
    def update_critical_path(self, points, start, end):

        new_points = points -  self.path_step_factor*jax.grad(
            self.action, argnums=1)(
                self.potential,
                points,
                start,
                end,
            )

        return new_points