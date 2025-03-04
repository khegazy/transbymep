import jax
import jax.numpy as jnp
from functools import partial

from Popcornn.tools.logging import logging


class gradientDescent(logging):
    def __init__(
            self,
            potential: callable,
            config: dict,
            action: callable,
            minima_step_factor: float = None,
            minima_num_steps: int = None,
            path_step_factor: float = None,
            path_num_steps: int = None,
    ):
        """
        Initialize the GradientDescent optimizer.

        Parameters:
        -----------
        potential : callable
            The potential function.
        config : dict
            Configuration dictionary.
        action : callable
            The action function.
        minima_step_factor : float, optional
            The step factor for minima (default is None).
        minima_num_steps : int, optional
            The number of steps for minima (default is None).
        path_step_factor : float, optional
            The step factor for paths (default is None).
        path_num_steps : int, optional
            The number of steps for paths (default is None).
        """
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


    @partial(jax.jit, static_argnums=[0])
    def update_minimum(self, point: jnp.ndarray) -> jnp.ndarray:
        """
        Update the point using gradient descent.

        Parameters:
        -----------
        point : jnp.ndarray
            The current point.

        Returns:
        --------
        jnp.ndarray
            The updated point.
        """
        # returns the new point, and the val / grad norm at the old point.

        grad = jax.grad(self.potential)(point)
        new_point = point - self.minima_step_factor*grad

        return new_point

    
    
    def find_minima(
            self,
            initial_points=None) -> list:
        """
        Find the minima of the potential function.

        Parameters:
        -----------
        initial_points : list, optional
            List of initial points (default is None).

        Returns:
        --------
        list
            List of minima points.
        """
        if "minima" in self.config:
            initial_points = self.config["minima"] if initial_points is None\
                else initial_points
        else:
            initial_points = [] if initial_points is None else initial_points
        
        self.minima = [
            self.find_minimum(jnp.array(point)) for point in initial_points
        ]
        return self.minima


    def find_minimum(
            self,
            point: jnp.ndarray,
            log_frequency: int = 1000
    ) -> jnp.ndarray:
        """
        Find the minimum point.

        Parameters:
        -----------
        point : jnp.ndarray
            The initial point.
        log_frequency : int, optional
            Logging frequency (default is 1000).

        Returns:
        --------
        jnp.ndarray
            The minimum point.
        """
        print("computing minima...")

        for step in range(self.minima_num_steps):
            point = self.update_minimum(point)
            if step % log_frequency == 0:
                self.training_logger(step, self.potential(point))

        return point
    
    def find_critical_paths(
            self,
            initial_points: jnp.ndarray,
            start: jnp.ndarray,
            end: jnp.ndarray,
            num_steps: int = None,
            log_frequency: int = 1000
    ) -> List[ jnp.ndarray]:
        """
        Find critical paths.

        Parameters:
        -----------
        initial_points : array_like
            Initial points.
        start : array_like
            Start point.
        end : array_like
            End point.
        num_steps : int, optional
            Number of steps (default is None).
        log_frequency : int, optional
            Logging frequency (default is 1000).

        Returns:
        --------
        list
            List of critical paths.
        """
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
    
    @partial(jax.jit, static_argnums=[0])
    def update_critical_path(self,
                             points: jnp.ndarray,
                             start: jnp.ndarray,
                             end: jnp.ndarray
                             ) -> jnp.ndarray:
        """
        Update critical path.

        Parameters:
        -----------
        points : jnp.ndarray
            Points.
        start : jnp.ndarray
            Start point.
        end : jnp.ndarray
            End point.

        Returns:
        --------
        jnp.ndarray
            Updated points.
        """
        new_points = points -  self.path_step_factor*jax.grad(
            self.action, argnums=1)(
                self.potential,
                points,
                start,
                end,
            )

        return new_points