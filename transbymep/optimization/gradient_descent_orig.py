import jax
import jax.numpy as jnp
from functools import partial

from transbymep.tools.logging import logging
from . import path_metrics as path_tools


@jax.jit
def update(
        params: dict,
        grad_fxn: callable,
        metrics: dict,
        learning_rate: float
) -> dict:
    """
    Update parameters using gradient descent.

    Parameters:
    -----------
    params : dict
        Model parameters.
    grad_fxn : callable
        Gradient function.
    metrics : dict
        Metrics.
    learning_rate : float
        Learning rate.

    Returns:
    --------
    dict
        Updated parameters.
    """
    grads = grad_fxn(params, metrics)
    return jax.tree_map(
        lambda param, g: param - g*learning_rate, params, grads
    )


class gradientDescent_(path_tools.ODEintegrator, logging):
    def __init__(
            self,
            potential: callable,
            path: object,
            loss_fxn: callable,
            metric_fxn: callable,
            config: dict,
            max_n_steps: int = 1e9,
    ):
        """
        Initialize the gradient descent optimizer.

        Parameters:
        -----------
        potential : callable
            The potential function.
        path : object
            The path object.
        loss_fxn : callable
            The loss function.
        metric_fxn : callable
            The metric function.
        config : dict
            Configuration dictionary.
        max_n_steps : int, optional
            Maximum number of steps (default is 1e9).
        """
        super().__init__(potential, path)
        self.potential = potential
        self.path
        self.loss_fxn = loss_fxn
        self.metric_fxn = metric_fxn
        self.config = config
        self.max_n_steps = max_n_steps
        self.grad_fxn = jax.grad(self.loss_fxn)


    def find_critical_path(
            self,
            n_steps: int = 10,
            log_frequency: int = 1000
    ) ->  None:
        """
        Find the critical path.

        Parameters:
        -----------
        n_steps : int, optional
            Number of steps (default is 10).
        log_frequency : int, optional
            Logging frequency (default is 1000).
        """
        print("computing critical_path...")
        n_steps = n_steps if n_steps is not None else self.max_n_steps

        for step in range(n_steps):
            metrics = self.path_integral()
            self.path.params = update(
                self.path.params,
                self.grad_fxn,
                self.metric_fxn,
                self.learning_rate
            )
            """
            if step % log_frequency == 0:
                self.training_logger(
                    step,
                    self.action(self.potential, points, start, end)
                )
            """

    
    @partial(jax.jit, static_argnums=[0])
    def update_path(self) -> float:
        """
        Update the path.

        Returns:
        --------
        float
            Loss value.
        """
        loss = self.loss_fxn(self.path, self.potential)
        grads = jax.grad(loss)(self.path.weights)
        self.path.weights = [(w - self.step_size*dw, b - self.step_size*db)\
            for (w,b), (dw, db) in zip( self.path.weights, grads)]
        """
        new_points = points -  self.path_step_factor*jax.grad(
            self.action, argnums=1)(
                self.potential,
                points,
                start,
                end,
            )
        """

        return loss