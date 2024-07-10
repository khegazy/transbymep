import jax
import jax.numpy as jnp
from functools import partial

from transbymep.tools.logging import logging


@jax.jit
def update(params, grad_fxn, metrics, learning_rate):
    grads = grad_fxn(params, metrics)
    return jax.tree_map(
        lambda param, g: param - g*learning_rate, params, grads
    )


class gradientDescent:
    """
    Gradient descent optimizer.

    Attributes:
        path (object): The path to optimize.
        integrator (object): The integrator to compute gradients.
        loss_fxn (callable): The loss function to minimize.
        config (dict): Configuration parameters.
        max_n_steps (int): Maximum number of optimization steps.
    """
    def __init__(
            self,
            path,
            integrator,
            loss_fxn,
            config,
            max_n_steps: int = int(1e9),
    ) -> None:
        self.path = path
        self.integrator = integrator
        self.loss_fxn = loss_fxn
        self.config = config
        self.max_n_steps = max_n_steps

    def find_critical_path(
            self,
            n_steps: int = 10,
            log_frequency: int = 1000
    ) -> None:
        """
        Find the critical path using gradient descent optimization.

        Args:
            n_steps (int, optional): Number of optimization steps. Defaults to 10.
            log_frequency (int, optional): Frequency of logging. Defaults to 1000.
        """
        print("computing critical_path...")
        n_steps = n_steps if n_steps is not None else self.max_n_steps

        for step in range(n_steps):
            loss, grad = self.loss_fxn(self.integrator)
            print("loss", loss)
            print("grad", grad)
            """
            self.path.params = update(
                self.path.params,
                self.grad_fxn,
                self.metric_fxn,
                self.learning_rate
            )
            """
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
        Update the path weights using gradient descent.

        Returns:
            float: Loss value after the update.
        """
        loss = self.loss_fxn(self.path, self.potential)
        grads = jax.grad(loss)(self.path.weights)
        self.path.weights = [(w - self.step_size*dw, b - self.step_size*db)
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
