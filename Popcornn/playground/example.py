import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, Dopri5, ODETerm, Tsit5
from diffrax import backward_hermite_coefficients, CubicInterpolation
import equinox as eqx
import jax.nn as jnn


def vector_field2(
        t: jnp.array,
        y: jnp.array,
        interp: CubicInterpolation
) -> jnp.array:
    """
    Defines the vector field for the ordinary differential equation.

    Args:
        t (jnp.array): Time.
        y (jnp.array): Input vector.
        interp (CubicInterpolation): Interpolation object.

    Returns:
        jnp.array: Result of the vector field computation.
    """
    return -y + interp.evaluate(t)


def predict_float(
        param: jnp.array,
        time: jnp.array
) -> jnp.array:
    """
    Predicts floating point values.

    Args:
        param (jnp.array): Model parameters.
        time (jnp.array): Time.

    Returns:
        jnp.array: Predicted values.
    """
    return jnp.linalg.norm(jnp.matmul(jnp.array([time]), param))


def int_fxn(
        t: jnp.array,
        y: jnp.array,
        args
) -> jnp.array:
    """
    Integrates a function.

    Args:
        t (jnp.array): Time.
        y (jnp.array): Input vector.
        args (float): Additional argument.

    Returns:
        jnp.array: Result of the integration.
    """
    return jnp.linalg.norm(args*t)


class Func(eqx.Module):
    """
    Function class using an MLP neural network.

    Args:
        data_size (int): Input size.
        width_size (int): Width of the MLP.
        depth (int): Depth of the MLP.
        key (jax.random.PRNGKey): Random key for initialization.
    """

    mlp: eqx.nn.MLP

    def __init__(
            self,
            data_size: int,
            width_size: int,
            depth: int,
            *,
            key: jax.random.PRNGKey,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            key=key,
        )

    def __call__(
            self,
            t: jnp.array,
            y: jnp.array,
            args
    ) -> jnp.array:
        """
        Compute the function output.

        Args:
            t (jnp.array): Time.
            y (jnp.array): Input data.
            args (float): Additional argument.

        Returns:
            jnp.array: Output of the function.
        """
        return jnp.linalg.norm(self.mlp(t))


path = Func(1, 2, 4, 2)
"""
class path(eqx.nn.Module):
    anything after works
    weights: array

    def _init__():
    def __call__():
        calculate spline/mlp
"""


#@jax.jit
def solve(path_: Func) -> jnp.array:
    """
    Solves the ordinary differential equation.

    Args:
        path_ (Func): Function representing the vector field.

    Returns:
        jnp.array: Solution of the differential equation.
    """
    t0 = 0
    t1 = 1.
    #ts = jnp.linspace(t0, t1, len(points))
    #coeffs = backward_hermite_coefficients(ts, points)
    #interp = CubicInterpolation(ts, coeffs)
    #term = ODETerm(vector_field2)
    #test_fxn = lambda t, y, args : jnp.linalg.norm(args)
    #term = ODETerm(test_fxn)
    term = ODETerm(path_)
    solver = Tsit5()
    dt0 = None,
    y0 = 0.
    #sol = diffeqsolve(term, solver, t0, t1, dt0, y0, args=interp)
    sol = diffeqsolve(term, solver, t0, t1, dt0, y0)
    (y1,) = sol.ys
    return y1

grad_fxn = jax.grad(solve)

points = jnp.array([3.0, 0.5, -0.8, 1.8])
print("Forward", solve(path))
print("Grads", grad_fxn(path))
