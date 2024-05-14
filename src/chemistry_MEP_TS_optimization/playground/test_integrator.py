import jax
import jax.numpy as jnp
from jax import random
from typing import Callable
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController, Tsit5, DirectAdjoint


def potential(point: jnp.ndarray) -> float:
    """
    Calculate the potential energy of the system at a point.

    Args:
        point (jnp.ndarray): Input point.

    Returns:
        float: Potential energy.
    """
    return jnp.sum(point**2)


def predict(param: jnp.ndarray, time: float) -> jnp.ndarray:
    """
    Predict the position of a point at a given time.

    Args:
        param (jnp.ndarray): Parameters.
        time (float): Time.

    Returns:
        jnp.ndarray: Predicted position.
    """
    return jnp.matmul(jnp.array([time]), param)


def predict_float(param: jnp.ndarray, time: float) -> float:
    """
    Predict the position of a point at a given time and return its norm.

    Args:
        param (jnp.ndarray): Parameters.
        time (float): Time.

    Returns:
        float: Norm of the predicted position.
    """
    print("test", param, time)
    return jnp.linalg.norm(jnp.matmul(jnp.array([time]), param))


key = random.PRNGKey(0)
param = random.normal(key, (1, 2))

print("Predict")
print(param)
print(predict(param, 0.5))

print("Potential")
print(potential(predict(param, 0.5)))
grad_pot = jax.grad(potential)
print("Grad Potential", grad_pot(predict(param, 0.5)))

solver = Dopri5()
save_at = SaveAt(dense=True)

stepsize_controller = PIDController(rtol=1e-1, atol=1e-1)


def int_fxn(t, y, args):
    return predict_float(args, t)


# lambda t, y, *args: jnp.linalg.norm(potential(predict(*args, t)))
def integrate(params: jnp.ndarray) -> float:
    """
    Integrate the given function.

    Args:
        params (jnp.ndarray): Parameters.

    Returns:
        float: Result of integration.
    """
    t0 = 0.
    t1 = 1.
    solver = Tsit5()
    test_fxn = lambda t, y, *args : jnp
    term = ODETerm(int_fxn)
    solution = diffeqsolve(
        term, solver, t0, t1, dt0=0.1, y0=0, args=params
    )
    (y1,) = solution.ys
    return y1

print("Integrate", integrate(param))

grad_fxn = jax.grad(integrate)
print("Grad", grad_fxn(param))
