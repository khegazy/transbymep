import jax
from jax import grad
import jax.numpy as jnp

from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController

from jaxopt import Bisection


"""
def f(t):
    return t**2,t**3

vector_field = lambda t, y, args: jnp.linalg.norm(jax.jacfwd(f)(t))
"""


def f(t: float) -> float:
    """
    A function that returns square of a number.

    Args:
        t (float): Input value.

    Returns:
        float: Output value.
    """
    return t**2


jac = jax.jacfwd(f)


def vector_field(t: float, x: float, y: float) -> float:
    """
    Defines the vector field for the ordinary differential equation.

    Args:
        t (float): Time.
        x (float): First coordinate.
        y (float): Second coordinate.

    Returns:
        float: Result of the vector field computation.
    """
    print(t,x,y)
    return jnp.linalg.norm(jac(t))

term = ODETerm(vector_field)
solver = Dopri5()
# saveat = SaveAt(dense=True)
saveat = SaveAt(ts=[0.1*i for i in range(11)])
stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

sol = diffeqsolve(term, solver, t0=0, t1=1, dt0=None, y0=0, saveat=saveat,
                  stepsize_controller=stepsize_controller)

# length = sol.evaluate(1.)
# print("Curve length: ", length)
print(sol.ts)
print(sol.ys)

targets = (jnp.arange(8) + 1.)/10. * length
print(targets)

"""
bisec = Bisection(optimality_fun=lambda x, target: sol.evaluate(x) - target, lower=0., upper=1.)
for t in targets:
    result = bisec.run(target=t).params
    print(t, result, sol.evaluate(result))
"""
