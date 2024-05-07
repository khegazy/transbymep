import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, Dopri5, ODETerm, Tsit5
from diffrax import backward_hermite_coefficients, CubicInterpolation
import equinox as eqx
import jax.nn as jnn


def vector_field2(t, y, interp):
    return -y + interp.evaluate(t)

def predict_float(param, time):
    return jnp.linalg.norm(jnp.matmul(jnp.array([time]), param))

def int_fxn(t, y, args):
    return jnp.linalg.norm(args*t)

class Func(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            key=key,
        )

    def __call__(self, t, y, args):
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
def solve(path_):
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