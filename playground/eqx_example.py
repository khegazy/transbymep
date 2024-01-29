import time

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax


is_test = True
class Func(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, data_size, out_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.mlp(jnp.array([t]))


class TestFunc(eqx.Module):
    weight : jnp.array

    def __init__(self, data_size, out_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.weight = jnp.ones((data_size, out_size))

    def __call__(self, t, y, args):
        return jnp.matmul(jnp.array([t]), self.weight)


class NeuralODE(eqx.Module):
    func: Func

    def __init__(self, data_size, out_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        if is_test:
            self.func = TestFunc(data_size, out_size, width_size, depth, key=key)
        else:
            self.func = Func(data_size, out_size, width_size, depth, key=key)

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys

key = jrandom.PRNGKey(0)
path = NeuralODE(1, 2, 4, 2, key=key)

@eqx.filter_value_and_grad
def grad_loss(model, ti, yi):
    #y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])
    y_pred = model(ti, jnp.array([0,0]))
    if is_test:
        return jnp.linalg.norm(y_pred) + jnp.linalg.norm(model.func.weight)
    else:
        return jnp.linalg.norm(y_pred) + jnp.linalg.norm(model.func.mlp.layers[0].weight)

loss, grads = grad_loss(path, jnp.arange(10)/9, jnp.arange(10))
print("loss", loss)
print("grad", grads) 

print("starting weights")
optim = optax.adabelief(0.01)
opt_state = optim.init(eqx.filter(path, eqx.is_inexact_array))
for i in range(150):
    if i%10 == 0 and i > 0:
        if is_test:
            weight_norm = jnp.linalg.norm(path.func.weight)
        else:
            weight_norm = jnp.linalg.norm(path.func.mlp.layers[0].weight)
        print("loss and weights at", i, loss, weight_norm)
    loss, grads = grad_loss(path, jnp.arange(10)/9, jnp.arange(10))
    updates, opt_state = optim.update(grads, opt_state)
    path = eqx.apply_updates(path, updates)

"""
@eqx.filter_jit
def make_step(ti, yi, model, opt_state):
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state
"""    