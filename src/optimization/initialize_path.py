import jax
import optax  # https://github.com/deepmind/optax
import jax.numpy as jnp
import jax.random as jrnd
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt

def randomly_initialize_path(path, filter_spec, n_points, order_points=False, seed=1910):
    key = jrnd.key(seed)
    times = jrnd.uniform(key, shape=(n_points, 1), minval=0.1, maxval=0.9)
    times = jnp.expand_dims(jnp.linspace(0, 1, n_points+2)[1:-1], -1)

    n_dims = len(path.initial_point)
    rnd_dims = []
    for idx in range(n_dims):
        _, key = jrnd.split(key)
        min_val = jnp.min(
            jnp.array([path.initial_point[idx], path.final_point[idx]])
        )
        max_val = jnp.max(
            jnp.array([path.initial_point[idx], path.final_point[idx]])
        )
        print("MIN MAX", min_val, max_val)
        rnd_vals = jrnd.uniform(key, (n_points, 1), minval=min_val, maxval=max_val)
        if order_points or idx == 0:
            if path.initial_point[idx] > path.final_point[idx]:
                rnd_dims.append(-1*jnp.sort(-1*rnd_vals, axis=0))
            else:
                rnd_dims.append(jnp.sort(rnd_vals, axis=0))
        else:
            rnd_dims.append(rnd_vals)
    
    return initialize_path(
        path, filter_spec, times, jnp.concatenate(rnd_dims, axis=-1)
    )


def loss_init(diff_path, static_path, times, points):
    path = eqx.combine(diff_path, static_path)
    preds = jax.vmap(path.geometric_path, (0))(times)
    return jnp.mean((points - preds)**2)


def initialize_path(path, filter_spec, times, init_points, lr=0.001, max_steps=5000):

    print("INFO: Beginning path initialization")
    loss_grad_fxn = eqx.filter_value_and_grad(loss_init)
    loss, prev_loss = 2e-10, 1e-10
    optimizer = optax.adam(lr,)
    opt_state = optimizer.init(eqx.filter(path, eqx.is_inexact_array))
    idx, rel_error = 0, 100
    while (idx < 1500 or loss > 1e-8) and idx < max_steps:
        diff_path, static_path = eqx.partition(path, filter_spec)
        prev_loss = loss
        loss, grads = loss_grad_fxn(diff_path, static_path, times, init_points)

        updates, opt_state = optimizer.update(grads, opt_state)
        path = eqx.apply_updates(path, updates)
        rel_error = jnp.abs(prev_loss - loss)/prev_loss
        idx = idx + 1
        if idx % 250 == 0:
            print(f"\tIteration {idx}: Loss {loss} | Relative Error {rel_error}")
            fig, ax = plt.subplots()
            geo_path, _ = path.get_path()
            ax.plot(init_points[:,0], init_points[:,1], 'ob')
            ax.plot(geo_path[:,0], geo_path[:,1], '-k')
            fig.savefig(f"./plots/initialization/init_path_{idx}.png")

        #print(prev_loss, loss, jnp.abs(prev_loss - loss)/prev_loss)
    
    print(f"INFO: Finished path initialization after {idx} iterations")
    fig, ax = plt.subplots()
    geo_path, _ = path.get_path()
    ax.plot(init_points[:,0], init_points[:,1], 'ob')
    ax.plot(geo_path[:,0], geo_path[:,1], '-k')
    fig.savefig("./plots/init_path.png")

    return path









