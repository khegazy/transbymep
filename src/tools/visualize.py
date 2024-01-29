import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial

@partial(jax.jit, static_argnums=[0,1,2,3,4])
def contour_vals(function, x_min, x_max, y_min, y_max, step_size=0.01):
    x_vals = jnp.arange(x_min, x_max, step_size)
    y_vals = jnp.arange(y_min, y_max, step_size)
    l,r = jnp.meshgrid(x_vals, y_vals)
    print("left", l)
    print("right", r)
    args = jnp.reshape(jnp.stack([l,r],axis=2), (-1, 2))
    print("args", args)
    z_vals = jax.vmap(function.evaluate, (0))(args)
    return x_vals, y_vals, jnp.reshape(z_vals, (x_vals.shape[0], y_vals.shape[0]))#jnp.apply_along_axis(function, 2, args)


def contour_2d(
        ax,
        function,
        x_min,
        x_max,
        y_min,
        y_max,
        levels,
    ):
    if x_min is None or x_max is None or y_min is None or y_max is None:
        raise ValueError("Must specify max and min for x and y when plotting contours.")
    if levels is None:
        raise ValueError("Must specify contour levels when plotting pes_fxn countours.")
 
    print("plotting contour")
    x_vals, y_vals, z_vals = contour_vals(function, x_min, x_max, y_min, y_max)
    ax.contour(x_vals, y_vals, z_vals, levels=levels)
    print("done")
    return ax


def plot_path(
        path,
        name,
        pes_fxn=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        levels=None,
        plot_dir="./plots/",
    ):

    fig, ax = plt.subplots()
    if pes_fxn is not None:
        ax = contour_2d(ax, pes_fxn, x_min, x_max, y_min, y_max, levels)
    
    ax.plot(path[:,0], path[:,1], color='r', linestyle='-')
    ax.set_title(name)
    fig.savefig(os.path.join(plot_dir, name+".png"))
    print("Plotted", os.path.join(plot_dir, name+".png"))
    return fig, ax


def animate_optimization_2d(
        paths,
        title,
        contour_file,
        pes_fxn=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        levels=None,
        plot_dir='./plots/'
    ):


    fig, ax = plt.subplots()
    if pes_fxn is not None:
        ax = contour_2d(ax, pes_fxn, x_min, x_max, y_min, y_max, levels)
    #ax = contour_2d(function, x_min, x_max, y_min, y_max, levels)
    ax.set_title(title)
    fig.savefig(contour_file + '.png')
    plot_artist = ax.plot([], [], color='red', linestyle='-')[0]


    def animation_function(path):
        plot_artist.set_data(path[:,0], path[:,1])
        return plot_artist


    ani = animation.FuncAnimation(fig, animation_function, frames=paths)
    ani.save(os.path.join(plot_dir, contour_file + ".gif"))

