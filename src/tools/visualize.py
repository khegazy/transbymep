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
    args = jnp.stack([l,r],axis=2)
    return x_vals, y_vals, jnp.apply_along_axis(function, 2, args)


def contour_2d(
        function,
        x_min,
        x_max,
        y_min,
        y_max,
        levels,
        paths,
        title,
        contour_file):

    x_vals, y_vals, z_vals = contour_vals(function, x_min, x_max, y_min, y_max)

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.contour(x_vals, y_vals, z_vals, levels=levels)
    fig.savefig(contour_file + '.png')
    plot_artist = ax.plot([],[],color='red', linestyle='none', marker='o')[0]


    def animation_function(path):
        plot_artist.set_data(path[:,0], path[:,1])
        return plot_artist


    ani = animation.FuncAnimation(fig, animation_function,frames=paths)
    ani.save(contour_file + '.gif')