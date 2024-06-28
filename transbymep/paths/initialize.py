import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple

def random_layer_params(
        m: int,
        n: int,
        key: jax.random.PRNGKey,
        scale: float = 1
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate random weights and biases for a neural network layer.

    Parameters:
    -----------
    m : int
        Number of input neurons.
    n : int
        Number of output neurons.
    key : jax.random.PRNGKey
        The random key for generating the parameters.
    scale : float, optional
        Scaling factor for the random parameters (default is 1).

    Returns:
    --------
    Tuple[jnp.ndarray, jnp.ndarray]
        Tuple containing the randomly initialized weights and biases.
    """
    w_key, b_key = random.split(key)
    return scale*random.normal(w_key, (n, m)), scale*random.normal(b_key, (m,))
