import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import optax  # https://github.com/deepmind/optax


# Toy data
def get_data(
        dataset_size: int,
        *,
        key: jnp.ndarray
) -> tuple:
    """
    Generate toy dataset.

    Args:
        dataset_size (int): Size of the dataset.
        key (jnp.ndarray): Random key for data generation.

    Returns:
        tuple: Tuple of input and output data arrays.
    """
    x = jrandom.normal(key, (dataset_size, 1))
    y = 5 * x - 2
    return x, y


# Toy dataloader
def dataloader(
        arrays: tuple,
        batch_size: int,
        *,
        key: jnp.ndarray
) -> tuple:
    """
    Generate batches of data.

    Args:
        arrays (tuple): Tuple of input and output data arrays.
        batch_size (int): Size of each batch.
        key (jnp.ndarray): Random key for shuffling.

    Yields:
        tuple: Batch of input and output data arrays.
    """
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def main(
    dataset_size: int = 10000,
    batch_size: int = 256,
    learning_rate: float = 3e-3,
    steps: int = 1000,
    width_size: int = 8,
    depth: int = 1,
    seed: int = 5678,
) -> None:
    """
    Main function for training a neural network model.

    Args:
        dataset_size (int, optional): Size of the dataset. Defaults to 10000.
        batch_size (int, optional): Size of each batch. Defaults to 256.
        learning_rate (float, optional): Learning rate. Defaults to 3e-3.
        steps (int, optional): Number of training steps. Defaults to 1000.
        width_size (int, optional): Width of the MLP layers. Defaults to 8.
        depth (int, optional): Depth of the MLP. Defaults to 1.
        seed (int, optional): Random seed. Defaults to 5678.
    """
    data_key, loader_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 3)
    data = get_data(dataset_size, key=data_key)
    data_iter = dataloader(data, batch_size, key=loader_key)

    # Step 1
    model = eqx.nn.MLP(
        in_size=1, out_size=1, width_size=width_size, depth=depth, key=model_key
    )

    # Step 2
    filter_spec = jtu.tree_map(lambda _: False, model)
    print("initial filter_spec\n", filter_spec)
    filter_spec = eqx.tree_at(
        lambda tree: (tree.layers[-1].weight, tree.layers[-1].bias),
        filter_spec,
        replace=(True, True),
    )
    print("final filter_spec\n", filter_spec)

    diff_model, static_model = eqx.partition(model, filter_spec)
    print("diff model\n", diff_model)
    print("static model\n", static_model)

    # Step 3
    @eqx.filter_jit
    def make_step(model, x, y, opt_state):
        @eqx.filter_grad
        def loss(diff_model, static_model, x, y):
            model = eqx.combine(diff_model, static_model)
            pred_y = jax.vmap(model)(x)
            return jnp.mean((y - pred_y) ** 2)

        diff_model, static_model = eqx.partition(model, filter_spec)
        grads = loss(diff_model, static_model, x, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state

    # And now let's train for a short while -- in exactly the usual way -- and see what
    # happens. We keep the original model around to compare to later.
    original_model = model
    optim = optax.sgd(learning_rate)
    opt_state = optim.init(model)
    for step, (x, y) in zip(range(steps), data_iter):
        model, opt_state = make_step(model, x, y, opt_state)
    print(
        f"Parameters of first layer at initialisation:\n"
        f"{jtu.tree_leaves(original_model.layers[0])}\n"
    )
    print(
        f"Parameters of first layer at end of training:\n"
        f"{jtu.tree_leaves(model.layers[0])}\n"
    )
    print(
        f"Parameters of last layer at initialisation:\n"
        f"{jtu.tree_leaves(original_model.layers[-1])}\n"
    )
    print(
        f"Parameters of last layer at end of training:\n"
        f"{jtu.tree_leaves(model.layers[-1])}\n"
    )


if __name__ == '__main__':
    main()
