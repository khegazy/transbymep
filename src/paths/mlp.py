import jax
import jax.numpy as jnp
import equinox as eqx

from .base_path import BasePath


class MLPpath(BasePath):
    mlp: eqx.nn.MLP

    def __init__(
        self,
        potential,
        initial_point,
        final_point,
        n_embed=32,
        depth=3,
        seed=123,
    ):
        super().__init__(
            potential=potential,
            initial_point=initial_point,
            final_point=final_point,
        )
        key = jax.random.PRNGKey(seed)
        self.mlp = eqx.nn.MLP(
            in_size=1,
            out_size=self.final_point.shape[-1],
            width_size=n_embed,
            depth=depth,
            activation=jax.nn.softplus,
            key=key,
        )

    def geometric_path(self, time, y=None, *args):
        scale = 1.
        return self.mlp(time) * scale \
            - (1 - time) * (self.mlp(jnp.array([0.])) * scale - self.initial_point)\
            - time * (self.mlp(jnp.array([1.])) * scale - self.final_point)

    def get_path(self, times=None):
        if times is None:
            times = jnp.expand_dims(
                jnp.linspace(0, 1., 1000, endpoint=True), -1
            )
        elif len(times.shape) == 1:
            times = jnp.expand_dims(times, -1)
        
        geo_path = jax.vmap(self.geometric_path, in_axes=(0, None))(times, 0)
        # pot_path = jax.vmap(self.potential.energy, in_axes=(0))(geo_path)
        pot_path = self.potential.energy(geo_path)
        return geo_path, pot_path




"""
class MLPpath_orig(eqx.Module):
    mlp: eqx.nn.MLP
    initial_point: jnp.array
    final_point: jnp.array
    potential: PotentialBase

    def __init__(
        self,
        potential,
        initial_point,
        final_point,
        n_embed=32,
        depth=3,
        seed=123,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.potential = potential
        self.initial_point = jnp.array(initial_point)
        self.final_point = jnp.array(final_point)

        key = jax.random.PRNGKey(seed)
        self.mlp = eqx.nn.MLP(
            in_size=1,
            out_size=self.final_point.shape[-1],
            width_size=n_embed,
            depth=depth,
            activation=jax.nn.softplus,
            key=key,
        )

    def pes_path(self, t, y, *args):
        return self.potential.evaluate(self.geometric_path(t, y , *args))
    
    def pes_ode_term(self, t, y, *args):
        return self.potential.evaluate(self.geometric_path(jnp.array([t]), y , *args))
    
    def geometric_path(self, time, y, *args):
        return self.mlp(time)\
            - (1 - time)*(self.mlp(jnp.array([0.])) - self.initial_point)\
            - time*(self.mlp(jnp.array([1.])) - self.final_point)

    def get_path(self, times=None):
        if times is None:
            times = jnp.expand_dims(
                jnp.linspace(0, 1., 1000, endpoint=True), -1
            )
        elif len(times.shape) == 1:
            times = jnp.expand_dims(times, -1)
        
        geo_path = jax.vmap(self.geometric_path, in_axes=(0, None))(times, 0)
        pot_path = jax.vmap(self.potential.evaluate, in_axes=(0))(geo_path)
        return geo_path, pot_path
"""














"""
class Points(NamedTuple):
    initial : jnp.array
    final : jnp.array

class LayerParams(NamedTuple):
    weight : jnp.ndarray
    bias : jnp.ndarray

def predict_(params, time):
    activation = jnp.array([time])
    for p in params[:-1]:
        linear = jnp.matmul(activation, p.weight) + p.bias
        activation = jax.nn.gelu(linear)
    
    return jnp.matmul(activation, params[-1].weight) + params[-1].bias

@jax.jit
def predict(params, points, time):
    #print("mlp time", time.val)
    #if time > 1. or time < 0:
    #    raise ValueError("Input time runs between [0,1]")i
    return predict_(params, time)\
        - (1 - time)*(predict_(params, 0) - points.initial)\
        - time*(predict_(params, 1.) - points.final)


class MLPpath():
    def __init__(
            self,
            initial_point,
            
final_point
,
            depth=5,
            width=512,
            seed=185,
            debug=False,
        ):
        self.points = Points(jnp.array(initial_point), jnp.array(final_point))
        print("POINTS", self.points)
        self.width = width
        self.depth = depth
        self.params = None

        self.init_network_params(random.PRNGKey(seed))
        if debug:
            self.params = [(jnp.ones((1,len(final_point))), 0)]
        
        self.eval = partial(predict, self.params, self.points)
    
    def init_network_params(self, key):
        sizes = [1] + [self.width]*self.depth + [len(self.points.final)]
        keys = random.split(key, len(sizes))
        self.params = [
            LayerParams(*init.random_layer_params(n, m, k))\
            for m, n, k in zip(sizes[:-1], sizes[1:], keys)
        ]
"""