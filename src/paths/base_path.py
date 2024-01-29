import jax
import jax.numpy as jnp
import equinox as eqx

from ..tools import metrics
from ..potentials.base_class import PotentialBase

class BasePath(eqx.Module):
    initial_point: jnp.array
    final_point: jnp.array
    potential: PotentialBase

    def __init__(
        self,
        potential,
        initial_point,
        final_point,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.potential = potential
        self.initial_point = jnp.array(initial_point)
        self.final_point = jnp.array(final_point)

    def geometric_path(self, time, y, *args):
        raise NotImplementedError()
    
    def get_path(self, times=None):
        raise NotImplementedError()
    
    def pes_path(self, t, y, *args):
        t = jnp.array([t]).transpose()
        return self.potential.evaluate(self.geometric_path(t, y , *args))
    
    def pes_ode_term(self, t, y, in_integral=True, *args):
        t = jnp.array([t]).transpose()
        return self.potential.evaluate(self.geometric_path(jnp.array([t]), y , *args))
        #return self.pes_path(jnp.array([t]), y, *args)
 
    def total_path(self, t, y, *args):
        t = jnp.array([t]).transpose()
        geo_path = self.geometric_path(t, y , *args)
        return geo_path, self.potential.evaluate(geo_path)
    
    def total_grad_path(self, t, y, *args):
        t = jnp.array([t]).transpose()
        geo_path, geo_grad = jax.jvp(self.geometric_path, (t, y), (jnp.ones_like(t), 1.))
        """
        geo_path_ = self.geometric_path(t, y , *args)
        jax.debug.print("geo0 at {}: {}", t, geo_path_)
        jax.debug.print("geo1 at {}: {}",t, geo_path)
        jax.debug.print("geoG at {}: {}",t, geo_grad)
        """
        pes_path, pes_grad = eqx.filter_value_and_grad(self.potential.evaluate)(geo_path)
        return geo_path, geo_grad, pes_path, pes_grad
    
    def vre_residual(self, t, y, *args):
        return metrics.vre_residual(*self.total_grad_path(t, y, *args))