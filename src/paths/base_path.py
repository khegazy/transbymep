import jax
import jax.numpy as jnp
import equinox as eqx

from ..tools import metrics
from ..potentials.base_class import PotentialBase

class BasePath(eqx.Module):
    initial_point: jnp.array
    final_point: jnp.array
    potential: PotentialBase
    #point_option: int
    #point_arg: float

    def __init__(
        self,
        potential,
        initial_point,
        final_point,
        #add_azimuthal_dof=False,
        #add_translation_dof=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.potential = potential
        self.initial_point = jnp.array(initial_point)
        self.final_point = jnp.array(final_point)
        """
        self.point_option = 0
        self.point_arg = 0
        if add_azimuthal_dof:
            self.point_option = 1
            self.point_arg = add_azimuthal_dof
        elif add_translation_dof:
            self.point_option = 2
        """

    def tree_filter_fxn(self, tree, get_len=False):
        if get_len:
            return 3
        return (
            tree.initial_point,
            tree.final_point,
            tree.potential,
            #tree.point_option,
            #tree.point_arg
        )

    """
    def point_transform(self, point):
        if self.point_option == 0:
            self.identity_transform(point)
        elif self.point_option == 0:
            self.azimuthal_transform(point, self.point_arg)
        elif self.point_option == 0:
            self.translation_transform(point)

    def identity_transform(self, point):
        return point

    def azimuthal_transform(self, point, shift):
        return jnp.concatenate([
            [jnp.sqrt(point[0]**2 + point[-1]**2) - shift],
            point[1:-1]
        ])

    def translation_transform(self, point):
        return jnp.concatenate([
            [point[0] + point[-1]],
            point[1:-1]
        ])
    """
    
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
    
    def E_vre(self, t, y, *args):
        return metrics.E_vre(*self.total_grad_path(t, y, *args))
    
    def E_pvre(self, t, y, *args):
        return metrics.E_pvre(*self.total_grad_path(t, y, *args))
    
    def E_pvre_mag(self, t, y, *args):
        return metrics.E_pvre_mag(*self.total_grad_path(t, y, *args))
    
    def vre_residual(self, t, y, *args):
        return metrics.vre_residual(*self.total_grad_path(t, y, *args))