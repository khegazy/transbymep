import jax
import jax.numpy as jnp

class PotentialBase:
    def __init__(self, add_azimuthal_dof=False, add_translation_dof=False, **kwargs) -> None:
        self.point_option = 0
        self.point_arg = 0
        if add_azimuthal_dof:
            self.point_option = 1
            self.point_arg = add_azimuthal_dof
        elif add_translation_dof:
            self.point_option = 2

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError
    
    def gradient(self, *args, **kwargs):
        raise NotImplementedError
    
    def point_transform(self, point, do_identity=False):
        if self.point_option == 0 or do_identity:
            return self.identity_transform(point)
        elif self.point_option == 1:
            return self.azimuthal_transform(point, self.point_arg)
        elif self.point_option == 2:
            return self.translation_transform(point)
    
    def identity_transform(self, point):
        return point

    def azimuthal_transform(self, point, shift):
        return jnp.concatenate([
            jnp.array([jnp.sqrt(point[0]**2 + point[-1]**2)]) - shift,
            point[1:-1]
        ])

    def translation_transform(self, point):
        return jnp.concatenate([
            [point[0] + point[-1]],
            point[1:-1]
        ])