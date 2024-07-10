import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Any

from .base_path import BasePath


class BSpline(BasePath):
    """
    B-Spline path representation.

    Attributes:
    -----------
    degree : int
        The degree of the B-Spline.
    points : jnp.array
        The control points of the B-Spline.
    knots : jnp.array
        The knot vector of the B-Spline.
    """
    degree: int
    points: jnp.array
    knots: jnp.array

    def __init__(
        self,
        potential: callable,
        initial_point: jnp.array,
        final_point: jnp.array,
        degree: int = 2,
        n_anchors: int = 4,
        **kwargs: Any
    ) -> None:
        """
        Initialize the B-Spline path.

        Parameters:
        -----------
        potential : A callable function
            The potential function.
        initial_point : jnp.array
            The initial point of the path.
        final_point : jnp.array
            The final point of the path.
        degree : int, optional
            The degree of the B-Spline (default is 2).
        n_anchors : int, optional
            The number of anchor points (default is 4).
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(
            potential=potential,
            initial_point=initial_point,
            final_point=final_point,
            **kwargs
        )

        self.degree = degree
        delta_geo = (self.final_point - self.initial_point)/float(n_anchors + 2) 
        self.points = jnp.array([
            self.initial_point + delta_geo*(i + 1) for i in range(n_anchors)
        ])
        delta_time = 1./(n_anchors + 1)
        self.knots = jnp.array([
            (i + 1)/float(n_anchors + 1) for i in range(n_anchors)
        ])
        print("This method is not finished")
        raise NotImplementedError
    
    def geometric_path(
            self,
            time: float,
            y: Any,
            *args: Any
    ) -> None:
        """
        Compute the geometric path at the given time.

        Parameters:
        -----------
        time : float
            The time at which to evaluate the geometric path.
        y : Any
            Placeholder for additional arguments.
        *args : Any
            Additional arguments.

        Returns:
        --------
        None
        """
        idx = self.degree + int(time/self.delta_time)
        time_diffs = time - self.knots[idx-(self.degree-1):idx+self.degree]
