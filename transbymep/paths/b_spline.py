from typing import Any
import torch

from .base_path import BasePath
from .linear import LinearPath


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

    def __init__(
        self,
        degree: int = 3,
        n_anchors: int = 32,
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
        super().__init__(**kwargs)

        self.degree = degree
        self.n_anchors = n_anchors
        control_points = torch.zeros(n_anchors, self.initial_point.shape[-1], device=self.device, dtype=torch.float64)  # shape: (n_anchors, n_output)
        self.control_points = torch.nn.Parameter(control_points)
        self.knots = torch.linspace(0, 1, n_anchors + degree + 1, device=self.device, dtype=torch.float64).unsqueeze(0)  # shape: (1, n_anchors + degree + 1)

        self.base = LinearPath(**kwargs)
    
    def get_geometry(self, time: float, *args):
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
        self.points = self.cox_deboor(torch.arange(self.n_anchors), self.degree, time, self.knots) @ self.control_points
        return self.base.get_geometry(time) + self.points
        
    def cox_deboor(
            self,
            i: torch.Tensor,
            j: int,
            t: torch.Tensor,
            knots: torch.Tensor
        ) -> torch.Tensor:
            """
            Cox-DeBoor recursion formula.

            Parameters:
            -----------
            i : int
                The index of the control point.
            j : int
                The degree of the B-Spline.
            t : float
                The time parameter.
            knots : np.array
                The knot vector.

            Returns:
            --------
            float
                The value of the B-Spline at the given time.
            """
            if j == 0:
                out = ((knots[:, i] <= t) & (t < knots[:, i + 1])).float()  # shape: (n_data, n_anchors)
            else:
                out = (t - knots[:, i]) / (knots[:, i + j] - knots[:, i]) * self.cox_deboor(i, j - 1, t, knots) + \
                    (knots[:, i + j + 1] - t) / (knots[:, i + j + 1] - knots[:, i + 1]) * self.cox_deboor(i + 1, j - 1, t, knots)
            # out = 0 if torch.isnan(out) else out
            # out = torch.where(torch.isnan(out), torch.tensor(0.0), out)
            return out