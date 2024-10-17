from typing import Any
import torch
from torch import nn
from scipy.special import comb

from .base_path import BasePath


class Bezier(BasePath):
    """
    Bezier path representation.

    Attributes:
    -----------
    degree : int
        The degree of the Bezier.
    points : jnp.array
        The control points of the Bezier.
    """

    degree: int
    points: torch.Tensor

    def __init__(
        self,
        degree: int = 3,
        **kwargs: Any
    ) -> None:
        """
        Initialize the Bezier path.

        Parameters:
        -----------
        degree : int, optional
            The degree of the Bezier (default is 2).
        n_anchors : int, optional
            The number of anchor points (default is 4).
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(**kwargs)

        self.degree = degree
        t = torch.linspace(0, 1, self.degree + 1, device=self.device, dtype=torch.float64)
        t = t[1:-1]  # Exclude the endpoints
        control_points = torch.lerp(self.initial_point, self.final_point, t.unsqueeze(1))  # Linear interpolation
        self.control_points = nn.Parameter(control_points, requires_grad=True)

        self.comb = torch.tensor([comb(self.degree, i) for i in range(self.degree + 1)], device=self.device, dtype=torch.float64)

    def get_geometry(self, time, *args) -> None:
        """
        Compute the geometric path at the given time.

        Parameters:
        -----------
        time : float
            The time at which to compute the geometric path.
        y : Any
            The current state.
        *args : Any
            Additional arguments.
        """

        # result = torch.zeros((time.shape[0], self.initial_point.shape[0]), device=self.device)
        
        i = torch.arange(self.degree + 1, device=self.device)
        bernstein = self.comb * (1 - time)**(self.degree - i) * time**i
        
        points = torch.cat((self.initial_point.unsqueeze(0), self.control_points, self.final_point.unsqueeze(0)), dim=0)
        result = torch.matmul(bernstein, points)
        
        return result
    

