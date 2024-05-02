import torch
from .base_potential import BasePotential


class LennardJones(BasePotential):
    def __init__(self, epsilon=1.0, sigma=1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.sigma = sigma

    def forward(self, points):
        """
        args:
            points (torch.Tensor): shape (n_points, n_atoms * 3)
        """
        pos = points.unflatten(-1, (-1, 3))
        dist = torch.linalg.norm(pos[..., None, :, :] - pos[..., :, None, :], dim=-1)
        energy = 4 * torch.nansum(1 / dist ** 12 - 1 / dist ** 6, dim=(-1, -2)) / 2
        return  energy