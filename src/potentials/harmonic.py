import torch
from .base_potential import BasePotential


class Harmonic(BasePotential):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, points):
        """
        args:
            points (torch.Tensor): shape (n_points, n_atoms * 3)
        """
        pos = points.unflatten(-1, (2, 3))
        # dist = torch.linalg.norm(pos[..., None, :, :] - pos[..., :, None, :], dim=-1)
        # inv_dist = 1 / (dist + 1e-6)
        # inv_dist = torch.where(dist > 0, inv_dist, torch.zeros_like(dist))
        # energy = torch.sum(dist ** 2, dim=(-1, -2)) / 2
        # energy = torch.sum(pos ** 2, dim=(-1, -2))
        energy = ((pos[..., 0, :] - pos[..., 1, :]) ** 2).sum(dim=-1)
        return energy