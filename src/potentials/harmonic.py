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
        dist = pos[..., None, :, :] - pos[..., :, None, :]
        ind = torch.triu_indices(dist.shape[-3], dist.shape[-2], offset=1)
        energy = torch.sum(torch.norm(dist[ind[0], ind[1]], dim=-1) ** 2) / 2
        return energy