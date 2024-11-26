import torch
from .base_potential import BasePotential, PotentialOutput


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
        dist = pos[..., None, :, :] - pos[..., :, None, :]
        ind = torch.triu_indices(dist.shape[-3], dist.shape[-2], offset=1)
        ivn_dist = self.sigma / torch.linalg.norm(dist[..., ind[0], ind[1], :], dim=-1)
        energy = 4 * self.epsilon * torch.sum((ivn_dist ** 12 - ivn_dist ** 6), dim=-1)
        return PotentialOutput(energy=energy, force=None)