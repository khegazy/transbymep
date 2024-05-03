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
        # dist = torch.linalg.norm(pos[..., None, :, :] - pos[..., :, None, :], dim=-1)
        # energy = 4 * self.epsilon * torch.sum(((self.sigma / dist) ** 12 - (self.sigma / dist) ** 6).triu(diagonal=1), dim=(-1, -2))
        energy = 0.0
        for i in range(pos.shape[-2]):
            for j in range(i + 1, pos.shape[-2]):
                dist = torch.linalg.norm(pos[..., i, :] - pos[..., j, :], dim=-1)
                inv_dist = self.sigma / dist
                energy += 4 * self.epsilon * (inv_dist ** 12 - inv_dist ** 6)
        return energy