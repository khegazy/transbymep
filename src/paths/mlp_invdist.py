import torch
from torch import nn

from .base_path import BasePath


class MLPinvdistpath(BasePath):

    def __init__(
        self,
        potential,
        initial_point,
        final_point,
        n_embed=32,
        depth=3,
        seed=123,
    ):
        super().__init__(
            potential=potential,
            initial_point=initial_point,
            final_point=final_point,
        )
        self.initial_point = self.initial_point.reshape(-1, 3)
        self.final_point = self.final_point.reshape(-1, 3)
        self.n_atoms = self.final_point.shape[0]
        self.initial_point = self.geo_to_invdist(self.initial_point)
        self.final_point = self.geo_to_invdist(self.final_point)
        self.activation = nn.SELU()
        input_sizes = [1] + [n_embed]*(depth - 1)
        output_sizes = input_sizes[1:] + [self.final_point.shape[-1]]
        self.layers = [
            nn.Linear(input_sizes[i//2], output_sizes[i//2]) if i%2 == 0\
            else self.activation\
            for i in range(depth*2 - 1)
        ]
        self.mlp = nn.Sequential(*self.layers)

    def geometric_path(self, time, *args):
        scale = 1
        invdist = scale * (self.mlp(time) - (1 - time) * self.mlp(torch.tensor([0.])) - time * self.mlp(torch.tensor([1.]))) \
            + ((1 - time) * self.initial_point + time * self.final_point)
        return self.invdist_to_geo(invdist)
        
    def geo_to_invdist(self, geo):
        dist = torch.linalg.norm(geo[..., :, None, :] - geo[..., None, :, :] + 1e-6, dim=-1)
        invdist = 1. / dist
        invdist[..., torch.eye(self.n_atoms, dtype=bool)] = 0
        return invdist.flatten()

    def invdist_to_geo(self, invdist):
        invdist = invdist.reshape(*invdist.shape[:-1], self.n_atoms, self.n_atoms)
        invdist = invdist + 1e-6
        dist = 1. / invdist
        dist[..., torch.eye(self.n_atoms, dtype=bool)] = 0
        center = torch.eye(self.n_atoms, device=dist.device) - 1. / self.n_atoms
        eigvals, eigvecs = torch.linalg.eigh(- 0.5 * center @ (dist ** 2) @ center)
        assert torch.all(eigvals[..., None, -3:] >= 0)
        geo = torch.sqrt(eigvals[..., None, -3:]) * eigvecs[..., :, -3:]
        return geo.flatten()

    """
    def get_path(self, times=None):
        if times is None:
            times = torch.unsqueeze(torch.linspace(0, 1., 1000), -1)
        elif len(times.shape) == 1:
            times = torch.unsqueeze(times, -1)

        geo_path = self.geometric_path(times)
        pes_path = self.potential(geo_path)
        
        return geo_path, pes_path
    """