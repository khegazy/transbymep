import torch
from torch import nn

from .base_path import BasePath


class MLPpath(BasePath):

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
        self.activation = nn.SELU()
        input_sizes = [1] + [n_embed]*(depth - 1)
        output_sizes = input_sizes[1:] + [self.final_point.shape[-1]]
        self.layers = [
            nn.Linear(input_sizes[i//2], output_sizes[i//2]) if i%2 == 0\
            else self.activation\
            for i in range(depth*2 - 1)
        ]
        self.mlp = nn.Sequential(*self.layers)
        self.Nevals = 0

    def geometric_path(self, time, *args):
        self.Nevals = self.Nevals + 1
        return self.mlp(time)\
            - (1 - time)*(self.mlp(torch.tensor([0.])) - self.initial_point)\
            - time*(self.mlp(torch.tensor([1.])) - self.final_point)

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