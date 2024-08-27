import torch
from .base_class import PotentialBase


class Schwefel(PotentialBase):
    def __init__(self, dim=2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dim = dim

    def forward(self, points):
        out = 418.9829 * self.dim - torch.sum(points * torch.sin(torch.sqrt(torch.abs(points))), dim=-1)
        return  out