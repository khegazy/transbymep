import torch
from .base_potential import BasePotential


class ASE(BasePotential):
    def __init__(self, calculator, **kwargs) -> None:
        super().__init__(**kwargs)
        self.calculator = calculator

    def forward(self, points):
        raise NotImplementedError