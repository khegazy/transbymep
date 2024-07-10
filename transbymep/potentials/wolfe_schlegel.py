import torch

from .base_potential import BasePotential

class WolfeSchlegel(BasePotential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "minima" in kwargs:
            self.minima = kwargs["minima"]
        else:
            self.minima = None
    
    def forward(self, points):
        points = self.point_transform(points)
        points = torch.movedim(points, -1, 0)
        x = points[0]
        y = points[1]

        return 10*(x**4 + y**4 - 2*x**2 - 4*y**2\
            + x*y + 0.2*x + 0.1*y)
