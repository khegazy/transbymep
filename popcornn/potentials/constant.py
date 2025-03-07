from .base_potential import BasePotential

class Constant(BasePotential):
    def __init__(self, scale=1., **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def forward(self, point):
        return self.scale