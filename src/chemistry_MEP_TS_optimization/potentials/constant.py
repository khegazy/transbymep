from .base_class import PotentialBase

class Constant(PotentialBase):
    def __init__(self, scale=1., **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def forward(self, point):
        return self.scale