import torch
from torch import nn

from .base_path import BasePath
from typing import Tuple, Optional
import numpy as np


class LinearPath(BasePath):
    """
    Linear path class for generating linear interpolation paths.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def geometric_path(self, time: float, *args):
        """
        Generates a geometric path using the MLP.

        Args:
            time (float): Time parameter for generating the path.
            *args: Additional arguments.

        Returns:
            torch.Tensor: The geometric path generated by the MLP.
        """
        return self.initial_point + time * self.vec