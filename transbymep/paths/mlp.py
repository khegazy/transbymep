import torch
from torch import nn

from .base_path import BasePath
from .linear import LinearPath
from typing import Tuple, Optional
import numpy as np

from ase import Atoms
from ase.io import write

class MLPpath(BasePath):
    """
    Multilayer Perceptron (MLP) path class for generating geometric paths.

    Args:
        n_embed (int, optional): Number of embedding dimensions. Defaults to 32.
        depth (int, optional): Depth of the MLP. Defaults to 3.
        base: Path class to correct. Defaults to LinearPath.
    """
    def __init__(
        self,
        n_embed: int = 32,
        depth: int = 3,
        base: BasePath = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.activation = nn.SELU()
        input_sizes = [1] + [n_embed]*(depth - 1)
        output_sizes = input_sizes[1:] + [self.final_point.shape[-1]]
        self.layers = [
            nn.Linear(
                input_sizes[i//2], output_sizes[i//2], dtype=torch.float64
            ) if i%2 == 0 else self.activation\
            for i in range(depth*2 - 1)
        ]
        self.mlp = nn.Sequential(*self.layers)
        self.mlp.to(self.device)
        self.neval = 0

        self.base = base if base is not None else LinearPath(**kwargs)
        # self.serialnumber = 0

    def get_geometry(self, time: float, *args):
        """
        Generates a geometric path using the MLP.

        Args:
            time (float): Time parameter for generating the path.
            *args: Additional arguments.

        Returns:
            torch.Tensor: The geometric path generated by the MLP.
        """
        n_data = len(time)
        # print("n_data", n_data)
        # print("time", time.flatten())
        # print("time diff", time.flatten()[1:] - time.flatten()[:-1])
        # print("serial number", self.serialnumber)
        # self.serialnumber += 1
        mlp_out = self.mlp(time) - (1 - time) * self.mlp(self.t_init) - time * self.mlp(self.t_final)
        # mlp_out = (mlp_out.view(n_data, self.n_atoms, 3) * (self.tags != 0)[None, :, None]).view(n_data,  self.n_atoms * 3)
        mlp_out = mlp_out.view(n_data, self.n_atoms, 3)
        mlp_out = mlp_out * (self.tags != 0)[None, :, None]
        mlp_out = mlp_out.view(n_data, self.n_atoms * 3)
        # return self.base.get_geometry(time) + \
        #     self.mlp(time) - (1 - time) * self.mlp(self.t_init) - time * self.mlp(self.t_final)
        out = self.base.get_geometry(time) + mlp_out
        # self.save_atoms(out)
        return out

    # def save_atoms(self, pos):
    #     traj = []
    #     pos = self.transform(pos)
    #     for i in range(pos.shape[0]):
    #         atoms = Atoms(
    #             numbers=self.numbers.detach().cpu().numpy(),
    #             positions=pos[i].view(-1, 3).detach().cpu().numpy(),
    #             cell=self.cell.detach().cpu().numpy(),
    #             pbc=self.pbc.detach().cpu().numpy(),
    #             tags=self.tags.detach().cpu().numpy(),
    #         )
    #         traj.append(atoms)
    #     write(f"test{self.serialnumber}.xyz", traj)

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