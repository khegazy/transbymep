import torch
import yaml
from ase import units
import os
import numpy as np
import sys
sys.path.append('/global/homes/e/ericyuan/GitHub')
from EScAIP.src import EfficientlyScaledAttentionInteratomicPotential

from .base_potential import BasePotential, PotentialOutput

class EScAIP(nn.Module):
    def __init__(self, config_file, checkpoint_file):
        super().__init__()
        with open(config_file) as f:
            config = yaml.safe_load(f)
        checkpoint = torch.load(checkpoint_file)
        self.module = EfficientlyScaledAttentionInteratomicPotential(**config['model'])
        self.load_state_dict(checkpoint['state_dict'])
        self.normalizers = checkpoint['normalizers']

        self.eval()
        self.to(torch.float64)
        self.to(self.device)
        self.requires_grad_(False)

    def forward(self, data):
        output = self.module(data)
        for key in output.keys():
            output[key] = output[key] * self.normalizers[key]['std'] + self.normalizers[key]['mean']
        return output

class EScAIPPotential(BasePotential):
    def __init__(self, config_file, checkpoint_file, **kwargs):
        """
        Constructor for EScAIPPotential

        Parameters
        ----------
        config_file: str
            path to the config file. eg. 'configs/s2ef/SPICE/EScAIP/L6_H16_256_.yml'
        checkpoint_file: str
            path to the checkpoint file. eg. 'SPICE_L6_H16_256_75Epochs_checkpoint.pt'
        """
        super().__init__(**kwargs)
        self.model = EScAIP(config_file, checkpoint_file)

    def forward(self, points):
        data = self.data_formatter(points)
        pred = self.model(data)
        energy = pred['energy']
        force = pred['forces']
        energy = energy.view(-1)
        force = force.view(*points.shape)
        return PotentialOutput(energy=energy, force=force)
    
    def data_formatter(self, pos):
        n_data = pos.numel() // (self.n_atoms * 3)
        data = Data(
            atomic_numbers=self.numbers.repeat(n_data, 1), 
            pos=pos.view(n_data, self.n_atoms, 3), 
            cell=self.cell.repeat(n_data, 1, 1),
            batch=torch.arange(n_data).repeat_interleave(self.n_atoms),
            natoms=torch.tensor([self.n_atoms for _ in range(n_data)]), 
            num_graphs=n_data, 
        )
        assert (self.pbc == self.model.module.use_pbc).all(), f'Path periodicity {self.pbc} does not match model periodicity {self.model.module.use_pbc}'
        return data