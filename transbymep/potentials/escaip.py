import torch
from torch import nn
from torch_geometric.data import Data
import yaml
from ase import units
import os
import numpy as np
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from torch_geometric.data import Data

import sys
# sys.path.append('/global/homes/e/ericyuan/GitHub')
sys.path.append('/global/homes/e/ericyuan/GitHub/EScAIP')
# from EScAIP.src import EfficientlyScaledAttentionInteratomicPotential

from .base_potential import BasePotential, PotentialOutput

class EScAIPPotential(BasePotential):
    def __init__(self, config_yml, checkpoint_path, cpu, **kwargs):
        super().__init__(**kwargs)
        calc = OCPCalculator(config_yml=config_yml, checkpoint_path=checkpoint_path, cpu=cpu)
        self.trainer = calc.trainer
        self.trainer.model.eval()    
        if self.trainer.ema is not None:
            self.trainer.ema.store()
            self.trainer.ema.copy_to()
        self.trainer.model.requires_grad_(False)

    def forward(self, points):
        data = self.data_formatter(points)
        pred = self.trainer.model(data)
        for key in pred.keys():
            pred[key] = self.trainer._denorm_preds(key, pred[key], data)
        energy = pred['energy'].view(-1)
        force = pred['forces'].view(*points.shape)
        # pred = []
        # for d in data:
        #     pred.append(self.trainer.model(d))
        # energy = torch.concat([p['energy'] for p in pred]).view(-1)
        # force = torch.concat([p['forces'] for p in pred]).view(*points.shape)
        return PotentialOutput(energy=energy, force=force)

    def data_formatter(self, pos):
        pos: torch.Tensor = pos.float()
        numbers: torch.Tensor = self.numbers
        cell: torch.Tensor = self.cell.float()
        pbc: torch.Tensor = self.pbc
        tags: torch.Tensor = self.tags
        n_atoms: int = self.n_atoms
        n_data: int = pos.shape[0]
        
        data = Data(
            atomic_numbers=numbers.repeat(n_data), 
            pos=pos.view(n_data * n_atoms, 3), 
            cell=cell.repeat(n_data, 1, 1),
            batch=torch.arange(n_data, device=self.device).repeat_interleave(n_atoms),
            natoms=torch.tensor(n_atoms, device=self.device).repeat(n_data), 
            num_graphs=n_data, 
            tags=tags.repeat(n_data),
        )
        # data = []
        # for i in range(n_data):
        #     d = Data(
        #         atomic_numbers=numbers, 
        #         pos=pos[i].view(n_atoms, 3),
        #         cell=cell.unsqueeze(0),
        #         batch=torch.tensor([0] * n_atoms, device=self.device),
        #         natoms=torch.tensor([n_atoms], device=self.device), 
        #         num_graphs=1, 
        #         # tags=tags.repeat(1),
        #     )
        #     data.append(d)

        return data


# class EScAIP(nn.Module):
#     def __init__(self, config_file, checkpoint_file, device='cuda'):
#         super().__init__()
#         with open(config_file) as f:
#             config = yaml.safe_load(f)
#         checkpoint = torch.load(checkpoint_file, map_location=device)
#         self.module = EfficientlyScaledAttentionInteratomicPotential(**config['model']).to(device)
#         self.load_state_dict(checkpoint['state_dict'])
#         self.normalizers = checkpoint['normalizers']
#         self.eval()

#     def forward(self, data):
#         output = self.module(data)
#         for key in output.keys():
#             output[key] = output[key] * self.normalizers[key]['std'] + self.normalizers[key]['mean']
#         return output

# class EScAIPPotential(BasePotential):
#     def __init__(self, config_file, checkpoint_file, **kwargs):
#         """
#         Constructor for EScAIPPotential

#         Parameters
#         ----------
#         config_file: str
#             path to the config file. eg. 'configs/s2ef/SPICE/EScAIP/L6_H16_256_.yml'
#         checkpoint_file: str
#             path to the checkpoint file. eg. 'SPICE_L6_H16_256_75Epochs_checkpoint.pt'
#         """
#         super().__init__(**kwargs)
#         self.model = EScAIP(config_file, checkpoint_file, self.device)
#         # self.model.to(torch.float64)
#         # self.model.to(self.device)
#         self.model.requires_grad_(False)

#     def forward(self, points):
#         data = self.data_formatter(points)
#         pred = self.model(data)
#         energy = pred['energy']
#         force = pred['forces']
#         energy = energy.view(-1)
#         force = force.view(*points.shape)
#         return PotentialOutput(energy=energy, force=force)
    
#     def data_formatter(self, pos):
#         pos: torch.Tensor = pos.float()
#         numbers: torch.Tensor = self.numbers
#         cell: torch.Tensor = self.cell
#         pbc: torch.Tensor = self.pbc
#         n_atoms: int = self.n_atoms
#         n_data: int = pos.shape[0]

#         data = Data(
#             atomic_numbers=numbers.repeat(n_data), 
#             pos=pos.view(n_data * n_atoms, 3), 
#             cell=cell.repeat(n_data, 1, 1),
#             batch=torch.arange(n_data, device=self.device).repeat_interleave(n_atoms),
#             natoms=torch.tensor(n_atoms, device=self.device).repeat(n_data), 
#             num_graphs=n_data, 
#         )
#         assert (pbc == self.model.module.use_pbc).all(), f'Path periodicity {pbc} does not match model periodicity {self.model.module.use_pbc}'
#         return data