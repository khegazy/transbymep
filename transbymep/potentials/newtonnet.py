import torch
from newtonnet.models import NewtonNet
from newtonnet.layers.activations import get_activation_by_string
from newtonnet.data import ExtensiveEnvironment
from newtonnet.data import batch_dataset_converter
from newtonnet.utils.ase_interface import MLAseCalculator
import yaml
from ase import units
import os
import numpy as np

from .base_potential import BasePotential, PotentialOutput

class NewtonNetPotential(BasePotential):
    # def __init__(self, config_dir, model_path, **kwargs):
    def __init__(self, model_path, settings_path, **kwargs):
        """
        Constructor for NewtonNetPotential

        Parameters
        ----------
        model_path: str or list of str
            path to the model. eg. '5k/models/best_model_state.tar'
        settings_path: str or list of str
            path to the .yml setting path. eg. '5k/run_scripts/config_h2.yml'
        device: 
            device to run model. eg. 'cpu', ['cuda:0', 'cuda:1']
        kwargs
        """
        super().__init__(**kwargs)
        # self.model = self.load_model(os.path.join(config_dir, model_path))
        # self.model = self.load_model(model_path)
        self.model = self.load_model(model_path, settings_path)
        self.n_eval = 0

    
    def forward(self, points):
        data = self.data_formatter(points)
        pred = self.model(data)
        self.n_eval += 1
        energy = pred['E'] * (units.kcal/units.mol)
        force = pred['F'] * (units.kcal/units.mol/units.Ang)
        energy = energy.view(-1)
        force = force.view(*points.shape)
        return PotentialOutput(energy=energy, force=force)
        

    def load_model(self, model_path, settings_path):
        # model = torch.load(model_path, map_location=self.device)
        calc = MLAseCalculator(model_path, settings_path, device=self.device)
        model = calc.models[0]
        model.eval()
        model.to(torch.float64)
        # model.to(self.device)
        model.requires_grad_(False)
        return model
    
    def data_formatter(self, pos):
        n_data = pos.numel() // (self.n_atoms * 3)
        data  = {
            'R': pos.view(n_data, self.n_atoms, 3),
            'Z': self.numbers.repeat(n_data, 1).cpu().numpy(),
            # 'Z': np.stack([self.numbers for _ in range(n_data)]),
            # 'E': torch.zeros((1, 1)),
            # 'F': torch.zeros((1, len(self.numbers), 3)),
        }
        N, NM, AM, _, _ = ExtensiveEnvironment().get_environment(data['R'], data['Z'])
        data['Z'] = torch.tensor(data['Z'], device=self.device)
        data['N'] = torch.tensor(N, device=self.device)
        data['NM'] = torch.tensor(NM, device=self.device)
        data['AM'] = torch.tensor(AM, device=self.device)
        # data = batch_dataset_converter(data, device=self.device)
        # print(data)
        # raise ValueError
        return data