import torch
from newtonnet.models import NewtonNet
from newtonnet.layers.activations import get_activation_by_string
from newtonnet.data import ExtensiveEnvironment
from newtonnet.data import batch_dataset_converter
import yaml
from ase import units

from .base_class import PotentialBase

class NewtonNetPotential(PotentialBase):
    def __init__(self, model_path, settings_path, numbers, device=None, **kwargs):
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
        # torch.set_default_tensor_type(torch.DoubleTensor)
        # if type(model_path) is list:
        #     self.models = [self.load_model(model_path_, settings_path_) for model_path_, settings_path_ in zip(model_path, settings_path)]
        # else:
        #     self.models = [self.load_model(model_path, settings_path)]
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path, settings_path)
        self.numbers = torch.tensor(numbers, dtype=torch.long, device=self.device)
        self.n_atoms = len(numbers)
        self.n_eval = 0

    
    def forward(self, points):
        data = self.data_formatter(points)
        pred = self.model(data)
        self.n_eval += 1
        return pred['E'].squeeze(dim=-1) * (units.kcal/units.mol)
        

    def load_model(self, model_path, settings_path):
        settings = yaml.safe_load(open(settings_path, 'r'))
        activation = get_activation_by_string(settings['model']['activation'])
        model = NewtonNet(
            resolution=settings['model']['resolution'],
            n_features=settings['model']['n_features'],
            activation=activation,
            n_interactions=settings['model']['n_interactions'],
            dropout=settings['training']['dropout'],
            max_z=10,
            cutoff=settings['data']['cutoff'],
            cutoff_network=settings['model']['cutoff_network'],
            normalize_atomic=settings['model']['normalize_atomic'],
            requires_dr=settings['model']['requires_dr'],
            device=self.device,
            create_graph=False,
            shared_interactions=settings['model']['shared_interactions'],
            # return_hessian=self.return_hessian,
            double_update_latent=settings['model']['double_update_latent'],
            layer_norm=settings['model']['layer_norm'],
            )

        model.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'])
        model = model
        model.eval()
        model.to(self.device)
        model.to(torch.float)
        for p in model.parameters():
            p.requires_grad = False
        return model
    
    def data_formatter(self, pos):
        n_data = pos.numel() // (self.n_atoms * 3)
        data  = {
            'R': pos.view(n_data, self.n_atoms, 3),
            'Z': self.numbers.repeat(n_data, 1),
            # 'E': torch.zeros((1, 1)),
            # 'F': torch.zeros((1, len(self.numbers), 3)),
        }
        N, NM, AM, _, _ = ExtensiveEnvironment().get_environment(data['R'], data['Z'])
        data.update({'N': torch.tensor(N), 'NM': torch.tensor(NM), 'AM': torch.tensor(AM)})
        # data = batch_dataset_converter(data, device=self.device)
        return data