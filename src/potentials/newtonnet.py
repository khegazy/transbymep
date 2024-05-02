import torch
from newtonnet.models import NewtonNet
from newtonnet.layers.activations import get_activation_by_string
import yaml

from .base_potential import BasePotential

class NewtonNetPotential(BasePotential):
    def __init__(self, model_path, settings_path, numbers, **kwargs):
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
        torch.set_default_tensor_type(torch.DoubleTensor)
        # if type(model_path) is list:
        #     self.models = [self.load_model(model_path_, settings_path_) for model_path_, settings_path_ in zip(model_path, settings_path)]
        # else:
        #     self.models = [self.load_model(model_path, settings_path)]
        self.model = self.load_model(model_path, settings_path)
        self.numbers = numbers
        raise NotImplementedError

    
    def forward(self, points):
        data = self.data_formatter(points)
        pred = self.model(data)
        return pred['E']
        

    def load_model(self, model_path, settings_path):
        settings = yaml.safe_load(open(settings_path, 'r'))
        activation = get_activation_by_string(settings['model']['activation'])
        model = NewtonNet(resolution=settings['model']['resolution'],
                            n_features=settings['model']['n_features'],
                            activation=activation,
                            n_interactions=settings['model']['n_interactions'],
                            dropout=settings['training']['dropout'],
                            max_z=10,
                            cutoff=settings['data']['cutoff'],
                            cutoff_network=settings['model']['cutoff_network'],
                            normalize_atomic=settings['model']['normalize_atomic'],
                            requires_dr=settings['model']['requires_dr'],
                            # device=self.device[0],
                            create_graph=False,
                            shared_interactions=settings['model']['shared_interactions'],
                            # return_hessian=self.return_hessian,
                            double_update_latent=settings['model']['double_update_latent'],
                            layer_norm=settings['model']['layer_norm'],
                            )

        model.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'])
        model = model
        # model.to(self.device[0])
        model.eval()
        return model
    
    def data_formatter(self, pos):
        """
        convert positions to input format of the model

        Parameters
        ----------
        pos: torch.Tensor

        Returns
        -------
        data: dict
            dictionary of arrays with following keys:
                - 'R':positions
                - 'Z':atomic_numbers
                - 'E':energy
                - 'F':forces
        """
        data  = {
            'R': pos,
            'Z': self.numbers,
            # 'E': torch.zeros((1, 1)),
            # 'F': torch.zeros((1, len(self.numbers), 3)),
        }
        return data