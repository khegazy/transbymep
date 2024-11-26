import torch
from torch import nn

from .base_path import BasePath
from .linear import LinearPath
activation_dict = {
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "gelu": nn.GELU(),
    "silu": nn.SiLU(),
    "selu": nn.SELU(),
}

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
        # n_embed: int = 32,
        n_embed: int = 1,
        depth: int = 3,
        activation: str = "selu",
        base: BasePath = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.activation = activation_dict[activation]
        # input_sizes = [1] + [n_embed]*(depth - 1)
        input_sizes = [1] + [self.final_point.shape[-1] * n_embed]*(depth - 1)
        output_sizes = input_sizes[1:] + [self.final_point.shape[-1]]
        self.layers = [
            nn.Linear(
                input_sizes[i//2], output_sizes[i//2], dtype=torch.float64, bias=True
            ) if i%2 == 0 else self.activation\
            for i in range(depth*2 - 1)
        ]
        # self.layers = [nn.Linear(1, self.final_point.shape[-1], dtype=torch.float64, bias=True)]
        # for i in range(depth):
        #     # self.layers.append(ResNetLayer(self.final_point.shape[-1]))
        #     self.layers.append(ResNetLayer(self.final_point.shape[-1], n_embed, activation))
        # self.input_layers = nn.Sequential(
        #     nn.Linear(1, n_embed, bias=True),
        #     self.activation,
        #     nn.Linear(n_embed, n_embed, bias=True),
        # )
        # self.layers = [
        #     nn.Sequential(
        #         nn.Linear(n_embed, n_embed, bias=True),
        #         self.activation,
        #         nn.Linear(n_embed, n_embed, bias=True),
        #     ) for i in range(depth)
        # ]
        # self.output_layers = nn.Sequential(
        #     nn.Linear(n_embed, n_embed, bias=True),
        #     self.activation,
        #     nn.Linear(n_embed, self.final_point.shape[-1], bias=True),
        #     # self.activation,
        #     # nn.Linear(self.final_point.shape[-1], self.final_point.shape[-1], bias=True),
        # )
        # self.layers = [self.input_layers] + self.layers + [self.output_layers]
        # self.layers = []
        # for i in range(depth - 1):
        #     self.layers.append(nn.Linear(input_sizes[i], output_sizes[i], bias=False))
        #     self.layers.append(self.activation)
        # self.layers.append(nn.Linear(input_sizes[-1], output_sizes[-1], bias=True))
        self.mlp = nn.Sequential(*self.layers)
        # self.mlp = nn.ModuleList(self.layers)
        self.mlp.to(self.device)
        # self.input_layers.to(self.device)
        # for layer in self.layers:
        #     layer.to(self.device)
        # self.output_layers.to(self.device)
        self.neval = 0

        self.base = base if base is not None else LinearPath(**kwargs)
        
        print("Number of trainable parameters in MLP:", sum(p.numel() for p in self.parameters() if p.requires_grad))
        print(self.mlp)

    def get_geometry(self, time: float, *args):
        """
        Generates a geometric path using the MLP.

        Args:
            time (float): Time parameter for generating the path.
            *args: Additional arguments.

        Returns:
            torch.Tensor: The geometric path generated by the MLP.
        """
        # n_data = len(time)
        # mlp_out = self.mlp(time) - (1 - time) * self.mlp(self.t_init) - time * self.mlp(self.t_final)
        # mlp_out = self.mlp(time) * torch.sin(time * torch.pi)
        mlp_out = self.mlp(time) * (1 - time) * time #* 4
        # mlp_out = self.mlp[0](time)
        # for layer in self.mlp[1:-1]:
        #     mlp_out = mlp_out + layer(mlp_out)
        # mlp_out = self.mlp[-1](mlp_out)
        # mlp_out = mlp_out * (1 - time) * time
        # mlp_out = (mlp_out.view(n_data, self.n_atoms, 3) * (self.tags != 0)[None, :, None]).view(n_data,  self.n_atoms * 3)
        # mlp_out = mlp_out.view(n_data, self.n_atoms, 3)
        # mlp_out = mlp_out * (self.tags != 0)[None, :, None]
        # mlp_out = mlp_out.view(n_data, self.n_atoms * 3)
        base_out = self.base.get_geometry(time) #* (1 - (1 - time) * time * 4)
        out = base_out + mlp_out
        # out = mlp_out
        return out

class ResNetLayer(nn.Module):
    def __init__(
        self,
        output_size: int,
        n_embed: int,
        activation: str = "selu",
    ):
        super().__init__()
        self.activation = activation_dict[activation]
        self.layer = nn.Sequential(
            nn.Linear(output_size, output_size * n_embed, dtype=torch.float64, bias=True),
            self.activation,
            nn.Linear(output_size * n_embed, output_size, dtype=torch.float64, bias=True),
        )
        # self.layer = SwiGLU(output_size, output_size)

    def forward(self, x):
        return x + self.layer(x)
    
class SwiGLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(in_features, out_features)
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.fc1(x) * self.activation(self.fc2(x))
