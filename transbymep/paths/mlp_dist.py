import torch
from torch import nn

from .base_path import BasePath
from .linear import LinearPath

class MLPDISTpath(BasePath):
    """
    Multilayer Perceptron (MLP) path class for generating geometric paths in inverse pairwise distances.

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

        self.ind = torch.triu_indices(self.n_atoms + 4, self.n_atoms + 4, 1)
        self.initial_point = self.cart_to_dist(self.initial_point[None, :]).squeeze(0)
        self.final_point = self.cart_to_dist(self.final_point[None, :]).squeeze(0)

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

    def get_geometry(self, time: float, *args):
        """
        Generates a geometric path using the MLP.

        Args:
            time (float): Time parameter for generating the path.
            *args: Additional arguments.

        Returns:
            torch.Tensor: The geometric path generated by the MLP.
        """
        mlp_out = self.mlp(time) - (1 - time) * self.mlp(self.t_init) - time * self.mlp(self.t_final)
        out = self.base.get_geometry(time) + mlp_out
        out = self.dist_to_cart(out)
        return out
    
    def cart_to_dist(self, cart):
        """
        Convert Cartesian coordinates to pairwise distances.

        Args:
            cart (torch.Tensor): Cartesian coordinates.

        Returns:
            torch.Tensor: Pairwise distances.
        """
        n_data = cart.shape[0]
        n_atoms = self.n_atoms

        # Reshape Cartesian coordinates
        cart = cart.view(n_data, n_atoms, 3)

        # Define reference axes of [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        axis = torch.zeros(n_data, 4, 3, device=self.device, dtype=cart.dtype)
        axis[:, 1, 0] = 1
        axis[:, 2, 1] = 1
        axis[:, 3, 2] = 1
        cart = torch.cat([axis, cart], dim=1)

        # Calculate pairwise distances
        dist = torch.norm(cart[:, self.ind[0], :] - cart[:, self.ind[1], :], dim=2)
        return dist
    
    def dist_to_cart(self, dist):
        """
        Convert pairwise distances to Cartesian coordinates.

        Args:
            dist (torch.Tensor): Pairwise distances.

        Returns:
            torch.Tensor: Cartesian coordinates.
        """
        n_data = dist.shape[0]
        n_atoms = self.n_atoms

        # Reshape pairwise distances
        dist_full = torch.zeros(n_data, n_atoms + 4, n_atoms + 4, device=self.device, dtype=dist.dtype)
        dist_full[:, self.ind[0], self.ind[1]] = dist
        dist_full[:, self.ind[1], self.ind[0]] = dist

        # Calculate Cartesian coordinates
        center = torch.eye(n_atoms + 4, device=self.device, dtype=dist.dtype)
        center = center - 1 / (n_atoms + 4)
        center = center[None, :, :]
        eigvals, eigvecs = torch.linalg.eigh(- 0.5 * center @ (dist_full ** 2) @ center)
        cart = eigvals[:, None, -3:].sqrt() * eigvecs[:, :, -3:]
        cart = cart.reshape(n_data, n_atoms + 4, 3)

        # Reconstruct reference axes
        cart = cart[:, 1:, :] - cart[:, :1, :]    # Subtract origin
        cart = cart[:, 3:, :] @ torch.linalg.solve(cart[:, :3, :], torch.eye(3, device=self.device, dtype=cart.dtype))    # Rotate axes

        return cart