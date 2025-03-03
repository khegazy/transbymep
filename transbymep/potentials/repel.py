import torch
from ase.data import covalent_radii

from .base_potential import BasePotential, PotentialOutput

class RepelPotential(BasePotential):
    def __init__(self, alpha=1.7, beta=0.01, r_max=3, skin=0.1, **kwargs):
        """
        Constructor for the Repulsive Potential from 
        Zhu, X., Thompson, K. C. & Martínez, T. J. 
        Geodesic interpolation for reaction pathways. 
        Journal of Chemical Physics 150, 164103 (2019).

        The potential is given by:
        E_ij = exp(-alpha * (r_ij - r0_ij) / r0_ij) + beta * r0_ij / r_ij
        E = sum_{i<j} E_ij * sigmoid((r_max - r_ij) / skin)

        Parameters
        ----------
        alpha: exponential term decay factor
        beta: inverse term weight
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.r_max = r_max
        self.skin = skin
        self.r0 = None
    
    def forward(self, points):
        if self.r0 is None:
            self.set_r0(self.numbers)
        points = points.view(-1, self.n_atoms, 3)
        r = torch.norm(points[:, self.ind[0]] - points[:, self.ind[1]], dim=-1)
        # energy = torch.sum(((1 - self.beta) * torch.exp(-self.alpha * (r - self.r0) / self.r0) + self.beta * self.r0 / r), dim=-1, keepdim=True)
        energy = torch.sum(
            (
                (torch.exp(-self.alpha * (r - self.r0) / self.r0) + self.beta * self.r0 / r)
                * torch.sigmoid((self.r_max - r) / self.skin)
            ),
            dim=-1, keepdim=True)
        return PotentialOutput(energy=energy)
    
    def set_r0(self, numbers):
        """
        Set the r0_ij values for the potential
        """
        radii = torch.tensor([covalent_radii[n] for n in numbers], device=self.device)
        r0 = radii.view(-1, 1) + radii.view(1, -1)
        self.ind = torch.triu_indices(r0.shape[0], r0.shape[1], offset=1, device=self.device)
        self.r0 = r0[None, self.ind[0], self.ind[1]]

