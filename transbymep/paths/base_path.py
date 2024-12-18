import torch
from dataclasses import dataclass
from transbymep.tools import read_atoms, pair_displacement
from transbymep.potentials.base_potential import BasePotential
from typing import Callable, Any
from ase import Atoms
from ase.io import read
import numpy as np


@dataclass
class PathOutput():
    """
    Data class representing the output of a path computation.

    Attributes:
    -----------
    path_geometry : torch.Tensor
        The coordinates along the path.
    path_velocity : torch.Tensor, optional
        The velocity along the path (default is None).
    path_energy : torch.Tensor
        The potential energy along the path.
    path_force : torch.Tensor, optional
        The force along the path (default is None).
    times : torch.Tensor
        The times at which the path was evaluated.
    """
    times: torch.Tensor
    path_geometry: torch.Tensor
    path_energy: torch.Tensor
    path_velocity: torch.Tensor = None
    path_force: torch.Tensor = None


class BasePath(torch.nn.Module):
    """
    Base class for path representation.

    Attributes:
    -----------
    initial_point : torch.Tensor
        The initial point of the path.
    final_point : torch.Tensor
        The final point of the path.
    potential : PotentialBase
        The potential function.

    Methods:
    --------
    geometric_path(time, y, *args) -> torch.Tensor:
        Compute the geometric path at the given time.

    get_path(times=None, return_velocity=False, return_force=False) -> PathOutput:
        Get the path for the given times.

    forward(t, return_velocity=False, return_force=False) -> PathOutput:
        Compute the path output for the given times.
    """
    initial_point: torch.Tensor
    final_point: torch.Tensor
    potential: BasePotential

    def __init__(
        self,
        potential: Callable,
        initial_point: torch.Tensor | Atoms | str,
        final_point: torch.Tensor | Atoms | str,
        device: torch.device = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the BasePath.

        Parameters:
        -----------
        potential : callable
            The potential function.
        initial_point : torch.Tensor
            The initial point of the path.
        final_point : torch.Tensor
            The final point of the path.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__()
        print("DEVICE", device)
        self.potential = potential
        self.set_points(
            initial_point, final_point, device
        )
        self.device = device
        self.t_init = torch.tensor(
            [[0]], dtype=torch.float64, device=self.device
        )
        self.t_final = torch.tensor(
            [[1]], dtype=torch.float64, device=self.device
        )
        self.neval = 0

    def set_points(
            self,
            initial_point: torch.Tensor | list | np.ndarray | Atoms | str,
            final_point: torch.Tensor | list | np.ndarray | Atoms | str,
            device: torch.device
    ) -> None:
        """
        Set the initial and final points of the path.

        Parameters:
        -----------
        initial_point : torch.Tensor, list, np.ndarray, ase.Atoms, str
            The initial point of the path.
        final_point : torch.Tensor, list, np.ndarray, ase.Atoms, str
            The final point of the path.
        device : torch.device
            The device on which to run the path.
        """
        assert type(initial_point) == type(final_point), "Initial and final points must be of the same type."
        if isinstance(initial_point, torch.Tensor) or isinstance(initial_point, list) or isinstance(initial_point, np.ndarray):
            if isinstance(initial_point, list) or isinstance(initial_point, np.ndarray):
                initial_point = torch.tensor(initial_point, dtype=torch.float64, device=device)
                final_point = torch.tensor(final_point, dtype=torch.float64, device=device)
            self.initial_point = initial_point
            self.final_point = final_point
            self.vec = self.final_point - self.initial_point
            self.transform = None
        elif isinstance(initial_point, Atoms) or isinstance(initial_point, str):
            initial_atoms = read(initial_point) if isinstance(initial_point, str) else initial_point
            final_atoms = read(final_point) if isinstance(final_point, str) else final_point
            assert (initial_atoms.get_positions().shape[0] == final_atoms.get_positions().shape[0]), "Initial and final points must have the same number of atoms."
            assert (initial_atoms.get_positions().shape[1] == 3), "Initial and final points must have 3D positions."
            assert (initial_atoms.get_atomic_numbers() == final_atoms.get_atomic_numbers()).all(), "Initial and final points must have the same atomic numbers."
            assert (initial_atoms.get_pbc() == final_atoms.get_pbc()).all(), "Initial and final points must have the same periodic boundary conditions."
            assert (initial_atoms.get_cell() == final_atoms.get_cell()).all(), "Initial and final points must have the same cell."
            assert (initial_atoms.get_tags() == final_atoms.get_tags()).all(), "Initial and final points must have the same tags."
            initial_dict = read_atoms(initial_atoms)
            final_dict = read_atoms(final_atoms)
            self.initial_point = initial_dict["positions"].flatten().to(device)
            self.final_point = final_dict["positions"].flatten().to(device)
            self.numbers = initial_dict["numbers"].to(device)
            self.pbc = initial_dict["pbc"].to(device)
            self.cell = initial_dict["cell"].to(device)
            self.tags = initial_dict["tags"].to(device)
            self.n_atoms = initial_dict["n_atoms"]
            self.potential.numbers = self.numbers
            self.potential.pbc = self.pbc
            self.potential.cell = self.cell
            self.potential.n_atoms = self.n_atoms
            self.vec = pair_displacement(initial_atoms, final_atoms).flatten().to(device)
            self.potential.tags = self.tags
            self.transform = self.wrap_points if self.pbc.any() else None
        else:
            raise ValueError("Invalid type for initial_point and final_point.")

    def wrap_points(
            self, 
            points: torch.Tensor,
            center: float = 0
    ) -> torch.Tensor:
        """PyTorch implementation of ase.geometry.wrap_positions function."""

        fractional = torch.linalg.solve(self.cell.T, points.view(*points.shape[:-1], self.n_atoms, 3).transpose(-1, -2)).transpose(-1, -2)

        # fractional[..., :, self.pbc] %= 1.0
        fractional = (fractional + center) % 1.0 - center    # TODO: Modify this to handle partially true PBCs

        return torch.matmul(fractional, self.cell).view(*points.shape)


    def get_geometry(
            self,
            time: torch.Tensor,
            *args: Any
    ) -> torch.Tensor:
        """
        Compute the geometric path at the given time.

        Parameters:
        -----------
        time : torch.Tensor
            The time at which to evaluate the geometric path.
        y : Any
            Placeholder for additional arguments.
        *args : Any
            Additional arguments.

        Returns:
        --------
        torch.Tensor
            The geometric path at the given time.
        """
        raise NotImplementedError()
    
    def forward(
            self,
            t : torch.Tensor = None,
            return_velocity: bool = False,
            return_energy: bool = False,
            return_force: bool = False
    ) -> PathOutput:
        """
        Forward pass to compute the path, potential, velocity, and force.

        Parameters:
        -----------
        t : torch.Tensor
            The time tensor at which to evaluate the path.
        return_velocity : bool, optional
            Whether to return velocity along the path (default is False).
        return_force : bool, optional
            Whether to return force along the path (default is False).

        Returns:
        --------
        PathOutput
            An instance of the PathOutput class containing the computed path, potential, velocity, force, and times.
        """
        if t is None:
            t = torch.linspace(0, 1, 1001)
        while len(t.shape) < 2:
            t = torch.unsqueeze(t, -1)
        t = t.to(torch.float64).to(self.device)

        self.neval += t.numel()
        # print(time)
        # if self.neval > 1e5:
        #     raise ValueError("Too many evaluations!")

        path_geometry = self.get_geometry(t)
        if self.transform is not None:
            path_geometry = self.transform(path_geometry)
        if return_energy or return_force:
            potential_output = self.potential(path_geometry)

        if return_energy:
            path_energy = potential_output.energy
        else:
            path_energy = None

        if return_force:
            if potential_output.force is not None:
                path_force = potential_output.force
            else:
                
                path_force = -torch.autograd.grad(
                    potential_output.energy,
                    path_geometry,
                    grad_outputs=torch.ones_like(potential_output.energy),
                    create_graph=self.training,
                )[0]
                #print("SHAPES", pes_path.shape, len(pes_path.shape), torch.ones(0), geo_path.shape)
                #print("CHECK IS GRADS BATCHD FOR LEN > 0")
                # force = torch.autograd.grad(
                #     torch.sum(pes_path),
                #     geo_path,
                #     create_graph=self.training,
                # )[0]
                #print("LEN F", len(force), force[0].shape)
                # if not is_batched:
                #     force = torch.unsqueeze(force, 0)
                #print("FORCES", force.shape)
        else:
            path_force = None
        if return_velocity:
            #print("VEL SHAPES", geo_path.shape, t.shape)
            # if is_batched:
            #     fxn = lambda t: torch.sum(self.geometric_path(t), axis=0)
            # else:
            #     fxn = lambda t: self.geometric_path(t)
            # velocity = torch.autograd.functional.jacobian(
            #     fxn, t, create_graph=self.training, vectorize=is_batched
            # )
            path_velocity = torch.autograd.functional.jacobian(
                lambda t: torch.sum(self.get_geometry(t), axis=0), t, create_graph=self.training, vectorize=True
            ).transpose(0, 1)[:, :, 0]
        else:
            path_velocity = None

        if return_energy or return_force:
            del potential_output
        
        return PathOutput(
            times=t,
            path_geometry=path_geometry,
            path_energy=path_energy,
            path_velocity=path_velocity,
            path_force=path_force,
        )
