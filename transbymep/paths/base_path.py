import torch
from dataclasses import dataclass
from transbymep.tools import metrics
from transbymep.potentials.base_class import PotentialBase
from typing import Callable, Any
from ase import Atoms
from ase.io import read, write


@dataclass
class PathOutput():
    """
    Data class representing the output of a path computation.

    Attributes:
    -----------
    geometric_path : torch.Tensor
        The geometric path.
    potential_path : torch.Tensor
        The potential path.
    velocity : torch.Tensor, optional
        The velocity along the path (default is None).
    force : torch.Tensor, optional
        The force along the path (default is None).
    times : torch.Tensor, optional
        The times at which the path was evaluated (default is None).
    """
    geometric_path: torch.Tensor
    potential_path: torch.Tensor
    velocity: torch.Tensor = None
    force: torch.Tensor = None
    times: torch.Tensor = None


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
    potential: PotentialBase

    def __init__(
        self,
        potential: Callable,
        initial_point: torch.Tensor | Atoms | str,
        final_point: torch.Tensor | Atoms | str,
        return_velocity: bool = False,
        return_force: bool = False,
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
        return_velocity : bool, optional
            Whether to return velocity along the path (default is False).
        return_force : bool, optional
            Whether to return force along the path (default is False).
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__()
        print("DEVICE", device
              

              )
        self.potential = potential
        self.set_points(
            initial_point, final_point, device
        )
        self.return_velocity = return_velocity
        self.return_force = return_force
        self.device = device
        self.t_init = torch.tensor(
            [[0]], dtype=torch.float64, device=self.device
        )
        self.t_final = torch.tensor(
            [[1]], dtype=torch.float64, device=self.device
        )

    def set_points(
            self,
            initial_point: torch.Tensor | Atoms | str,
            final_point: torch.Tensor | Atoms | str,
            device: torch.device
    ) -> None:
        """
        Set the initial and final points of the path.

        Parameters:
        -----------
        initial_point : torch.Tensor, ase.Atoms, str
            The initial point of the path.
        final_point : torch.Tensor, ase.Atoms, str
            The final point of the path.
        device : torch.device
            The device on which to run the path.
        """
        assert type(initial_point) == type(final_point), "Initial and final points must be of the same type."
        if isinstance(initial_point, torch.Tensor):
            self.initial_point = initial_point
            self.final_point = final_point
            self.vec = self.final_point - self.initial_point
            self.wrap_fn = None
        elif isinstance(initial_point, Atoms) or isinstance(initial_point, str):
            if isinstance(initial_point, str):
                initial_point = read(initial_point)
                final_point = read(final_point)
            assert (initial_point.get_positions().shape[0] == final_point.get_positions().shape[0]), "Initial and final points must have the same number of atoms."
            assert (initial_point.get_positions().shape[1] == 3), "Initial and final points must have 3D positions."
            assert (initial_point.get_atomic_numbers() == final_point.get_atomic_numbers()).all(), "Initial and final points must have the same atomic numbers."
            assert (initial_point.get_pbc() == final_point.get_pbc()).all(), "Initial and final points must have the same periodic boundary conditions."
            assert (initial_point.get_cell() == final_point.get_cell()).all(), "Initial and final points must have the same cell."
            self.initial_point = torch.tensor(
                initial_point.get_positions(), dtype=torch.float64, device=device
            ).flatten()
            self.final_point = torch.tensor(
                final_point.get_positions(), dtype=torch.float64, device=device
            ).flatten()
            self.numbers = torch.tensor(
                initial_point.get_atomic_numbers(), dtype=torch.int64, device=device
            )
            self.pbc = torch.tensor(
                initial_point.get_pbc(), dtype=torch.bool, device=device
            )
            self.cell = torch.tensor(
                initial_point.get_cell(), dtype=torch.float64, device=device
            )
            self.n_atoms = len(initial_point)
            self.potential.numbers = self.numbers
            self.potential.pbc = self.pbc
            self.potential.cell = self.cell
            self.potential.n_atoms = self.n_atoms
            pair = initial_point + final_point
            self.vec = torch.tensor(
                [pair.get_distance(i, i + self.n_atoms, mic=True, vector=True) for i in range(self.n_atoms)], dtype=torch.float64, device=device
            ).flatten()
            self.wrap_fn = self.wrap_points if self.pbc.any() else None
        else:
            raise ValueError("Invalid type for initial_point and final_point.")

    def wrap_points(
            self, 
            points: torch.Tensor,
    ) -> torch.Tensor:
        """PyTorch implementation of ase.geometry.wrap_positions function."""

        fractional = torch.linalg.solve(self.cell.T, points.view(*points.shape[:-1], self.n_atoms, 3).transpose(-1, -2)).transpose(-1, -2)

        fractional[..., :, self.pbc] %= 1.0

        return torch.matmul(fractional, self.cell)


    def geometric_path(
            self,
            time: torch.Tensor,
            y: Any,
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
    
    def get_path(
            self,
            t: torch.Tensor = None,
            return_velocity: bool = False,
            return_force: bool = False
    ) -> PathOutput:
        """
        Get the path for the given times.

        Parameters:
        -----------
        t : torch.Tensor, optional
            The times at which to evaluate the path (default is None).
        return_velocity : bool, optional
            Whether to return velocity along the path (default is False).
        return_force : bool, optional
            Whether to return force along the path (default is False).

        Returns:
        --------
        PathOutput
            An instance of the PathOutput class representing the computed path.
        """
        if t is None:
            t = torch.linspace(0, 1, 1001)
        
        return self.forward(
            t, return_velocity=return_velocity, return_force=return_force
        )
    
    def forward(
            self,
            t,
            return_velocity: bool = False,
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
        if len(t.shape) == 1:
            t = torch.unsqueeze(t, -1)
        t = t.to(torch.float64).to(self.device)
        geo_path = self.geometric_path(t)
        if self.wrap_fn is not None:
            geo_path = self.wrap_fn(geo_path)
        # traj = [Atoms(
        #         numbers=self.numbers.detach().cpu().numpy(), 
        #         positions=pos.reshape(self.n_atoms, 3).detach().cpu().numpy(),
        #         pbc=self.pbc.detach().cpu().numpy(),
        #         cell=self.cell.detach().cpu().numpy()
        #     ) for pos in geo_path]
        # write("test.xyz", traj)
        # raise ValueError("STOP")
        pes_path = self.potential(geo_path)

        velocity, force = None, None
        is_batched = len(pes_path.shape) > 0
        if self.return_force or return_force:
            #print("SHAPES", pes_path.shape, len(pes_path.shape), torch.ones(0), geo_path.shape)
            #print("CHECK IS GRADS BATCHD FOR LEN > 0")
            force = torch.autograd.grad(
                torch.sum(pes_path),
                geo_path,
                create_graph=self.training,
            )[0]
            #print("LEN F", len(force), force[0].shape)
            if not is_batched:
                force = torch.unsqueeze(force, 0)
            #print("FORCES", force.shape)
        if self.return_velocity or return_velocity:
            #print("VEL SHAPES", geo_path.shape, t.shape)
            if is_batched:
                fxn = lambda t: torch.sum(self.geometric_path(t), axis=0)
            else:
                fxn = lambda t: self.geometric_path(t)
            velocity = torch.autograd.functional.jacobian(
                fxn, t, create_graph=self.training, vectorize=is_batched
            )
            #print("VEL INIT SHAPE", velocity.shape)
            #print("VEL TEST", velocity[:5])
            velocity = torch.transpose(velocity, 0, 1)
            if is_batched:
                velocity = velocity[:,:,0]
            #print("VEL F OUTPUT", velocity.shape, force.shape)
        
        return PathOutput(
            geometric_path=geo_path,
            potential_path=pes_path,
            velocity=velocity,
            force=force,
            times=t
        )