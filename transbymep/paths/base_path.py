import torch
from dataclasses import dataclass
from transbymep.tools import metrics
from transbymep.potentials.base_class import PotentialBase
from typing import Callable, Any


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
        initial_point: torch.Tensor,
        final_point: torch.Tensor,
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
        self.initial_point = torch.tensor(initial_point, device=device)
        self.final_point = torch.tensor(final_point, device=device)
        self.return_velocity = return_velocity
        self.return_force = return_force
        self.device = device

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
    
    """
    def get_path(self, times=None):
        raise NotImplementedError()
    
    def pes_path(self, t, y, *args):
        t = torch.tensor([t]).transpose()
        return self.potential.evaluate(self.geometric_path(t, y , *args))
    
    def pes_ode_term(self, t, y, in_integral=True, *args):
        t = torch.tensor([t]).transpose()
        return self.potential.evaluate(self.geometric_path(torch.tensor([t]), y , *args))
 
    def total_path(self, t, y, *args):
        t = torch.tensor([t]).transpose()
        geo_path = self.geometric_path(t, y , *args)
        return geo_path, self.potential.evaluate(geo_path)
    """
    
    def get_path(
            self,
            times: torch.Tensor = None,
            return_velocity: bool = False,
            return_force: bool = False
    ) -> PathOutput:
        """
        Get the path for the given times.

        Parameters:
        -----------
        times : torch.Tensor, optional
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
        if times is None:
            times = torch.unsqueeze(
                torch.linspace(0, 1., 1000),
                dim=-1
            )
        elif len(times.shape) == 1:
            times = torch.unsqueeze(times, -1)
        
        times = times.to(torch.float64).to(self.device)
        return self.forward(
            times, return_velocity=return_velocity, return_force=return_force
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
        t = t.to(torch.float64).to(self.device)
        geo_path = self.geometric_path(t)
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

    """
    def total_grad_path(self, t, y, *args):
        t = torch.tensor([t]).transpose()
        geo_path, geo_grad = jax.jvp(self.geometric_path, (t, y), (jnp.ones_like(t), 1.))
        pes_path, pes_grad = eqx.filter_value_and_grad(self.potential.evaluate)(geo_path)
        return geo_path, geo_grad, pes_path, pes_grad
    """

    """
    def eval_self(self, fxn_name, *input):
        path_output = self(*input, self.metric_args[fxn_name])
        return self.fxn_name(path_output, *input)
    """ 
    
    """
    # Loss functions
    def E_vre(self, t, y, *args):
        return metrics.E_vre(*self.total_grad_path(t, y, *args))
    
    def E_pvre(self, t, y, *args):
        return metrics.E_pvre(*self.total_grad_path(t, y, *args))
    
    def E_pvre_mag(self, t, y, *args):
        return metrics.E_pvre_mag(*self.total_grad_path(t, y, *args))
    
    def vre_residual(self, t, y, *args):
        return metrics.vre_residual(*self.total_grad_path(t, y, *args))
    """