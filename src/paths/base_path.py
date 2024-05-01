import torch
from dataclasses import dataclass

from ..tools import metrics
from ..potentials.base_potential import BasePotential

@dataclass
class PathOutput():
    geometric_path: torch.Tensor
    potential_path: torch.Tensor
    velocity: torch.Tensor = None
    force: torch.Tensor = None
    times: torch.Tensor = None


class BasePath(torch.nn.Module):
    initial_point: torch.Tensor
    final_point: torch.Tensor
    potential: BasePotential

    def __init__(
        self,
        potential,
        initial_point,
        final_point,
        return_velocity=False,
        return_force=False,
        **kwargs
    ):
        super().__init__()
        self.potential = potential
        self.initial_point = torch.tensor(initial_point)
        self.final_point = torch.tensor(final_point)
        self.return_velocity = return_velocity
        self.return_force = return_force

    def geometric_path(self, time, y, *args):
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
    
    def get_path(self, times=None, return_velocity=False, return_force=False):
        if times is None:
            times = torch.unsqueeze(torch.linspace(0, 1., 1000), -1)
        elif len(times.shape) == 1:
            times = torch.unsqueeze(times, -1)
        
        return self.forward(
            times, return_velocity=return_velocity, return_force=return_force
        )
    
    def forward(self, t, return_velocity=False, return_force=False):
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
                create_graph=(not is_batched),
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
                fxn, t, create_graph=(not is_batched), vectorize=is_batched
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