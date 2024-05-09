import torch
from torchdiffeq import odeint

from .metrics import Metrics


class ODEintegrator(Metrics):
    def __init__(self, potential, integrator='adaptive', solver='dopri5', rtol=1e-7, atol=1e-9, dx=0.01):
        self.potential = potential
        
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.dx = dx

        if integrator == 'adaptive':
            self.path_integral = self.adaptive_path_integral
        elif integrator == 'parallel':
            self.path_integral = self.parallel_path_integral
        else:
            raise ValueError("integrator argument must be either 'adaptive' or 'parallel'.")

    def _integrand_wrapper(self, t, y, path, ode_fxn):
        vals = path(t)
        return ode_fxn(vals)

    def adaptive_path_integral(self, path, fxn_name, t_init=0., t_final=1.):
        if fxn_name not in dir(self):
            metric_fxns = [
                attr for attr in dir(Metrics)\
                    if attr[0] != '_' and callable(getattr(Metrics, attr))
            ]
            raise ValueError(f"Can only integrate metric functions, either add a new function to the Metrics class or use one of the following:\n\t{metric_fxns}")
        eval_fxn = getattr(self, fxn_name)

        def ode_fxn(t, y, *args):
            return eval_fxn(path=path, t=torch.tensor([t]))

        integral = odeint(
            func=ode_fxn,
            y0=torch.tensor([0], dtype=torch.float),
            t=torch.tensor([t_init, t_final]),
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol
        )
        return integral[-1]

    def parallel_path_integral(self, path, fxn_name, t_init=0., t_final=1.):


