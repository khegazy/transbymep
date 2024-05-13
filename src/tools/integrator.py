import time
import torch
import numpy as np
from torchdiffeq import odeint

from .metrics import Metrics


class ODEintegrator(Metrics):
    def __init__(
            self,
            potential,
            integrator='adaptive',
            solver='dopri5',
            rtol=1e-7,
            atol=1e-9,
            dx=0.01,
            is_multiprocess=False,
            do_load_balance=False,
            process=None
        ):
        self.potential = potential
        self.is_multiprocess = is_multiprocess
        self.do_load_balance = do_load_balance
        self.process = process
        
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.dx = dx

        if integrator == 'adaptive':
            self.path_integral = self.adaptive_path_integral
        elif integrator == 'parallel':
            self.path_integral = self.parallel_path_integral
        else:
            raise ValueError("integrator argument must be either 'adaptive', 'multiprocess', or 'parallel'.")
        
        if self.is_multiprocess:
            if self.process is None or not self.process.is_distributed:
                raise ValueError("Must run program in distributed mode with multiprocess integrator.")
            self.inner_path_integral = self.path_integral
            self.path_integral = self.multiprocess_path_integral
            self.run_times = np.ones(self.process.world_size)
            if self.do_load_balance:
                self.mp_times = np.linspace(0, 1, self.process.world_size+1)

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

    def multiprocess_path_integral(self, path, fxn_name, t_init=0., t_final=1., mp_times=None):
        
        if self.do_load_balance:
            frac_run_time = np.sum(self.run_times)/self.process.world_size
            for idx in range(len(self.mp_times)):
                n_shifts = len(self.mp_times) - idx

                rt_delta = self.run_times[idx] - frac_run_time
                pt_delta = self.mp_times[idx]*(rt_delta/self.run_times[idx])
                print("DELTAS", rt_delta, pt_delta)

                pt_shifts = -1*pt_delta*(1. - np.arange(n_shifts)/n_shifts)
                print(pt_shifts, np.arange(n_shifts)/n_shifts)
                self.mp_times[idx:-1] = self.mp_times[idx:-1] + pt_shifts
                print("NEW PATH TIMES", self.mp_times)
                
                rt_shifts = [-1] + [1./n_shifts,]*n_shifts
                rt_shifts = rt_delta*np.array(rt_shifts)
                self.run_times[idx:] += rt_shifts
                print("NEW RUN TIMES", self.run_times)
            print("MP TIMES", self.mp_times)
        if mp_times is None:
            mp_times = torch.linspace(t_init, t_final, self.process.world_size+1)
        
        start_time = time.time()
        integral = self.path_integral(
            path=path,
            fxn_name=fxn_name,
            t_init=mp_times[self.process.rank],
            t_final=mp_times[self.process.rank+1]
        )
        self.run_time[self.process.rank] = time.time() - start_time

        return integral





