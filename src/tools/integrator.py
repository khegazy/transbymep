import time
import torch
import torch.distributed as dist
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
            process=None,
            is_multiprocess=False,
            is_load_balance=False,
        ):
        self.potential = potential
        self.is_multiprocess = is_multiprocess
        self.is_load_balance = is_load_balance
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
            self.run_time = torch.tensor([1], requires_grad=False)# = np.ones(self.process.world_size)
            if self.is_load_balance:
                self.mp_times = torch.linspace(
                    0, 1, self.process.world_size+1, requires_grad=False
                )

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
        
        #print(self.process.rank, self.run_times)
        torch.distributed.barrier()
        if self.is_load_balance:
            if self.process.is_master:
                run_times = [
                    torch.zeros(1, dtype=torch.float)\
                    for _ in range(self.process.world_size)
                ]
                dist.gather(
                    torch.tensor([self.run_time], dtype=torch.float32),
                    gather_list=run_times,
                    dst=0
                )
                run_times = torch.tensor(run_times).detach().numpy()
                #print("ALL RUN TIMES", run_times, self.mp_times)
                frac_run_time = np.sum(run_times)/self.process.world_size
                #print("INIT MP TIMES", self.mp_times)
                for idx in range(len(self.mp_times)-2):
                    n_shifts = len(self.mp_times) - idx - 2

                    rt_delta = run_times[idx] - frac_run_time
                    pt_delta = (self.mp_times[idx+1] - self.mp_times[idx])
                    pt_delta *= rt_delta/run_times[idx]
                    #print("DELTAS", rt_delta, pt_delta)

                    pt_shifts = -1*pt_delta*(1. - torch.arange(n_shifts)/n_shifts)
                    #print(self.mp_times[idx+1:-1], pt_shifts)
                    self.mp_times[idx+1:-1] = self.mp_times[idx+1:-1] + pt_shifts
                    #print("NEW PATH TIMES", self.mp_times)
                    
                    rt_shifts = [-1]
                    if n_shifts > 0:
                        rt_shifts = rt_shifts + [1./n_shifts,]*n_shifts
                    rt_shifts = rt_delta*np.array(rt_shifts)
                    run_times[idx:] += rt_shifts
                    #print("NEW RUN TIMES", self.run_times)
                #print("MP TIMES", self.mp_times)
            else:
                dist.gather(
                    torch.tensor([self.run_time], dtype=torch.float32), dst=0
                )
            dist.broadcast(self.mp_times, src=0)
            mp_times = self.mp_times.detach().numpy()
        if mp_times is None:
            mp_times = np.linspace(t_init, t_final, self.process.world_size+1)
        
        start_time = time.time()
        integral = self.inner_path_integral(
            path=path,
            fxn_name=fxn_name,
            t_init=mp_times[self.process.rank],
            t_final=mp_times[self.process.rank+1]
        )
        #self.run_times[self.process.rank] = time.time() - start_time
        self.run_time = time.time() - start_time

        return integral





