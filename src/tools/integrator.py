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
            if self.is_load_balance:
                self.balance_load = self._adaptive_load_balance
        elif integrator == 'parallel':
            self.dx_remove = self.dx/5
            self.integral_times = torch.linspace(0, 1, 1000)
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
        #print("NEVALS", self.process.rank, path.Nevals)
        return integral[-1]

    def _geo_deltas(self, geos):
        return torch.sqrt(torch.sum((geos[0:] - geos[:-1])**2, dim=-1))
 
 
    def _remove_parallel_points_timeList(self, geos, eval_times):
        deltas = self._geo_deltas(geos)
        remove_mask = deltas < self.dx_remove
        while torch.any(remove_mask):
            remove_mask = torch.concatenate(
                [
                    torch.tensor([False]), # Always keep t_init
                    remove_mask[:-2],
                    torch.tensor([remove_mask[-1] or remove_mask[-2]]),
                    torch.tensor([False]), # Always keep t_final
                ]
            )
            print("test not", remove_mask, ~remove_mask)
            eval_times = eval_times[~remove_mask]
            geos = geos[~remove_mask]
            deltas = self._geo_deltas(geos)
            remove_mask = deltas < self.dx_remove
        
        return geos, eval_times
 
    def _add_parallel_points_timeList(self, path, old_geos, old_times, eval_times, idxs_old, idxs_new):
        # Calculate new geometries
        new_geos = path.geometric_path(eval_times)
        
        # Place new geomtries between existing 
        geos = torch.zeros(
            (len(old_geos)+len(new_geos), new_geos.shape[-1]),
            requires_grad=False
        )
        geos[idxs_old] = old_geos
        geos[idxs_new] = new_geos

        # Place new times between existing 
        times = torch.zeros(
            (len(old_times)+len(eval_times)), requires_grad=False
        )
        times[idxs_old] = old_geos
        times[idxs_new] = new_geos

        return geos, times

   
    def _parallel_integral_points_timeList(self, path):
        old_geos = torch.tensor([])
        old_times = torch.tensor([])
        idxs_old = torch.tensor([], dtype=torch.int)
        eval_times = self.integral_times
        idxs_new = torch.arange(len(eval_times))
        while len(eval_times):
            # Add points where points are too far
            geos, times = self._add_parallel_points(
                path, old_geos, old_times, eval_times, idxs_old, idxs_new
            )

            # Remove points that are too close
            geos, times = self._remove_parallel_points(geos, times)

            # Determine where difference between structures is too small
            deltas = self._geo_deltas(geos)
            add_mask = deltas > self.dx
            print("Check that if a point is removed we don't add it back here")
            eval_times = (times[1:][add_mask] + times[:-1][add_mask])/2
            idxs_new = torch.where(add_mask)
            idxs_old = torch.arange(len(times)) + torch.concatenate(asdf)
        self.integral_times = times
        
        return geos




    def parallel_path_integral(self, path, fxn_name, t_init=0., t_final=1.):
        geos = path(self.integral_times)
        delta_cum = torch.cumsum(self._geo_deltas(geos))
        delta_frac = (delta_cum/self.dx).to(torch.int)
        mask = torch.concatenate(
            [
                torch.tensor([True], dtype=torch.bool),
                delta_frac[:-1] != delta_frac[0:],
                torch.tensor([True], dtype=torch.bool)
            ],
            requires_grad=False
        )
        eval_geos = geos[mask]
        eval_times = self.integral_times[mask]

    def _adaptive_load_balance(self, t_init=0., t_final=1.):
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
            run_times = torch.tensor(run_times)
            #print("----------------------------------------")
            #print("ALL RUN TIMES", run_times)
            total_run_time = torch.sum(run_times)
            run_fracs = run_times/total_run_time
            #frac_run_time = total_run_time/self.process.world_size
            rt_frac_deltas = run_fracs - 1./self.process.world_size
            #print("RUN FRACS", run_fracs, torch.sum(run_fracs))
            #print("RUN DFRACS", rt_frac_deltas)

            mp_deltas = self.mp_times[1:] - self.mp_times[:-1]
            mp_deltas = mp_deltas*(1. - rt_frac_deltas)
            self.mp_times[1:] = torch.cumsum(mp_deltas, dim=0)
            self.mp_times -= self.mp_times[0] - t_init
            self.mp_times *= t_final/self.mp_times[-1]
            #print("MP DIFFS", self.mp_times[1:] - self.mp_times[:-1])
            #print("MP TIMES", self.mp_times)
        else:
            dist.gather(
                torch.tensor([self.run_time], dtype=torch.float32), dst=0
            )
 
    def multiprocess_path_integral(self, path, fxn_name, t_init=0., t_final=1., mp_times=None):
        
        #print(self.process.rank, self.run_times)
        if mp_times is None:
            if self.is_load_balance:
                self.balance_load(t_init=t_init, t_final=t_final)
                dist.broadcast(self.mp_times, src=0)
                mp_times = self.mp_times.detach().numpy()
            else:
                mp_times = np.linspace(t_init, t_final, self.process.world_size+1)
        elif self.is_load_balance:
            raise ValueError("Cannot supply mp_times when is_load_balance is True")
        
        if np.any(mp_times > t_final):
            raise ValueError(f"Found an evaluation time larger than t_final {mp_times}")
        
        start_time = time.time()
        integral = self.inner_path_integral(
            path=path,
            fxn_name=fxn_name,
            t_init=mp_times[self.process.rank],
            t_final=mp_times[self.process.rank+1]
        )
        #self.run_times[self.process.rank] = time.time() - start_time
        self.run_time = time.time() - start_time
        """
        if self.process.is_distributed:
            print("NEVALS", self.process.rank, path.module.Nevals, self.run_time)
        else:
            print("NEVALS", self.process.rank, path.Nevals, self.run_time)
        """
        path.module.Nevals = 0
        path.Nevals = 0

        return integral





    
    
    def _multiprocess_path_integral(self, path, fxn_name, t_init=0., t_final=1., mp_times=None):
        
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
                run_times = torch.tensor(run_times)
                print("ALL RUN TIMES", run_times)
                total_run_time = torch.sum(run_times)
                run_fracs = run_times/total_run_time
                #frac_run_time = total_run_time/self.process.world_size
                rt_deltas = run_fracs - 1./self.process.world_size
                print("RUN FRACS", run_fracs)
                #print("RUN DELTA", run_deltas)
                mp_deltas = self.mp_times[1:] - self.mp_times[:-1]
                mp_targets = mp_deltas - rt_deltas 
                print("MP TARGETS", mp_targets, torch.sum(mp_targets))
                #print("FRAC SUBTRACT", frac_run_time, torch.sum(run_times))
                #print("RT Deltas", rt_deltas, torch.sum(rt_deltas))
                #print("INIT MP TIMES", self.mp_times)
                psum = 0
                for idx in range(self.process.world_size - 1):
                    print("LOOP", idx ,"------------------------")
                    shift = mp_targets[idx] - (self.mp_times[idx+1] - self.mp_times[idx])
                    self.mp_times[idx+1] = self.mp_times[idx] + mp_targets[idx]
                    print("AFTER SHIFT", shift, self.mp_times)
                if self.mp_times[-1] - self.mp_times[-2] != mp_targets[-1]:
                    print('MISSED LAST TARGET', self.mp_times[-1] - self.mp_times[-2], mp_targets[-1])
                print("MP DIFFS", self.mp_times[1:] - self.mp_times[:-1])
                print("MP TIMES", self.mp_times)
                """
                for idx in range(self.process.world_size - 1):
                    print("LOOP", idx ,"------------------------")
                    print("RT DELTAS", rt_deltas)
                    n_shifts = self.process.world_size - 1 - idx

                    pt_delta = (self.mp_times[idx+1] - self.mp_times[idx])
                    pt_delta *= rt_deltas[idx]/run_times[idx]
                    pt_delta = rt_deltas[idx]
                    psum += pt_delta
                    print("PT DELTAS", pt_delta)

                    pt_shifts = -1*pt_delta*(1 - torch.arange(n_shifts, dtype=float)/n_shifts)
                    print("MONEY", n_shifts, self.mp_times[idx+1:-1], pt_shifts)
                    self.mp_times[idx+1:-1] = self.mp_times[idx+1:-1] + pt_shifts
                    print("NEW PATH TIMES", self.mp_times)
                    
                    rt_deltas[idx] -= pt_delta
                    rt_deltas[idx+1:] += pt_delta/n_shifts
                    rt_shifts = [-1]
                    if n_shifts > 0:
                        rt_shifts = rt_shifts + [1./n_shifts,]*n_shifts
                    rt_shifts = rt_deltas[idx]*np.array(rt_shifts)
                    run_times[idx:] += rt_shifts
                    #print("NEW RUN TIMES", self.run_times)
                #print("MP TIMES", self.mp_times)
                print("FINAL PSUM", psum)
                """
            else:
                dist.gather(
                    torch.tensor([self.run_time], dtype=torch.float32), dst=0
                )
            dist.broadcast(self.mp_times, src=0)
            self.mp_times.detach().numpy()
            mp_times = self.mp_times
            if torch.any(mp_times > 1):
                adsf
        if mp_times is None:
            mp_times = np.linspace(t_init, t_final, self.process.world_size+1)
        
        if self.process.is_master:
            print(mp_times)
        start_time = time.time()
        integral = self.inner_path_integral(
            path=path,
            fxn_name=fxn_name,
            t_init=mp_times[self.process.rank],
            t_final=mp_times[self.process.rank+1]
        )
        #self.run_times[self.process.rank] = time.time() - start_time
        self.run_time = time.time() - start_time
        """
        if self.process.is_distributed:
            print("NEVALS", self.process.rank, path.module.Nevals, self.run_time)
        else:
            print("NEVALS", self.process.rank, path.Nevals, self.run_time)
        """
        path.module.Nevals = 0
        path.Nevals = 0

        return integral





