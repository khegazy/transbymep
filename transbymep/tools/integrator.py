import time
import torch
import torch.distributed as dist
import numpy as np
from dataclasses import dataclass
from enum import Enum

from torchdiffeq import odeint
from torchpathdiffeq import SerialAdaptiveStepsizeSolver, RKParallelAdaptiveStepsizeSolver
from .metrics import Metrics

from .metrics import Metrics

@dataclass
class IntegralOutput():
    integral: torch.Tensor
    times: torch.Tensor
    geometries: torch.Tensor

class ODEintegrator(Metrics):
    def __init__(
            self,
            computation='parallel',
            solver='dopri5',
            rtol=1e-7,
            atol=1e-9,
            remove_cut=0.1,
            process=None,
            is_multiprocess=False,
            is_load_balance=False,
            n_added_evals=3,
            device=None,
        ):
        self.is_multiprocess = is_multiprocess
        self.is_load_balance = is_load_balance
        self.process = process
        
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.previous_integral_state = None
        self.add_y_arg = False

        if computation == 'serial':
            self.is_parallel = False
            self._integrator = SerialAdaptiveStepsizeSolver(
                solver=self.solver,
                atol=atol,
                rtol=rtol,
                y0=torch.tensor([0], dtype=torch.float, device=device),
                t_init=0.,
                t_final=1.,
                device=device,
            )
            if self.is_load_balance:
                self.balance_load = self._serial_load_balance
        elif computation == 'parallel':
            self.is_parallel = True
            self.remove_cut = remove_cut
            self._integrator = RKParallelAdaptiveStepsizeSolver(
                solver=self.solver,
                atol=self.atol,
                rtol=self.rtol,
                remove_cut=self.remove_cut,
                y0=torch.tensor([0], dtype=torch.float, device=device),
                t_init=0.,
                t_final=1.,
                device=device,
            )
        else:
            raise ValueError(f"integrator argument must be either 'parallel' or 'serial', not {computation}.")
        
        if self.is_multiprocess:
            if self.process is None or not self.process.is_distributed:
                raise ValueError("Must run program in distributed mode with multiprocess integrator.")
            self.inner_path_integral = self.path_integral
            self.integrator = self.multiprocess_path_integral
            self.run_time = torch.tensor([1], requires_grad=False)# = np.ones(self.process.world_size)
            if self.is_load_balance:
                self.mp_times = torch.linspace(
                    0, 1, self.process.world_size+1, requires_grad=False
                )

        # if device is not None:
        #     self.device = torch.device(device)
        # else:
        #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def integrator(self, path, fxn_name, t_init=0., t_final=1., times=None):
        ode_fxn, _ = self._get_ode_eval_fxn(fxn_name=fxn_name, path=path)

        integral_output = self._integrator.integrate(
            ode_fxn=ode_fxn,
            state=self.previous_integral_state,
            t=times,
            t_init=t_init,
            t_final=t_final
        )
        self.previous_integral_state = integral_output
        return integral_output


    def path_integral(
            self, path, fxn_name, t_init=0., t_final=1., record_evals=False
        ):
        if record_evals:
            path.begin_time_recording()
        
        integral_output = self.integrator(
            path=path,
            fxn_name=fxn_name,
            t_init=t_init,
            t_final=t_final
        )

        if record_evals:
            time_record, geo_record = path.get_eval_record()
            path.end_eval_recording()
        else:
            time_record = None
            geo_record = None
        
        return integral_output
        return IntegralOutput(
            integral=integral,
            times=time_record,
            geometries=geo_record
        )

    def _get_ode_eval_fxn(self, fxn_name, path):
        if fxn_name not in dir(self):
            metric_fxns = [
                attr for attr in dir(Metrics)\
                    if attr[0] != '_' and callable(getattr(Metrics, attr))
            ]
            raise ValueError(f"Can only integrate metric functions, either add a new function to the Metrics class or use one of the following:\n\t{metric_fxns}")
        eval_fxn = getattr(self, fxn_name)

        if self.is_parallel:
            def ode_fxn(t, *args):
                return eval_fxn(path=path, t=t)
        else:
            def ode_fxn(t, *args):
                t = torch.tensor([[t]])
                #print("ODEF", t)
                output = eval_fxn(path=path, t=t)
                #print("ODEF out", output, output.requires_grad)
                return output[0]
        
        return ode_fxn, eval_fxn


    def _integrand_wrapper(self, t, y, path, ode_fxn):
        vals = path(t)
        return ode_fxn(vals)

    def serial_path_integral(self, path, fxn_name, t_init=0., t_final=1.):
        print("TODO: adaptive integrator evaluates t>1, how to set hard limits?")
        ode_fxn, _ = self._get_ode_eval_fxn(fxn_name=fxn_name, path=path)
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


    def _serial_load_balance(self, t_init=0., t_final=1.):
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
 
 
    def _parallel_path_integral(self, path, fxn_name, t_init=0., t_final=1., eval_times=None):
       
        print("INPUT EVAL TIMES", eval_times.shape)
        self._parallel_integral(
            ode_fxn=lambda x: torch.abs(path.geometric_path(x)),
            t_init=t_init,
            t_final=t_final,
            eval_times=self.geo_integral_times
        )
        #geos, times = self._parallel_integral_geometries(path, eval_times)
        geos, times = self._parallel_integral(path, eval_times)
        self.integral_times = times
        
        delta_cum = torch.cumsum(self._geo_deltas(geos), dim=0)
        delta_frac = (delta_cum/self.dx).to(torch.int)
        mask = torch.concatenate(
            [
                torch.tensor([True], dtype=torch.bool),
                delta_frac[:-1] != delta_frac[1:],
                torch.tensor([True], dtype=torch.bool)
            ],
            dim=0
        )
        eval_geos = geos[mask]
        eval_times = self.integral_times[mask]
        print("eval times shape", eval_times.shape)
        print(eval_times[:,0])
        delta_times = eval_times[1:] - eval_times[:-1]
        print("means", torch.mean(delta_times))

        _, eval_fxn = self._get_ode_eval_fxn(fxn_name=fxn_name, path=path)
        loss_evals = eval_fxn(path=path, t=eval_times)
        print("EVALS", loss_evals)
        integral = torch.sum(loss_evals[:-1]*delta_times)

        return integral

   
    def _parallel_integral_geometries(self, path, eval_times):
        times = None
        old_geos = torch.tensor([])
        old_times = torch.tensor([])
        idxs_old = torch.tensor([], dtype=torch.int)
        idxs_new = torch.arange(len(eval_times))
        while len(eval_times) > 0:
            if times is not None:
                print("TIMES BEFORE ADD", len(times), times[77:85,0])
            # Add points where points are too far
            geos, times = self._add_parallel_geometries(
                path, old_geos, old_times, eval_times, idxs_old, idxs_new
            )
            print("TIMES AFTER ADD", len(times), times[77:85,0])

            # Remove points that are too close
            geos, times = self._remove_parallel_geometries(geos, times)
            if len(times) == 2:
                time_mask = torch.arange(len(eval_times)) % 2 == 0
                time_mask[-1] = True
                geos, times = self._parallel_integral_geometries(
                    path, eval_times[time_mask]
                )
            print("TIMES AFTER REMOVE", len(times))#, times[:10])

            # Determine where difference between structures is too small
            deltas = self._geo_deltas(geos)
            add_mask = deltas > self.dxdx
            n_adds = torch.sum(add_mask)
            if n_adds > 0:
                add_idxs = torch.where(add_mask)[0]
                #db_idxs = torch.where(add_mask)[0]
                #print("ADD BETWEEN", db_idxs[:5], times[db_idxs[0]:db_idxs[0]+2])
                eval_deltas = (times[1:][add_mask] - times[:-1][add_mask]) # [n_adds]
                eval_deltas = eval_deltas/(self.n_added_evals + 1) # [n_adds]
                eval_deltas = torch.unsqueeze(eval_deltas, 1)\
                    *(1 + torch.unsqueeze(torch.arange(self.n_added_evals), 0)) #[n_adds, n_added_evals]
                eval_times = times[add_idxs]*torch.ones((n_adds, self.n_added_evals)) #[n_adds, n_added_evals]
                print("EVAL TIME SHAPE", eval_times.shape)
                eval_times = torch.unsqueeze(eval_times, 1) + eval_deltas 
                eval_times = torch.unsqueeze(eval_times.flatten(), 1)
                print("IDXS FROM MASK", n_adds)#, torch.where(add_mask))
                idxs_new = torch.unsqueeze(add_idxs, 1)\
                    + torch.unsqueeze(1 + torch.arange(self.n_added_evals), 0)\
                    + torch.unsqueeze(
                        self.n_added_evals*torch.arange(n_adds), 1
                    )
                idxs_new = idxs_new.flatten()
                idxs_old = torch.arange(len(times), dtype=torch.int)
                for idx in torch.where(add_mask)[0]:
                    idxs_old[idx+1:] = idxs_old[idx+1:] + self.n_added_evals
            else:
                eval_times = torch.tensor([])
            
            old_geos = geos
            old_times = times
            
            if len(eval_times) > 0:
                print("LARGE DELTAS", deltas[deltas > self.dxdx][:10])
                print("LARGE IDXS", torch.where(deltas[deltas > self.dxdx])[0][:10])
                print("LAST DELTAS", deltas[-10:])
                print("LEN TIMES", len(times))
            print("NEW IDXS", idxs_new[:10])
            print("CUR TIMES", times[:10])
            print("NEW TIMES", eval_times[:10])
            if len(eval_times) and eval_times[1,0] == 0:
                raise ValueError(f"Second time value is 0 {eval_times[:10,0]}")
            if len(eval_times) and torch.any(eval_times[:,0] < 0.0):
                neg_idxs = torch.where(eval_times[:,0] < 0.0)[0]
                raise ValueError(f"Eval Time < 0: {eval_times[neg_idxs,0]}")
            if len(eval_times) and torch.any(eval_times[:,0] > 1.0):
                large_idxs = torch.where(eval_times[:,0] > 1.0)[0]
                raise ValueError(f"Eval Time > 1.0: {eval_times[large_idxs,0]}")
            if len(eval_times) and torch.any(times[1:] - times[:-1] < 0):
                neg_idxs = torch.where(times[1:] - times[:-1] < 0.0)[0]
                raise ValueError(f"Incorrect Time Order: {eval_times[neg_idxs:neg_idxs+2,0]}")

            

        #self.integral_times = times
        
        return geos, times



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