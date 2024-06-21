import time
import torch
import torch.distributed as dist
import numpy as np
from torchdiffeq import odeint
from dataclasses import dataclass
from enum import Enum

class degree(Enum):
    P = 0
    P1 = 1

@dataclass
class IntegralOutput():
    integral: torch.Tensor
    times: torch.Tensor
    geometries: torch.Tensor



class ODEPathIntegrator():
    def __init__(self, p, ode_fxn=None, t_init=0, t_final=1):
          self.p = p
          self.ode_fxn = ode_fxn
          self.t_init = t_init
          self.t_final = t_final

    def _parallel_integral(self, ode_fxn, t_init=0., t_final=1., t=None, ode_args=None):
        t_add = t
        if t_add is None:
            if self.previous_t is None: #TODO check name of last function too, don't want to use times from other function
                t_add = torch.unsqueeze(
                    torch.linspace(t_init, t_final, 1000), 1
                )
            else:
                mask = (self.previous_t[:,0] <= t_final)\
                    + (self.previous_t[:,0] >= t_init)
                t_add = self.previous_t[mask]

        t = torch.tensor([])
        y_previous = torch.tensor([])
        idxs_previous = torch.tensor([], dtype=torch.int)
        idxs_add = torch.arange(len(t_add))
        while len(t_add) > 0:
        
            # Evaluate new points and add new evals and points to arrays
            y, t = self._add_evals(
                ode_fxn, y_previous, t, t_add, idxs_previous, idxs_add
            )

            # Calculate error
            self._calculate_error(t, y)
            
            # Remove points that are too close
            geos, times = self._remove_evals(evals, points)
            if len(times) == 2:
                time_mask = torch.arange(len(eval_times)) % 2 == 0
                time_mask[-1] = True
                geos, times = self._parallel_integral_geometries(
                    path, eval_times[time_mask]
                )


    def _calculate_error(self, t, y, y0=0):
        integral_p, y_p = self._RK_steps(t, y, y0=y0, degr=degree.P)
        integral_p1, y_p1 = self._RK_steps(t, y, y0=y0, degr=degree.P1)
        error = y_p1 - y_p
        print(integral_p, integral_p1, error.shape, torch.mean(error), torch.amax(error))
        adsf
        return error, y_p, y_p1

    def _RK_integral(self, t, y, y0=0, degr=degree.P1):
        dt = t[1:] - t[:-1]
        tableau_c = self.calculate_tableau_c(dt, degr=degr)
        assert(tableau_c.shape[-1] == max(2, self.p))
        assert(torch.all(torch.sum(tableau_c, dim=-1) == 1))
        
        _t_p = h[0::self.p]
        h = _t_p[1:] - _t_p[:-1]
        
        y_steps = torch.reshape(y[1:], max(self.p, 2))
        y_steps = torch.concatenate(
            [
                torch.concatenate(
                    [torch.tensor([y[0]]), y_steps[:-1,-1]]
                ).unsqueeze(0),
                y_steps
            ],
            dim=0
        )

        RK_steps = h*torch.sum(tableau_c*y_steps, dim=-1)      # Sum over k evaluations weighted by c
        integral = y0 + torch.sum(RK_steps)                    # Sum over all steps with step size h
        return integral, RK_steps


    def _calculate_tableau_c(self, dt, degr):
        if degr == degree.P:
            tableau_c_fxn = self._tableau_c_p
        elif degr == degree.P1:
            tableau_c_fxn = self._tableau_c_p1
        
        return tableau_c_fxn(dt)


    def _tableau_c_p(self, dt):
        """
        a | c
        -----
        0 | 1
        1 | 0
        """

        n_steps = len(dt)/2
        return torch.concatenate(
            [torch.ones((n_steps, 1)), torch.zeros((n_steps, 1))],
            dim=-1
        )

    def _tableau_c_p1(self, dt):
        """
        Heun's Method, aka Trapazoidal Rule

        a | c
        -----
        0 | 0.5
        1 | 0.5
        """
        
        n_steps = len(dt)/2
        return torch.ones((n_steps, 2))*0.5


    def _geo_deltas(self, geos):
        return torch.sqrt(torch.sum((geos[1:] - geos[:-1])**2, dim=-1))
  
    def _remove_evals(self, evals, points):
        deltas = self._geo_deltas(geos)
        remove_mask = deltas < self.dxdx_remove
        #print("REMOVE DELTAS", deltas[:10])
        while torch.any(remove_mask):
            # Remove largest time point when geo_delta < dxdx_remove
            remove_mask = torch.concatenate(
                [
                    torch.tensor([False]), # Always keep t_init
                    remove_mask[:-2],
                    torch.tensor([remove_mask[-1] or remove_mask[-2]]),
                    torch.tensor([False]), # Always keep t_final
                ]
            )
            #print("N REMoVES", torch.sum(remove_mask), remove_mask[:10])

            #print("test not", remove_mask, ~remove_mask)
            eval_times = eval_times[~remove_mask]
            geos = geos[~remove_mask]
            deltas = self._geo_deltas(geos)
            remove_mask = deltas < self.dxdx_remove
        
        if len(eval_times) == 2:
            print("WARNING: dxdx is too large, all integration points have been removed")
        
        return geos, eval_times
 
    def _add_evals(self, ode_fxn, old_evals, old_points, points, old_idxs, new_idxs):
        if len(points) == 0:
            RuntimeWarning("Do not expect empty points to add.")
            return old_evals, old_points
        
        # Calculate new geometries
        #print("ADD PARALLEL EVAL TIMES", eval_times)
        new_evals = ode_fxn(points)
        
        # Place new geometries between existing 
        combined_evals = torch.zeros(
            (len(old_evals)+len(new_evals), new_evals.shape[-1]),
            requires_grad=False
        ).detach()
        #print("GEO SHAPES", geos.shape, old_geos.shape, idxs_old.shape)
        if len(old_idxs):
            combined_evals[old_idxs] = old_evals
        combined_evals[new_idxs] = new_evals

        # Place new times between existing 
        combined_points = torch.zeros(
            (len(old_points)+len(points), 1), requires_grad=False
        )
        #print("OLD IDXS", idxs_old[77:85])
        #print("NEW IDXS", idxs_new)
        if len(old_idxs):
            combined_points[old_idxs] = old_points
        combined_points[new_idxs] = points

        return combined_evals, combined_points