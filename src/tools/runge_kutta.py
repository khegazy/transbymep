import torch
from . import tableau
from .solvers import degree, steps, ParallelAdaptiveStepsizeSolver

TABLEAU_CALCULATORS = {
    (steps.ADAPTIVE, 1) : (tableau.euler_tableau_b, tableau.trapezoid_tableau_b)
}

class RKParallelAdaptiveStepsizeSolver(ParallelAdaptiveStepsizeSolver):
    def __init__(self, p, atol, rtol, ode_fxn=None, t_init=0., t_final=1.):
        super().__init__(
            p=p, atol=atol, rtol=rtol, ode_fxn=ode_fxn, t_init=t_init, t_final=t_final
        )
        self.tableau_b_p, self.tableau_b_p1 = TABLEAU_CALCULATORS[(steps.ADAPTIVE, self.p)]
    

    def _calculate_integral(self, t, y, y0=0, degr=degree.P1):
        dt = t[1:] - t[:-1]
        tableau_c = self._calculate_tableau_b(dt, degr=degr)
        assert(tableau_c.shape[-1] == max(2, self.p))
        assert(torch.all(torch.sum(tableau_c, dim=-1) == 1))
        
        _t_p = t[0::self.p]
        h = _t_p[1:] - _t_p[:-1]
        
        # The first last point in the h-1 RK step is the first point in the h RK step
        """
        combined_tableau_c = tableau_c
        combined_tableau_c[:-1,-1] += tableau_c[1:,0]
        combined_tableau_c = torch.concatenate(
            [combined_tableau_c[0], torch.flatten(combined_tableau_c[1:,1:])]
        )
        """
        
        y_steps = torch.reshape(y[1:], (-1, max(self.p-1, 1)))
        y_steps = torch.concatenate(
            [
                torch.concatenate(
                    [torch.tensor([y[0]]), y_steps[:-1,-1]]
                ).unsqueeze(1),
                y_steps
            ],
            dim=1
        )

        print("SHAPES", h.shape, tableau_c.shape, y_steps.shape)
        # Shapes
        # dt: [t-1]
        # h: [dt/p x 1]
        # tableau_c: [dt/p x p]
        # y_steps: [dt/p x p]
        RK_steps = h*torch.sum(tableau_c*y_steps, dim=-1)   # Sum over k evaluations weighted by c
        integral = y0 + torch.sum(RK_steps)                    # Sum over all steps with step size h
        return integral, RK_steps


    def _calculate_tableau_b(self, dt, degr):
        if degr == degree.P:
            tableau_b_fxn = self.tableau_b_p
        elif degr == degree.P1:
            tableau_b_fxn = self.tableau_b_p1
        
        return tableau_b_fxn(dt)
