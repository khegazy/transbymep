import jax.numpy as jnp
import diffrax

solver_dict = {
    "Tsit5" : diffrax.Tsit5,
    "Dopri5" : diffrax.Dopri5
}

class ODEintegrator():
    def __init__(self, potential, solver="Tsit5"):
        self.potential = potential
        #self.save_at = diffrax.SaveAt(dense=True)
        self.stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
        
        if solver not in solver_dict:
            raise ValueError("Cannot handle solver {solver}, add it to solver_dict or use {solver_dict.keys}")
        self.solver = solver_dict[solver]()
    
    def path_integral(self, ode_fxn, t_init=0., t_final=1.):
       #ode_term = diffrax.ODETerm(path.pes_path)
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(ode_fxn),
            self.solver,
            t0=t_init,
            t1=t_final,
            dt0=None,
            y0=0,
            saveat=diffrax.SaveAt(ts=jnp.array([t_init, t_final])),
            stepsize_controller=self.stepsize_controller,
            max_steps=int(1e6)
        )
        return solution.ys[-1]

"""
from jaxopt import Bisection
from functools import partial
@jax.jit
def integrand(potential_eval, path_eval, t, y, *args):
    return jnp.linalg.norm(potential_eval(path_eval(t)))

class ODEintegrator:
    def __init__(self, potential):
        self.potential = potential
        self.solution = None

        self.solver = Dopri5()
        self.save_at = SaveAt(dense=True)
        self.stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

    def jacobian(self, t, y, *args):
        #jac =jax.jacfwd(self.fxn)(t) 
        #print("JAC", jac)
        return jnp.linalg.norm(self.jacobian_(t))
    
    def integrand(self, t, y, *args):
        return jnp.linalg.norm(self.potential.eval(self.path.eval(t)))

    def integrate(self, path, t_init=0, t_final=1.):
        self.path = path
        #self.jacobian_ = jax.jacfwd(self.fxn)
        #self.term = ODETerm(self.jacobian)
        self.term = ODETerm(self.integrand)
        #partial(integrand, self.potential.eval, self.path.eval))
        self.solution = diffeqsolve(
            self.term,
            self.solver,
            t0=t_init,
            t1=t_final,
            dt0=None,
            y0=0,
            saveat=self.save_at,
            stepsize_controller=self.stepsize_controller
        )
        return self.solution.evaluate(t_final)
    
    def eval(self, t):
        if self.solution is None:
            self.integrate()
        return self.solution(t)

    def partition_path(self, n_partitions):
        targets = (jnp.arange(n_partitions-2) + 1.)/n_partitions*self.eval(1.)
        print(targets)

        bisec = Bisection(
            optimality_fun=lambda x, target: self.eval(x) - target,
            lower=0.,
            upper=1.
        )
        result = [bisec.run(target=t).params for t in targets]
        #print(t, result, sol.evaluate(result))
        return result


def f(t):
    return t**2,t**3
"""