import jax
import jax.numpy as jnp
from jax import grad
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController
from jaxopt import Bisection
from functools import partial


def path_integral(path_eval, potential_eval, solver, save_at, stepsize_controller, t_init=0., t_final=1.):
    #solver = Dopri5()
    #save_at = SaveAt(dense=True)
    #stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
    
    def integrand(t, y, *args):
        return jnp.linalg.norm(potential_eval(path_eval(t)))
    term = ODETerm(integrand)
    
    solution = diffeqsolve(
        term,
        solver,
        t0=t_init,
        t1=t_final,
        dt0=None,
        y0=0,
        saveat=save_at,
        stepsize_controller=stepsize_controller
    )
    return solution
    return solution.evaluate(t_final)

class ODEintegrator:
    def __init__(self, potential, path, t_init=0., t_final=1.):
        self.solver = Dopri5()
        self.save_at = SaveAt(dense=True)
        self.stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
        self.t_init = t_init
        self.t_final = t_final
        
        def integrand(t, y, *args):
            return jnp.linalg.norm(self.potential.eval(self.path.eval(t)))
        def path_integral(t_init=0., t_final=1.):
            term = ODETerm(integrand)
            solution = diffeqsolve(
                term,
                self.solver,
                t0=t_init,
                t1=t_final,
                dt0=None,
                y0=0,
                saveat=self.save_at,
                stepsize_controller=self.stepsize_controller
            )
            return solution

        self.path_integral = path_integral

"""
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