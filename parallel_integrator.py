import torch
from src.tools.runge_kutta import RKParallelAdaptiveStepsizeSolver

integrator = RKParallelAdaptiveStepsizeSolver(p=1, atol=0.001, rtol=0.0001, remove_cut=0.105)
def sin2(t, w=0.2):
    return torch.sin(t*w*2*torch.pi)**2

def ifxn(t):
    return torch.ones_like(t)

def xfxn(t):
    return t**2

def expsin2(t, w=0.2):
    return torch.exp(-t/0.2)*torch.sin(t**2*w*2*torch.pi)**2

print(integrator.integrate(sin2))