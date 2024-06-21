import torch

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