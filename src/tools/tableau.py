import torch

def euler_tableau_b(dt):
    """
    c | b
    -----
    0 | 1
    1 | 0
    """

    return torch.concatenate(
        [torch.ones((1, 1)), torch.zeros((1, 1))],
        dim=-1
    )

def trapezoid_tableau_b(dt):
    """
    Heun's Method, aka Trapazoidal Rule

    c | b
    -----
    0 | 0.5
    1 | 0.5
    """
    
    return torch.ones((1, 2))*0.5