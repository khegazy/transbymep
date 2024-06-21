import torch

def _compute_error_ratios(y_p, y_p1, rtol, atol, norm):
    error_estimate = torch.abs(y_p1 - y_p)
    error_tol = atol + rtol*torch.max(y_p.abs(), y_p1.abs())
    return norm(error_estimate/error_tol).abs()