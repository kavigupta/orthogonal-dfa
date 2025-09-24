import torch


def normalize(v: torch.Tensor):
    v = v - v.mean()
    v = v / v.norm(p=2)
    return v

def corr_norm(a: torch.Tensor, b: torch.Tensor):
    """
    Compute the correlation between two vectors after normalizing them.
    """
    return torch.dot(a, b)
