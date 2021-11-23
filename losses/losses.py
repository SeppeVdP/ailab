import torch

import utilities.utils as utils


def mse(input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """TODO: implement this method"""
    minus = input_tensor - target
    power = pow(minus, 2)
    return torch.mean(power)
