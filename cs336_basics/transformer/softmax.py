from typing import Optional

import einx
import torch
from torch import Tensor, nn
from jaxtyping import Float

def softmax(x:Float[Tensor, "..."], dim:int) -> Float[Tensor, "..."]:
    """softmax function

    Args:
        x (Tensor): input tensor
        dim (int): index of the dimension to be reduced

    Returns:
        Tensor: A tensor with dim-indexed dimension reduced to 1
    """
    x_stable = x - torch.amax(x, dim, True)
    x_exp = torch.exp(x_stable)
    x_sum = torch.sum(x_exp, dim, True)
    return torch.divide(x_exp, x_sum)
