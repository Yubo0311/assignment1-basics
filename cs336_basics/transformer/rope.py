import einx
import torch
from typing import Optional

from torch import Tensor, nn

class RoPE(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len:int,
            device: Optional[torch.device] = None
    )->None:
        super().__init__()
        exponential = -torch.tensor([2*k/d_k for k in range(d_k//2)])
        thetas = einx.multiply("seq_len, half_d -> seq_len half_d", torch.tensor(range(max_seq_len)),torch.pow(theta, exponential))
        cosines = thetas.cos()
        sines = thetas.sin()
        self.register_buffer("cos_theta", cosines, persistent=False)
        self.register_buffer("sin_theta", sines, persistent=False)
        
    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        len_seq = x.shape[-2]
        x_even = x[...,0::2]
        x_odd = x[...,1::2]
        cos = self.cos_theta[token_positions]
        sin = self.sin_theta[token_positions]
        x_even_rot = einx.multiply(
            "batch seq half_d, seq half_d -> batch seq half_d",
            x_even,
            cos) - einx.multiply(
            "batch seq half_d, seq half_d -> batch seq half_d",
            x_odd,
            sin
            )
        x_odd_rot = einx.multiply(
            "batch seq half_d, seq half_d -> batch seq half_d",
            x_odd,
            cos) + einx.multiply(
            "batch seq half_d, seq half_d -> batch seq half_d",
            x_even,
            sin
            )
        return torch.stack((x_even_rot, x_odd_rot), dim = -1).flatten(-2)

        


