from math import ceil
from typing import Optional
import einx
from torch import Tensor, nn
from .linear import Linear
import torch
class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor)->Tensor:
        return x*torch.sigmoid(x)

class GLU(nn.Module):
    def __init__(self, d_model:int, d_ff:int, device:Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.d_ff = d_ff
        self.linear1 = Linear(d_model, d_ff, device, dtype)
        self.linear2 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x:Tensor)->Tensor:
        return einx.multiply("... a, ... a -> ... a", torch.sigmoid(self.linear1.forward(x)), self.linear2.forward(x))

class SwiGLU(nn.Module):
    def __init__(self,
                 d_model:int,
                 d_ff:int = -1, 
                 device: Optional[torch.device] = None, 
                 dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff =  ceil(d_model*3/8)*64 if d_ff == -1 else d_ff
        self.linear1 = Linear(d_model, self.d_ff, device, dtype)
        self.linear2 = Linear(self.d_ff, d_model, device, dtype)
        self.linear3 = Linear(d_model, self.d_ff, device, dtype)
        self.silu = SiLU()

    def forward(self, x:Tensor)-> Tensor:
        return self.linear2.forward(
            einx.multiply(
                "... a, ... a -> ... a",
                self.silu.forward(self.linear1.forward(x)),
                self.linear3.forward(x)
            )
        )