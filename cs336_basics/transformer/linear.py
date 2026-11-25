from math import sqrt
from git import Optional
import torch
import einx


class Linear(torch.nn.Module):
    def __init__(
            self,
            d_in: int,
            d_out: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
            ):
        super().__init__()
        var = 2/(d_in+d_out)
        std = sqrt(var)
        w = torch.empty([d_out, d_in], device = device, dtype = dtype)
        torch.nn.init.trunc_normal_(w, mean=0, std=std, a = -3*std, b = 3*std)
        self.weight = torch.nn.Parameter(w)
        
    def forward(self, x: torch.Tensor)-> torch.Tensor :
        logits = einx.dot("... [d_in], d_out [d_in] -> ... d_out", x, self.weight)
        return logits

