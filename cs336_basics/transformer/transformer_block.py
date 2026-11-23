from typing import Optional
import torch
from jaxtyping import Float
from torch import nn, Tensor
from .rms_norm import RMSNorm
from .attention import Multihead_self_attention
from .linear import Linear
from .activation import SwiGLU
class Transformer_block(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int,
            theta: float,
            device: Optional[torch.device] = None,
            dtype:Optional[torch.dtype] = None
    ):
        super().__init__()
        self.rms_1 = RMSNorm(d_model)
        self.rms_2 = RMSNorm(d_model)
        self.multi_head_self_attn = Multihead_self_attention(d_model, num_heads, max_seq_len, theta)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
    def forward(self, x:Float[Tensor, "... d_model"])->Float[Tensor, "... d_model"]:
        len_seq = x.shape[-2]
        y = x+self.multi_head_self_attn.forward(self.rms_1.forward(x), torch.tensor(list(range(1, len_seq+1)))) #generate a default token_positions index tensor to run rope
        return y + self.ffn.forward(self.rms_2.forward(y))