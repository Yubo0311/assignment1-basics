from typing import Optional
import torch
from jaxtyping import Float
from torch import nn, Tensor
from .rms_norm import RMSNorm
from .attention import Multihead_self_attention
from .linear import Linear
from .ffn import SwiGLU
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
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.attn = Multihead_self_attention(d_model, num_heads, max_seq_len, theta)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
    def forward(self, x:Float[Tensor, "... d_model"])->Float[Tensor, "... d_model"]:
        len_seq = x.shape[-2]
        y = x+self.attn.forward(self.ln1.forward(x), torch.tensor(list(range(1, len_seq+1)))) #generate a default token_positions index tensor to run rope
        return y + self.ffn.forward(self.ln2.forward(y))