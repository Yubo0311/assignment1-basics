from typing import Optional

import torch
import einx
from torch import Tensor, nn
from jaxtyping import Float, Bool
from .softmax import softmax
from .linear import Linear
from .rope import RoPE
def scaled_dot_attention(
        queries: Float[Tensor, "batch_size ... queries d_k"],
        keys:Float[Tensor, "batch_size ... keys d_k"],
        values: Float[Tensor, "batch_size ... keys d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None
) -> Float[Tensor, "batch_size ... seq_len d_v"]:
    d_k = keys.shape[-1]
    similarity = einx.dot(
        "batch_size ... keys [d_k], batch_size ... queries [d_k] -> batch_size ... keys queries",
        keys,
        queries
        )/torch.sqrt(torch.tensor(d_k))
    if mask is not None:
        mask = einx.id("... queries keys -> ... keys queries", mask)
        similarity = similarity.masked_fill(~mask, -torch.inf)
    weights:Float[Tensor, "batch_size ... keys queries"] = softmax(similarity, -2)
    weighted_value = einx.dot("batch_size ... [key_seq_len] query_seq_len, batch_size ... [key_seq_len] d_v -> batch_size ... query_seq_len d_v", weights, values)
    return weighted_value

class Multihead_self_attention(nn.Module):
    def __init__(
            self, 
            d_model:int,
            num_heads:int,
            max_seq_len: Optional[int] = None,
            theta: Optional[float] = None,
            device : Optional[torch.device] = None,
            dtype : Optional[torch.dtype] = None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model//num_heads
        self.d_v = self.d_k
        self.enable_rope = False
        self.query_rope = None
        self.key_rope = None
        assert num_heads*self.d_k == d_model
        self.Q = Linear(d_model, d_model, device, dtype)
        self.K = Linear(d_model, d_model, device, dtype)
        self.V = Linear(d_model, d_model, device, dtype)
        if max_seq_len is not None and theta is not None:
            self.enable_rope = True
            self.query_rope = RoPE(theta, self.d_k, max_seq_len, device)
            self.key_rope = RoPE(theta, self.d_v, max_seq_len, device)
        self.O = Linear(d_model, d_model, device, dtype)

    def forward(self,x:Float[Tensor, "... d_model"], token_positions: Optional[Tensor] = None) -> Float[Tensor, "... d_model"]:
        seq_len = x.shape[-2]
        Qs = einx.id("... seq_len (num_heads d_head) ->  ... num_heads seq_len d_head" , self.Q.forward(x), num_heads = self.num_heads)
        Ks = einx.id("... seq_len (num_heads d_head) ->  ... num_heads seq_len d_head" , self.K.forward(x), num_heads = self.num_heads)
        Vs = einx.id("... seq_len (num_heads d_head) ->  ... num_heads seq_len d_head" , self.V.forward(x), num_heads = self.num_heads)
        if token_positions is not None and self.key_rope is not None:
            Ks = self.key_rope.forward(Ks, token_positions)
            Qs = self.query_rope.forward(Qs, token_positions)
        attn_values = scaled_dot_attention(Qs, Ks, Vs,  ~torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1))
        return einx.id("... num_heads seq_len d_head -> ... seq_len (num_heads d_head)",attn_values)



    
