import einx
from typing import Optional
from torch import nn, Tensor
from jaxtyping import Int, Float
import torch
class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings:int,
            embedding_dim: int,
            device :Optional[torch.device] = None,
            dtype :Optional[torch.dtype] = None
    ):
        super().__init__()
        d_out = embedding_dim
        d_in = num_embeddings
        weight = torch.empty([d_in,d_out], device = device, dtype = dtype)
        nn.init.trunc_normal_(weight, mean=0, std=1, a = -3, b = 3)
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: Int[Tensor, " ..."])->Float[Tensor, " ... d_model"]:
        embeddings = self.weight[token_ids] 
        return embeddings