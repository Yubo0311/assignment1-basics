import einx
from typing import Optional
from torch import nn, Tensor
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
        weights = torch.empty([d_out, d_in], device = device, dtype = dtype)
        nn.init.trunc_normal_(weights, mean=0, std=1, a = -3, b = 3)
        self.weights = nn.Parameter(weights)

    def forward(self, token_ids: Tensor)->Tensor:
        embeddings = self.weights[:, token_ids] # (embedding_dim, batch_num, seq_len)
        embeddings = einx.id("embedding_dim batch_num seq_len -> batch_num seq_len embedding_dim", embeddings)
        return embeddings