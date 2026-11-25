import torch, einx
from jaxtyping import Float, Int
from torch import Tensor
from ..transformer import softmax
def cross_entropy(
        inputs: Float[Tensor, "... vocab_size"],
        targets: Int[Tensor, "..."]
        )-> Float[Tensor, ""]:
    shifted = inputs-torch.amax(inputs, -1, True)
    target_logits = einx.get_at("... [v], ... -> ...", shifted, targets)
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    return torch.mean(log_sum_exp - target_logits)
