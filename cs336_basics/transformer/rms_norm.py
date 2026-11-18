from typing import Optional
import einx
import torch
from torch import Tensor
from torch import nn
class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model:int,
            eps:float = 1e-5,
            device:Optional[torch.dtype] = None,
            dtype:Optional[torch.dtype] = None
    ):
        """Root Mean Square Normalization

        Args:
            d_model (int): hidden dimension of the model
            eps (float, optional): epsilon value for numerical stability. Defaults to 1e-5.
            device (_type_, optional): device of parameters. Defaults to None.
            dtype (_type_, optional): dtype of parameters. Defaults to None.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.dtype = dtype
        self.device = device
        self.weights = nn.Parameter(torch.ones(size = (d_model,) ,dtype = dtype, device = device))

    def forward(self, x: Tensor)->Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32) 
        def layer_rms_norm(y:Tensor):
            y_square_sum = torch.sum(y.square())
            y_square_mean = y_square_sum/self.d_model + self.eps
            y_rms = torch.sqrt(y_square_mean)
            return y_rms
        einx_layer_rms_norm = einx.torch.adapt_with_vmap(layer_rms_norm)
        rms_x = einx_layer_rms_norm("batch_size seq_len [d_model] -> batch_size seq_len", x)
        x = einx.divide("batch_size seq_len d_model, batch_size seq_len -> batch_size seq_len d_model", x, rms_x)
        result = einx.multiply("batch_size seq_len d_model, d_model -> batch_size seq_len d_model", x, self.weights)
        return result.to(in_dtype)