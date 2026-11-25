from torch import Tensor, nn
from jaxtyping import Float, Int
from .tokenizer import train_bpe, Tokenizer
from .transformer import Embedding, Transformer_block, Linear, softmax, RMSNorm
class Transformer_LM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([Transformer_block(d_model, num_heads, d_ff, context_length, rope_theta) for i in range(num_layers)])
        self.lm_head = Linear(d_model, vocab_size)
        self.ln_final = RMSNorm(d_model)


    def forward(self, x:Int[Tensor, " batch_size sequence_length"])->Float[Tensor, " batch_size sequence_length vocab_size"]:
        x = self.token_embeddings.forward(x)
        for block in self.layers:
            x = block.forward(x)
        x = self.ln_final.forward(x)
        x = self.lm_head.forward(x)
        return x

        