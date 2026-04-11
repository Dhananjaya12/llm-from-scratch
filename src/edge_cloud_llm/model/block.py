import torch
import torch.nn as nn

from edge_cloud_llm.model.attention import MultiHeadSelfAttention
from edge_cloud_llm.model.feedforward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_hidden_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x