import torch
import torch.nn as nn

from edge_cloud_llm.model.embeddings import TokenPositionEmbedding
from edge_cloud_llm.model.block import TransformerBlock


class MiniTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        num_heads: int,
        num_layers: int,
        ff_hidden_dim: int,
    ):
        super().__init__()

        self.embedding = TokenPositionEmbedding(vocab_size, d_model, max_seq_len)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_hidden_dim=ff_hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        x = self.embedding(input_ids)

        mask = self._build_causal_mask(seq_len, input_ids.device)

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits