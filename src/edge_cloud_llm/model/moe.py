from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    def __init__(self, n_embd: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, mult * n_embd),
            nn.GELU(),
            nn.Linear(mult * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class MoELayer(nn.Module):
    def __init__(self, n_embd: int, num_experts: int = 4):
        super().__init__()
        self.num_experts = num_experts

        # Router: decides which expert to use
        self.router = nn.Linear(n_embd, num_experts)

        # Experts
        self.experts = nn.ModuleList([
            ExpertFFN(n_embd) for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        x: (B, T, C)
        """
        B, T, C = x.shape

        # Flatten tokens
        x_flat = x.view(-1, C)  # (B*T, C)

        # Router scores
        router_logits = self.router(x_flat)  # (B*T, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-1 expert selection
        # top_expert = torch.argmax(router_probs, dim=-1)  # (B*T,)
        topk_vals, topk_idx = torch.topk(router_probs, k=2, dim=-1)

        # Output buffer
        output = torch.zeros_like(x_flat)

        for i in range(2):  # top-2 experts
            expert_ids = topk_idx[:, i]
            weights = topk_vals[:, i]

            for expert_id in range(self.num_experts):
                mask = expert_ids == expert_id
                if mask.sum() == 0:
                    continue

                selected = x_flat[mask]
                out = self.experts[expert_id](selected)

                output[mask] += out * weights[mask].unsqueeze(-1)

        # Reshape back
        output = output.view(B, T, C)

        # ---- Load balancing loss ----
        mean_probs = router_probs.mean(dim=0)
        balance_loss = (mean_probs * mean_probs).sum()

        return output, balance_loss