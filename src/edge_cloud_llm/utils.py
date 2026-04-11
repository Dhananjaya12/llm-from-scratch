from __future__ import annotations

import torch


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    filtered = logits.clone()

    if top_k is not None and top_k > 0:
        top_k = min(top_k, filtered.size(-1))
        values, _ = torch.topk(filtered, top_k)
        min_topk = values[:, [-1]]
        filtered[filtered < min_topk] = float("-inf")

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_mask = cumulative_probs > top_p
        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[:, 0] = False

        for b in range(filtered.size(0)):
            remove_ids = sorted_indices[b][sorted_mask[b]]
            filtered[b, remove_ids] = float("-inf")

    return filtered