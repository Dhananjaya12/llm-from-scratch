"""
model/lora.py
=============
LoRA (Low-Rank Adaptation) helpers that work with your existing GPT model.

Two public functions:
    apply_lora_to_model(model, ...)  – replaces target Linear layers in-place,
                                       freezes all base weights, returns model
    merge_lora_weights(model)        – folds A·B back into W for zero-overhead
                                       inference; call after SFT training

One public class:
    LoRALinear                       – drop-in nn.Linear with a trainable
                                       low-rank bypass: y = xW + x(AB)·scale
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Replaces nn.Linear with a frozen base weight + trainable low-rank adapter.

        y = x @ W.T  +  x @ (A @ B).T * (alpha / rank)
            ^^^^^^^^      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            frozen          trainable  (only 2 tiny matrices)

    Init:
        A  ~ Kaiming-uniform  (standard)
        B  = zeros            ← critical: adapter starts as identity (no-op)
                                so training begins from the pretrained model.

    Args:
        in_features:   same as nn.Linear
        out_features:  same as nn.Linear
        rank:          LoRA rank r  (try 4, 8, 16, 32, 64)
        lora_alpha:    scaling hyperparameter (often rank × 2)
        lora_dropout:  dropout probability on the LoRA path (0.0 to disable)
        bias:          whether to include a bias term (frozen)
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        rank:         int   = 8,
        lora_alpha:   float = 16.0,
        lora_dropout: float = 0.05,
        bias:         bool  = True,
    ):
        super().__init__()
        self.rank         = rank
        self.scale        = lora_alpha / rank   # applied to adapter output
        self._merged      = False

        # Frozen base weight (copied from the pretrained layer)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias_param = (
            nn.Parameter(torch.zeros(out_features), requires_grad=False)
            if bias else None
        )

        # Trainable adapter: W_delta = A @ B,  shape same as W
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))  # ← zeros!

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_drop = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

    # ------------------------------------------------------------------ nn.Module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base (frozen) path
        out = nn.functional.linear(x, self.weight, self.bias_param)

        if not self._merged:
            # LoRA path: x @ A @ B  (both projections happen in fp32/fp16)
            out = out + self.lora_drop(x) @ self.lora_A @ self.lora_B * self.scale

        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.weight.shape[1]}, out={self.weight.shape[0]}, "
            f"rank={self.rank}, merged={self._merged}"
        )

    # ------------------------------------------------------------------ merge / unmerge

    def merge(self) -> None:
        """Fold A·B into W so inference has zero extra computation."""
        if not self._merged:
            # (A @ B).T has shape (out_features, in_features) = same as weight
            delta = (self.lora_A @ self.lora_B).T * self.scale
            self.weight.data += delta
            self._merged = True

    def unmerge(self) -> None:
        """Undo the merge (e.g. to resume training)."""
        if self._merged:
            delta = (self.lora_A @ self.lora_B).T * self.scale
            self.weight.data -= delta
            self._merged = False


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def apply_lora_to_model(
    model:          nn.Module,
    target_modules: List[str],
    rank:           int   = 8,
    lora_alpha:     float = 16.0,
    lora_dropout:   float = 0.05,
) -> nn.Module:
    """
    Freeze the entire model then replace every nn.Linear whose dotted name ends
    with one of the strings in *target_modules* with a LoRALinear.

    Typical target_modules for an attention-only LoRA:
        ["qkv"]              ← your GPT fuses Q,K,V into one matrix
    Or for separate projections:
        ["q_proj", "v_proj"] ← most common choice from the LoRA paper

    Args:
        model:          Pretrained GPT (or any nn.Module).
        target_modules: Substrings matched against the end of each layer name.
        rank:           LoRA rank r.
        lora_alpha:     Scaling factor α.
        lora_dropout:   Dropout on the LoRA path.

    Returns:
        The same model object, modified in-place.
    """
    # Step 1 – freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    replaced = 0
    # Step 2 – swap target layers
    # named_modules() gives (dotted_name, module) pairs for the whole tree
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(full_name.endswith(t) for t in target_modules):
            continue

        # Navigate to parent
        parts  = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr = parts[-1]

        lora_layer = LoRALinear(
            in_features  = module.in_features,
            out_features = module.out_features,
            rank         = rank,
            lora_alpha   = lora_alpha,
            lora_dropout = lora_dropout,
            bias         = module.bias is not None,
        )
        lora_layer.weight.data.copy_(module.weight.data)
        if module.bias is not None and lora_layer.bias_param is not None:
            lora_layer.bias_param.data.copy_(module.bias.data)

        setattr(parent, attr, lora_layer)
        replaced += 1

    if replaced == 0:
        raise ValueError(
            f"No layers matched target_modules={target_modules}. "
            f"Available Linear layer names: "
            + str([n for n, m in model.named_modules() if isinstance(m, nn.Linear)])
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(
        f"[LoRA] Replaced {replaced} layers | "
        f"trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)"
    )
    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Walk the model tree and call .merge() on every LoRALinear.
    Safe to call multiple times (idempotent).

    Returns the model (in-place operation).
    """
    n = 0
    for module in model.modules():
        if isinstance(module, LoRALinear) and not module._merged:
            module.merge()
            n += 1
    print(f"[LoRA] Merged {n} LoRA layers into base weights.")
    return model


def unmerge_lora_weights(model: nn.Module) -> nn.Module:
    """Undo merge on every LoRALinear (e.g. to resume training)."""
    n = 0
    for module in model.modules():
        if isinstance(module, LoRALinear) and module._merged:
            module.unmerge()
            n += 1
    print(f"[LoRA] Unmerged {n} LoRA layers.")
    return model
