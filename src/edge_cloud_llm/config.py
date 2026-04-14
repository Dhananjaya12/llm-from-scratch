"""
config.py
=========
Typed configuration dataclasses for the project.

Usage:
    from edge_cloud_llm.config import SFTConfig
    cfg = SFTConfig()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SFTConfig:
    """All knobs for a Phase-6 SFT run."""

    # ── Data ─────────────────────────────────────────────────────────────────
    sft_data_path: str = "artifacts/sft/train.jsonl"
    # Each line of the jsonl is: {"prompt": "...", "response": "..."}

    # ── Curriculum stages ────────────────────────────────────────────────────
    # Each value is the max total token length (prompt+response) allowed
    # in that curriculum stage.  Stages are advanced one per epoch.
    curriculum_stages: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])

    # ── Context packing ───────────────────────────────────────────────────────
    use_context_packing: bool = False   # set True for short datasets to save memory
    context_length: int = 1024         # used as max_length in SFTCollator
                                       # and as context_length in PackedSFTDataset

    # ── LoRA ─────────────────────────────────────────────────────────────────
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    # Names of nn.Linear sub-modules to replace with LoRALinear.
    # For your GPT model the fused QKV projection is called "qkv".
    lora_target_modules: List[str] = field(default_factory=lambda: ["qkv"])
    merge_weights_after_training: bool = True

    # ── Optimiser ────────────────────────────────────────────────────────────
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    epochs: int = 3
    batch_size: int = 4
    grad_accum_steps: int = 4
    use_amp: bool = True
    gradient_clip: float = 1.0

    # ── Checkpoint ───────────────────────────────────────────────────────────
    base_checkpoint: Optional[str] = "outputs/base/best.pt"
    output_dir: str = "outputs/sft"
    # How many validation batches to evaluate on each epoch (None = all)
    eval_batches: Optional[int] = 50
