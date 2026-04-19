"""
config.py  —  Single source of truth for the entire project.
=============================================================

Every knob for every phase lives here.  Scripts import what they need:

    from edge_cloud_llm.config import MODEL, DATA, BASE, SFT, build_model

To switch between Kaggle (fast) and full training, change the ONE active
preset block at the bottom of this file.  Nothing else needs to be touched.

Layout
------
  ModelConfig       — architecture  (shared by pretraining AND SFT)
  DataConfig        — data paths + slice sizes
  BaseTrainConfig   — pretraining training-loop settings
  SFTConfig         — SFT training-loop settings
  MODEL/DATA/BASE/SFT — the active instances every script imports
  build_model()     — convenience builder so GPT() args stay DRY
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL ARCHITECTURE  (used by BOTH train_base.py and train_sft.py)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """
    All architectural hyperparameters.

    Both train_base.py and train_sft.py call build_model(vocab_size) which
    reads from this config, so changing any field here propagates everywhere.

    Constraints:
      - n_embd must be divisible by n_head
      - (n_embd // n_head) must be even  (required by RoPE)
    """
    n_embd:      int   = 64       # hidden / embedding dimension
    n_head:      int   = 2        # number of attention heads
    n_layer:     int   = 2        # number of transformer blocks
    block_size:  int   = 64       # context window length in tokens
    dropout:     float = 0.1      # dropout used during pretraining
    rope_base:   float = 10000.0  # RoPE base frequency

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        )
        d_head = self.n_embd // self.n_head
        assert d_head % 2 == 0, (
            f"head dim ({d_head}) must be even for RoPE"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA  (paths + slice sizes)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """
    File paths and data-volume controls.

    train_chars / val_chars  — how many *characters* of WikiText-2 to load.
                               Smaller = faster.  None = use the full split.
    sft_max_examples         — cap on SFT JSONL rows.  None = use all.
    """
    tokenizer_path:   str            = "artifacts/tokenizer/bpe_tokenizer.json"
    sft_data_path:    str            = "artifacts/sft/train.jsonl"
    train_chars:      Optional[int]  = 5_000   # None → full WikiText train
    val_chars:        Optional[int]  = 500     # None → full WikiText val
    sft_max_examples: Optional[int]  = 500     # None → full Alpaca (~52k)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PRETRAINING  (train_base.py)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BaseTrainConfig:
    """Training-loop settings for scripts/train_base.py."""
    output_dir:       str            = "outputs/base"
    resume_from:      Optional[str]  = None
    learning_rate:    float          = 3e-4
    weight_decay:     float          = 0.01
    epochs:           int            = 1
    batch_size:       int            = 4
    grad_accum_steps: int            = 1
    warmup_steps:     int            = 20
    gradient_clip:    float          = 1.0
    use_amp:          bool           = True
    eval_batches:     Optional[int]  = 10


# ─────────────────────────────────────────────────────────────────────────────
# 4. SFT  (train_sft.py)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SFTConfig:
    """
    Training-loop settings for scripts/train_sft.py.

    The model architecture is NOT duplicated here — it is read from MODEL
    via build_model(), so there is exactly one place to change model size.
    """
    base_checkpoint:             Optional[str]  = "outputs/base/best.pt"
    output_dir:                  str            = "outputs/sft"
    # curriculum_stages: max (prompt+response) token length per epoch stage
    curriculum_stages:           List[int]      = field(default_factory=lambda: [64, 64, 128, 128])
    use_context_packing:         bool           = False
    use_lora:                    bool           = True
    lora_rank:                   int            = 4
    lora_alpha:                  float          = 8.0
    lora_dropout:                float          = 0.05
    lora_target_modules:         List[str]      = field(default_factory=lambda: ["qkv"])
    merge_weights_after_training:bool           = True
    learning_rate:               float          = 2e-4
    weight_decay:                float          = 0.01
    warmup_steps:                int            = 10
    epochs:                      int            = 2
    batch_size:                  int            = 2
    grad_accum_steps:            int            = 1
    use_amp:                     bool           = True
    gradient_clip:               float          = 1.0
    eval_batches:                Optional[int]  = 5


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ACTIVE PRESETS  ←  THE ONE PLACE YOU EDIT
# ─────────────────────────────────────────────────────────────────────────────
#
# Uncomment the preset you want.  Every script imports MODEL / DATA / BASE / SFT
# directly, so switching here is the only change needed anywhere in the project.
#
# ── Kaggle / fast (CPU or small GPU, whole pipeline in minutes) ────────────

# MODEL = ModelConfig(n_embd=64,  n_head=2, n_layer=2, block_size=64,  dropout=0.1)
# DATA  = DataConfig( train_chars=5_000, val_chars=500, sft_max_examples=500)
# BASE  = BaseTrainConfig(epochs=1, batch_size=4, grad_accum_steps=1,
#                         warmup_steps=20, eval_batches=5)
# SFT   = SFTConfig(  epochs=2, batch_size=2, grad_accum_steps=1,
#                     warmup_steps=10, curriculum_stages=[64, 64, 128, 128], lora_rank=4)

# ── Kaggle T4 — small model, full data ────────────────────────────────────
MODEL = ModelConfig(n_embd=128, n_head=4, n_layer=2, block_size=128, dropout=0.1)
DATA  = DataConfig( train_chars=None, val_chars=None, sft_max_examples=None)
BASE  = BaseTrainConfig(epochs=3, batch_size=8, grad_accum_steps=4,
                        warmup_steps=100, eval_batches=50)
SFT   = SFTConfig(  epochs=3, batch_size=4, grad_accum_steps=4,
                    warmup_steps=50, curriculum_stages=[128, 256, 512, 1024], lora_rank=8)

# ── Full training (Kaggle T4/P100, hours) ─────────────────────────────────
MODEL = ModelConfig(n_embd=128, n_head=4, n_layer=4, block_size=128, dropout=0.1)
DATA  = DataConfig( train_chars=None, val_chars=None, sft_max_examples=None)
BASE  = BaseTrainConfig(epochs=3, batch_size=8, grad_accum_steps=4,
                        warmup_steps=100, eval_batches=50)
SFT   = SFTConfig(  epochs=3, batch_size=4, grad_accum_steps=4,
                    warmup_steps=50, curriculum_stages=[128, 256, 512, 1024], lora_rank=8)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CONVENIENCE BUILDER  (keeps GPT() constructor args DRY)
# ─────────────────────────────────────────────────────────────────────────────

def build_model(vocab_size: int, cfg: ModelConfig = MODEL,
                dropout_override: Optional[float] = None):
    """
    Construct a GPT from a ModelConfig.

    dropout_override lets train_sft.py pass 0.0 without mutating the config.
    """
    from edge_cloud_llm.model.model_gpt import GPT
    dropout = dropout_override if dropout_override is not None else cfg.dropout
    return GPT(
        vocab_size = vocab_size,
        block_size = cfg.block_size,
        n_layer    = cfg.n_layer,
        n_head     = cfg.n_head,
        n_embd     = cfg.n_embd,
        dropout    = dropout,
        rope_base  = cfg.rope_base,
    )
