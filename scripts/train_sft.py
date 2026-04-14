"""
scripts/train_sft.py
=====================
Phase 6 SFT entry-point.  Mirrors the style of train_base.py.

Steps:
    1. Load pretrained base checkpoint (outputs/base/best.pt)
    2. Optionally apply LoRA – freeze base weights, add trainable adapters
    3. Build LengthCurriculumDataset from the JSONL produced by prepare_sft_data.py
    4. Run SFTTrainer for N epochs, advancing the curriculum each epoch
    5. Optionally merge LoRA weights into the base model for clean inference

Usage:
    python scripts/train_sft.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

from edge_cloud_llm.config import SFTConfig
from edge_cloud_llm.data import (
    BPETokenizer,
    LengthCurriculumDataset,
    PackedSFTDataset,
    SFTCollator,
)
from edge_cloud_llm.model import GPT, apply_lora_to_model, merge_lora_weights
from edge_cloud_llm.training import SFTTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sft_jsonl(path: str, tokenizer: BPETokenizer) -> list[dict]:
    """
    Read a JSONL file of {"prompt": str, "response": str} records.
    Returns a list of {"prompt_ids": list[int], "response_ids": list[int]}.
    """
    records = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            prompt_ids   = tokenizer.encode(row["prompt"],   add_special_tokens=False)
            response_ids = tokenizer.encode(row["response"], add_special_tokens=False)

            if not prompt_ids or not response_ids:
                skipped += 1
                continue

            records.append({"prompt_ids": prompt_ids, "response_ids": response_ids})

    print(f"Loaded {len(records)} SFT examples  ({skipped} skipped)")
    return records


def build_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Cosine LR with linear warm-up – same schedule as train_base.py."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = SFTConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── 1. Tokenizer ─────────────────────────────────────────────────────────
    tokenizer_path = "artifacts/tokenizer/bpe_tokenizer.json"
    tokenizer = BPETokenizer.from_file(tokenizer_path)
    print(f"Vocab size: {tokenizer.vocab_size}")

    # ── 2. Base model ─────────────────────────────────────────────────────────
    model = GPT(
        vocab_size  = tokenizer.vocab_size,
        block_size  = cfg.context_length,
        n_layer     = 4,
        n_head      = 4,
        n_embd      = 128,
        dropout     = 0.0,      # disable dropout during SFT (common practice)
        rope_base   = 10000.0,
    )

    # Load pretrained weights if available
    if cfg.base_checkpoint and Path(cfg.base_checkpoint).exists():
        ckpt = torch.load(cfg.base_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded base checkpoint: {cfg.base_checkpoint}")
    else:
        print("WARNING: no base checkpoint found – training from random init.")

    # ── 3. Optional LoRA ─────────────────────────────────────────────────────
    if cfg.use_lora:
        model = apply_lora_to_model(
            model,
            target_modules = cfg.lora_target_modules,
            rank           = cfg.lora_rank,
            lora_alpha     = cfg.lora_alpha,
            lora_dropout   = cfg.lora_dropout,
        )
        # Only the LoRA adapter parameters need gradients
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        # Full fine-tune: all params are trainable
        for p in model.parameters():
            p.requires_grad_(True)
        trainable_params = list(model.parameters())

    # ── 4. SFT dataset ────────────────────────────────────────────────────────
    raw_data = load_sft_jsonl(cfg.sft_data_path, tokenizer)

    if cfg.use_context_packing:
        # Pack multiple examples per context window (memory-efficient)
        train_dataset = PackedSFTDataset(
            raw_data       = raw_data,
            context_length = cfg.context_length,
            bos_token_id   = tokenizer.bos_token_id,
            eos_token_id   = tokenizer.eos_token_id,
            pad_token_id   = tokenizer.pad_token_id,
        )
        # PackedSFTDataset returns dicts – we still need a collate_fn
        # that stacks them into tensors; use a minimal lambda here
        def pack_collate(batch):
            from edge_cloud_llm.data.sft_dataset import SFTBatch
            import torch
            return SFTBatch(
                input_ids      = torch.stack([b["input_ids"]      for b in batch]),
                labels         = torch.stack([b["labels"]         for b in batch]),
                attention_mask = torch.stack([b["attention_mask"] for b in batch]),
            )
        collate_fn = pack_collate
    else:
        # Standard curriculum: short examples first, growing over epochs
        train_dataset = LengthCurriculumDataset(
            raw_data = raw_data,
            stages   = cfg.curriculum_stages,
        )
        collate_fn = SFTCollator(
            pad_token_id = tokenizer.pad_token_id,
            bos_token_id = tokenizer.bos_token_id,
            eos_token_id = tokenizer.eos_token_id,
            max_length   = cfg.context_length,
        )

    # ── 5. Optimiser + Scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr           = cfg.learning_rate,
        weight_decay = cfg.weight_decay,
    )

    steps_per_epoch = math.ceil(len(train_dataset) / cfg.batch_size / cfg.grad_accum_steps)
    total_steps     = steps_per_epoch * cfg.epochs
    scheduler = build_lr_scheduler(optimizer, cfg.warmup_steps, total_steps)

    # ── 6. Trainer ─────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model            = model,
        optimizer        = optimizer,
        device           = device,
        sft_dataset      = train_dataset,
        collate_fn       = collate_fn,
        batch_size       = cfg.batch_size,
        val_loader       = None,          # swap for a val SFTBatch loader if desired
        output_dir       = cfg.output_dir,
        scheduler        = scheduler,
        grad_accum_steps = cfg.grad_accum_steps,
        use_amp          = cfg.use_amp,
        gradient_clip    = cfg.gradient_clip,
    )

    # ── 7. Training loop ───────────────────────────────────────────────────────
    best_loss = float("inf")
    print(f"\n{'='*60}")
    print("Starting SFT training")
    print(f"  Epochs:        {cfg.epochs}")
    print(f"  Batch size:    {cfg.batch_size}  (×{cfg.grad_accum_steps} accum = {cfg.batch_size*cfg.grad_accum_steps} effective)")
    print(f"  LoRA:          {cfg.use_lora}  (rank={cfg.lora_rank})")
    print(f"  Context pack:  {cfg.use_context_packing}")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg.epochs + 1):
        print(f"\n── Epoch {epoch}/{cfg.epochs} ──")

        train_loss = trainer.train_epoch()
        print(f"  train_loss: {train_loss:.4f}  |  lr: {optimizer.param_groups[0]['lr']:.2e}")

        trainer.log_metrics(epoch, train_loss, val_loss=0.0)
        trainer.save_checkpoint(f"sft_epoch_{epoch}.pt", epoch=epoch)

        if train_loss < best_loss:
            best_loss = train_loss
            trainer.save_checkpoint("sft_best.pt", epoch=epoch)
            print("  → new best checkpoint saved")

        # Advance to longer examples next epoch
        if not cfg.use_context_packing and isinstance(train_dataset, LengthCurriculumDataset):
            train_dataset.advance_stage()

    # ── 8. Optional LoRA merge ─────────────────────────────────────────────────
    if cfg.use_lora and cfg.merge_weights_after_training:
        print("\nMerging LoRA adapters into base weights …")
        merge_lora_weights(model)
        merged_path = Path(cfg.output_dir) / "sft_merged.pt"
        torch.save({"model_state_dict": model.state_dict()}, merged_path)
        print(f"Merged model saved to {merged_path}")

    print(f"\nSFT complete.  Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
