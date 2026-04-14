"""
training/sft_trainer.py
========================
SFTTrainer extends the existing Trainer with three SFT-specific concerns:

  1. Input format  – expects SFTBatch (input_ids + labels + attention_mask)
                     instead of the (x, y) tuples used in pre-training.
  2. Loss masking  – labels already contain -100 for prompt positions;
                     nn.CrossEntropyLoss(ignore_index=-100) handles the rest.
  3. Curriculum    – calls dataset.advance_stage() at the end of each epoch
                     so the model graduates to longer examples over time.

The base Trainer handles: gradient accumulation, AMP, LR scheduler,
checkpoint save/load, and CSV logging. SFTTrainer reuses all of that.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from edge_cloud_llm.training.trainer import Trainer
from edge_cloud_llm.data.sft_dataset import (
    SFTBatch,
    LengthCurriculumDataset,
    IGNORE_INDEX,
)


class SFTTrainer(Trainer):
    """
    Drop-in replacement for Trainer when doing supervised fine-tuning.

    The key differences vs. the base Trainer:

    * train_epoch() iterates over SFTBatch objects rather than (x, y) pairs.
    * The loss function ignores label positions == IGNORE_INDEX (i.e. the
      prompt tokens that were masked during collation).
    * After every epoch it tries to advance the length curriculum.

    Args:
        sft_dataset:  A LengthCurriculumDataset (or any Dataset returning
                      {"input_ids", "labels", "attention_mask"} dicts).
        collate_fn:   Your SFTCollator instance.
        batch_size:   Batch size for the SFT DataLoader.
        All other kwargs are forwarded unchanged to the base Trainer.

    Note: the base Trainer's val_loader is still used for perplexity
    evaluation between epochs if provided.  You can pass val_loader=None
    to skip validation.
    """

    def __init__(
        self,
        model,
        optimizer,
        device: str,
        sft_dataset,        # LengthCurriculumDataset
        collate_fn,         # SFTCollator
        batch_size: int = 8,
        val_loader=None,
        output_dir: str = "outputs/sft",
        scheduler=None,
        grad_accum_steps: int = 1,
        use_amp: bool = False,
        gradient_clip: float = 1.0,
    ):
        # Base Trainer still gets a val_loader (can be None → skips eval)
        super().__init__(
            model            = model,
            optimizer        = optimizer,
            device           = device,
            train_loader     = None,      # we build it ourselves per epoch
            val_loader       = val_loader,
            output_dir       = output_dir,
            scheduler        = scheduler,
            grad_accum_steps = grad_accum_steps,
            use_amp          = use_amp,
        )
        self.sft_dataset    = sft_dataset
        self.collate_fn     = collate_fn
        self.batch_size     = batch_size
        self.gradient_clip  = gradient_clip

        # SFT loss: standard cross-entropy, but skip masked positions
        self.sft_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # ------------------------------------------------------------------

    def train_epoch(self) -> float:          # type: ignore[override]
        """
        One epoch of SFT.  Builds a fresh DataLoader every epoch so that
        after advance_stage() the new (longer) examples are included.
        """
        self.model.train()

        loader = DataLoader(
            self.sft_dataset,
            batch_size  = self.batch_size,
            shuffle     = True,
            collate_fn  = self.collate_fn,
            drop_last   = False,
        )

        total_loss  = 0.0
        total_steps = 0
        self.optimizer.zero_grad()

        pbar = tqdm(loader, desc="SFT training", leave=True)

        for step_idx, batch in enumerate(pbar, start=1):
            # batch is an SFTBatch dataclass
            input_ids      = batch.input_ids.to(self.device)       # (B, T)
            labels         = batch.labels.to(self.device)          # (B, T)
            attention_mask = batch.attention_mask.to(self.device)  # (B, T)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                # Your GPT forward: (idx, targets, use_cache) → (logits, loss, kv)
                # We pass targets=None and compute loss ourselves so we can use
                # our ignore_index-aware loss function.
                logits, _gpt_loss, _ = self.model(
                    input_ids, targets=None, use_cache=False
                )

                # Shift for next-token prediction:
                #   logits[:, :-1, :] predicts position 1..T
                #   labels[:,  1:  ] are the ground-truth for positions 1..T
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:  ].contiguous()

                loss = self.sft_loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if step_idx % self.grad_accum_steps == 0:
                # Gradient clipping before optimizer step
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in self.model.parameters() if p.requires_grad),
                    self.gradient_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()

            true_loss = loss.item() * self.grad_accum_steps
            total_loss  += true_loss
            total_steps += 1

            pbar.set_postfix(
                loss=f"{true_loss:.4f}",
                avg =f"{total_loss/total_steps:.4f}",
                lr  =f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )

        # Flush leftover gradient accumulation
        if total_steps % self.grad_accum_steps != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                (p for p in self.model.parameters() if p.requires_grad),
                self.gradient_clip,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()

        return total_loss / max(total_steps, 1)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_sft(self, val_loader, max_batches: int | None = None) -> float:
        """
        Evaluate SFT loss on a validation set of (prompt, response) pairs.
        The loader should yield SFTBatch objects (use the same SFTCollator).
        """
        self.model.eval()
        total_loss  = 0.0
        total_steps = 0

        pbar = tqdm(val_loader, desc="SFT eval", leave=True)

        for i, batch in enumerate(pbar, start=1):
            if max_batches is not None and i > max_batches:
                break

            input_ids = batch.input_ids.to(self.device)
            labels    = batch.labels.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits, _, _ = self.model(input_ids, targets=None, use_cache=False)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:   ].contiguous()
                loss = self.sft_loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

            total_loss  += loss.item()
            total_steps += 1
            pbar.set_postfix(loss=f"{total_loss/total_steps:.4f}")

        return total_loss / max(total_steps, 1)
