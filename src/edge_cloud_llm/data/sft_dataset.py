"""
data/sft_dataset.py
===================
All Phase 6 dataset utilities for Supervised Fine-Tuning.

Three classes:
  LengthCurriculumDataset  – wraps raw (prompt, response) pairs and exposes
                              a progressively larger subset per curriculum stage
  SFTCollator              – pads a batch and builds -100 masked labels
  PackedSFTDataset         – bins multiple short examples into one context window
                              with per-document attention masks to prevent leakage
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset

# PyTorch CrossEntropyLoss natively skips positions whose label == IGNORE_INDEX.
# -100 is the default; we make it explicit so every module uses the same value.
IGNORE_INDEX: int = -100


# ---------------------------------------------------------------------------
# 1. LengthCurriculumDataset
# ---------------------------------------------------------------------------

class LengthCurriculumDataset(Dataset):
    """
    Presents training examples in order of increasing length.

    Stage 0: only examples with total_len <= stages[0]
    Stage 1: total_len <= stages[1]
    ...
    Call .advance_stage() at the end of each epoch (or after N steps)
    to unlock the next bucket.

    Within each stage the examples are *sorted by length* so batches contain
    similarly-sized sequences → less padding → cheaper compute.

    Args:
        raw_data:        List of dicts with keys "prompt_ids" and "response_ids"
                         (both are plain Python lists of int token IDs).
        stages:          Ascending list of max-total-length thresholds.
                         Default: [128, 256, 512, 1024]
        sort_by_length:  If True (default), sort within each stage so batches
                         are length-homogeneous.
    """

    def __init__(
        self,
        raw_data: List[Dict],
        stages: List[int] | None = None,
        sort_by_length: bool = True,
    ):
        self.stages = stages or [128, 256, 512, 1024]
        self.current_stage = 0

        # Pre-compute total length for every example once
        lengths = [
            len(d["prompt_ids"]) + len(d["response_ids"])
            for d in raw_data
        ]

        if sort_by_length:
            order = sorted(range(len(raw_data)), key=lambda i: lengths[i])
            raw_data = [raw_data[i] for i in order]
            lengths   = [lengths[i]   for i in order]

        self._all_data = raw_data
        self._lengths  = lengths

        # Pre-compute which indices belong to each stage bucket
        self._stage_indices: List[List[int]] = []
        for max_len in self.stages:
            idx = [i for i, l in enumerate(self._lengths) if l <= max_len]
            self._stage_indices.append(idx)

        self._active_indices = self._stage_indices[0]
        self._log_stage()

    # ------------------------------------------------------------------ public

    def advance_stage(self) -> bool:
        """
        Move to the next curriculum stage.
        Returns True if advanced, False if already at the last stage.
        """
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            self._active_indices = self._stage_indices[self.current_stage]
            self._log_stage()
            return True
        print("[LengthCurriculum] Already at final stage – no advance.")
        return False

    def __len__(self) -> int:
        return len(self._active_indices)

    def __getitem__(self, idx: int) -> Dict:
        return self._all_data[self._active_indices[idx]]

    # ----------------------------------------------------------------- private

    def _log_stage(self) -> None:
        print(
            f"[LengthCurriculum] Stage {self.current_stage} "
            f"(max_len≤{self.stages[self.current_stage]}): "
            f"{len(self._active_indices)} examples"
        )


# ---------------------------------------------------------------------------
# 2. SFTCollator
# ---------------------------------------------------------------------------

@dataclass
class SFTBatch:
    """Typed container returned by SFTCollator.__call__."""
    input_ids:      torch.Tensor   # (B, T)  – full prompt+response token IDs
    labels:         torch.Tensor   # (B, T)  – prompt positions = IGNORE_INDEX
    attention_mask: torch.Tensor   # (B, T)  – 0 for padding, 1 for real tokens


class SFTCollator:
    """
    Converts a list of {"prompt_ids", "response_ids"} dicts into an SFTBatch.

    For each example the sequence is:
        [BOS] + prompt_ids + response_ids + [EOS]

    Labels mirror the sequence but with IGNORE_INDEX covering BOS+prompt,
    so loss is computed only on the response+EOS positions.

    Padding is dynamic (pad to the longest sequence in the batch) so short
    batches don't waste compute.

    Args:
        pad_token_id:  ID of <pad>
        bos_token_id:  ID of <bos>
        eos_token_id:  ID of <eos>
        max_length:    Hard truncation limit (in tokens).  Truncation removes
                       tokens from the END of the sequence – we prefer to
                       preserve as much of the response as possible.
    """

    def __init__(
        self,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        max_length: int = 1024,
    ):
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.max_length   = max_length

    def __call__(self, batch: List[Dict]) -> SFTBatch:
        all_input_ids: List[List[int]] = []
        all_labels:    List[List[int]] = []

        for item in batch:
            prompt_ids   = list(item["prompt_ids"])
            response_ids = list(item["response_ids"])

            # Full token sequence: the model reads every token
            full_ids: List[int] = (
                [self.bos_token_id]
                + prompt_ids
                + response_ids
                + [self.eos_token_id]
            )

            # Labels: mask everything up to (and including) the last prompt token
            n_masked = 1 + len(prompt_ids)   # BOS + prompt → ignore
            labels: List[int] = (
                [IGNORE_INDEX] * n_masked
                + response_ids               # predict each response token
                + [self.eos_token_id]        # predict EOS too
            )

            assert len(full_ids) == len(labels)

            # Truncate from the end when over budget
            if len(full_ids) > self.max_length:
                full_ids = full_ids[: self.max_length]
                labels   = labels[: self.max_length]

            all_input_ids.append(full_ids)
            all_labels.append(labels)

        # Dynamic padding: pad to the longest sequence in THIS batch only
        max_seq = max(len(ids) for ids in all_input_ids)

        padded_inputs:  List[List[int]] = []
        padded_labels:  List[List[int]] = []
        attn_masks:     List[List[int]] = []

        for input_ids, labels in zip(all_input_ids, all_labels):
            pad_len = max_seq - len(input_ids)
            padded_inputs.append(input_ids + [self.pad_token_id] * pad_len)
            padded_labels.append(labels    + [IGNORE_INDEX]      * pad_len)
            attn_masks.append(   [1] * len(input_ids) + [0] * pad_len)

        return SFTBatch(
            input_ids      = torch.tensor(padded_inputs, dtype=torch.long),
            labels         = torch.tensor(padded_labels, dtype=torch.long),
            attention_mask = torch.tensor(attn_masks,    dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# 3. PackedSFTDataset
# ---------------------------------------------------------------------------

class PackedSFTDataset(Dataset):
    """
    Bins multiple short SFT examples into a single context window to avoid
    padding waste.  Uses greedy first-fit packing.

    Each packed sequence carries a *block-diagonal attention mask* so that
    tokens from one example cannot attend to tokens from a different example.
    Without this, the model gets spurious cross-example context at training
    time that won't exist at inference time.

    Each item returned is a dict with:
        input_ids      – (context_length,) int64
        labels         – (context_length,) int64  [response positions only]
        attention_mask – (context_length, context_length) bool
                         True  = this (query, key) pair is allowed
                         False = masked out
    """

    def __init__(
        self,
        raw_data: List[Dict],
        context_length: int,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
    ):
        self.context_length = context_length
        self.pad_token_id   = pad_token_id

        self._sequences: List[Dict] = []
        self._pack(raw_data, bos_token_id, eos_token_id)
        print(f"[PackedSFTDataset] {len(raw_data)} examples → {len(self._sequences)} packed sequences")

    # ------------------------------------------------------------------ public

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Dict:
        item     = self._sequences[idx]
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        labels    = torch.tensor(item["labels"],    dtype=torch.long)
        doc_ids   = torch.tensor(item["doc_ids"],   dtype=torch.long)

        # Build block-diagonal causal mask (bool, True = can attend)
        T = self.context_length
        # query i can attend to key j iff: j <= i AND same doc AND not padding
        row = doc_ids.unsqueeze(1)   # (T, 1)
        col = doc_ids.unsqueeze(0)   # (1, T)
        same_doc   = row == col                                   # (T, T)
        causal     = torch.tril(torch.ones(T, T, dtype=torch.bool))
        not_pad    = (doc_ids != -1).unsqueeze(0).expand(T, T)    # row not padding
        attn_mask  = same_doc & causal & not_pad                  # (T, T)

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}

    # ----------------------------------------------------------------- private

    def _pack(
        self,
        raw_data: List[Dict],
        bos: int,
        eos: int,
    ) -> None:
        cur_input:  List[int] = []
        cur_labels: List[int] = []
        cur_docs:   List[int] = []
        doc_id = 0

        for ex in raw_data:
            p = [bos] + list(ex["prompt_ids"])
            r = list(ex["response_ids"]) + [eos]
            total = len(p) + len(r)

            if total > self.context_length:
                continue  # single example too long, skip

            if len(cur_input) + total > self.context_length:
                self._flush(cur_input, cur_labels, cur_docs)
                cur_input, cur_labels, cur_docs = [], [], []
                doc_id += 1

            cur_input.extend(p + r)
            cur_labels.extend([IGNORE_INDEX] * len(p) + r)
            cur_docs.extend([doc_id] * total)

        if cur_input:
            self._flush(cur_input, cur_labels, cur_docs)

    def _flush(
        self,
        input_ids: List[int],
        labels:    List[int],
        doc_ids:   List[int],
    ) -> None:
        pad = self.context_length - len(input_ids)
        self._sequences.append({
            "input_ids": input_ids + [self.pad_token_id] * pad,
            "labels":    labels    + [IGNORE_INDEX]      * pad,
            "doc_ids":   doc_ids   + [-1]                * pad,
        })
