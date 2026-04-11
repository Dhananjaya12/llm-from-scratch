from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from edge_cloud_llm.data.tokenizer import BPETokenizer


class NextTokenDataset(Dataset):
    """
    Builds next-token prediction samples from one long token stream.

    x = [t0, t1, t2, ..., t_{n-1}]
    y = [t1, t2, t3, ..., t_n]
    """

    def __init__(self, token_ids: list[int], block_size: int):
        if len(token_ids) <= block_size:
            raise ValueError("token_ids length must be greater than block_size")

        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx: int):
        x = self.token_ids[idx : idx + self.block_size]
        y = self.token_ids[idx + 1 : idx + self.block_size + 1]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


def load_wikitext_split_text(split: str) -> str:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)

    texts = []
    for row in ds:
        text = row["text"]
        if text and text.strip():
            texts.append(text)

    return "\n".join(texts)


def load_wikitext_split_rows(split: str) -> list[str]:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)

    rows = []
    for row in ds:
        text = row["text"]
        if text and text.strip():
            rows.append(text)

    return rows


def build_token_ids(text: str, tokenizer: BPETokenizer) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=True)


def load_tokenizer(tokenizer_path: str | Path) -> BPETokenizer:
    return BPETokenizer.from_file(tokenizer_path)


def create_dataloaders(
    tokenizer_path: str | Path,
    block_size: int,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, BPETokenizer]:
    tokenizer = load_tokenizer(tokenizer_path)

    train_text = load_wikitext_split_text("train")
    val_text = load_wikitext_split_text("validation")

    train_ids = tokenizer.encode(train_text, add_special_tokens=True)
    val_ids = tokenizer.encode(val_text, add_special_tokens=True)

    train_dataset = NextTokenDataset(train_ids, block_size=block_size)
    val_dataset = NextTokenDataset(val_ids, block_size=block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )

    return train_loader, val_loader, tokenizer