"""
scripts/prepare_sft_data.py
============================
Downloads the Alpaca instruction dataset and writes a JSONL file where each
line is {"prompt": "...", "response": "..."}.

The prompt is the classic Alpaca template:
    Below is an instruction...
    ### Instruction:
    {instruction}
    ### Input:
    {input}       ← omitted when empty
    ### Response:

The response is the "output" field from Alpaca.

Usage:
    python scripts/prepare_sft_data.py
    python scripts/prepare_sft_data.py --output artifacts/sft/train.jsonl --max_examples 5000
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    raise SystemExit("Install `datasets`: pip install datasets")


ALPACA_PROMPT_WITH_INPUT = """\
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

ALPACA_PROMPT_NO_INPUT = """\
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def format_example(row: dict) -> dict | None:
    instruction = (row.get("instruction") or "").strip()
    inp         = (row.get("input")       or "").strip()
    output      = (row.get("output")      or "").strip()

    if not instruction or not output:
        return None

    prompt = (
        ALPACA_PROMPT_WITH_INPUT.format(instruction=instruction, input=inp)
        if inp
        else ALPACA_PROMPT_NO_INPUT.format(instruction=instruction)
    )
    return {"prompt": prompt, "response": output}


def prepare_sft_data(output_path: str, max_examples: int | None = None) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading tatsu-lab/alpaca …")
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as f:
        for row in ds:
            if max_examples is not None and written >= max_examples:
                break
            ex = format_example(row)
            if ex is None:
                skipped += 1
                continue
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} examples to {output_path}  (skipped {skipped} empty rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SFT data from Alpaca")
    parser.add_argument(
        "--output",
        default="artifacts/sft/train.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Cap on number of examples (default: all ~52k)",
    )
    args = parser.parse_args()
    prepare_sft_data(args.output, args.max_examples)
