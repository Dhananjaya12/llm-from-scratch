from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


class BPETokenizer:
    """
    Byte-level BPE tokenizer wrapper.
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 8000,
        min_frequency: int = 2,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        if special_tokens is None:
            special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        tokenizer.train_from_iterator(texts, trainer=trainer)

        bos_id = tokenizer.token_to_id("<bos>")
        eos_id = tokenizer.token_to_id("<eos>")

        if bos_id is not None and eos_id is not None:
            tokenizer.post_processor = processors.TemplateProcessing(
                single="<bos> $A <eos>",
                pair="<bos> $A <eos> $B:1 <eos>:1",
                special_tokens=[
                    ("<bos>", bos_id),
                    ("<eos>", eos_id),
                ],
            )

        return cls(tokenizer)

    @classmethod
    def from_file(cls, tokenizer_path: str | Path) -> "BPETokenizer":
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return cls(tokenizer)

    def save(self, tokenizer_path: str | Path) -> None:
        tokenizer_path = Path(tokenizer_path)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(tokenizer_path))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> int | None:
        return self.tokenizer.token_to_id(token)

    @property
    def pad_token_id(self) -> int:
        token_id = self.token_to_id("<pad>")
        if token_id is None:
            raise ValueError("<pad> token not found in tokenizer.")
        return token_id

    @property
    def bos_token_id(self) -> int:
        token_id = self.token_to_id("<bos>")
        if token_id is None:
            raise ValueError("<bos> token not found in tokenizer.")
        return token_id

    @property
    def eos_token_id(self) -> int:
        token_id = self.token_to_id("<eos>")
        if token_id is None:
            raise ValueError("<eos> token not found in tokenizer.")
        return token_id

    @property
    def unk_token_id(self) -> int:
        token_id = self.token_to_id("<unk>")
        if token_id is None:
            raise ValueError("<unk> token not found in tokenizer.")
        return token_id

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        enc = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return enc.ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            specials = {
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
                self.unk_token_id,
            }
            token_ids = [tid for tid in token_ids if tid not in specials]
        return self.tokenizer.decode(token_ids)

    def encode_batch(self, texts: list[str], add_special_tokens: bool = True) -> list[list[int]]:
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        return [enc.ids for enc in encodings]