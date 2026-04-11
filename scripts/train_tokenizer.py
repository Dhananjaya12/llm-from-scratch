from pathlib import Path

from edge_cloud_llm.data.dataset import load_wikitext_split_rows
from edge_cloud_llm.data.tokenizer import BPETokenizer


def main():
    output_path = Path("artifacts/tokenizer/bpe_tokenizer.json")

    train_rows = load_wikitext_split_rows("train")

    tokenizer = BPETokenizer.train(
        texts=train_rows,
        vocab_size=8000,
        min_frequency=2,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
    )

    tokenizer.save(output_path)

    print("Tokenizer trained and saved.")
    print("Path:", output_path)
    print("Vocab size:", tokenizer.vocab_size)
    print("pad id:", tokenizer.pad_token_id)
    print("bos id:", tokenizer.bos_token_id)
    print("eos id:", tokenizer.eos_token_id)
    print("unk id:", tokenizer.unk_token_id)


if __name__ == "__main__":
    main()