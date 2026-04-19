"""Sanity-check the tokenizer. Path comes from config.py."""
from edge_cloud_llm.config import DATA
from edge_cloud_llm.data.tokenizer import BPETokenizer


def main():
    tokenizer = BPETokenizer.from_file(DATA.tokenizer_path)
    text = "Machine learning helps computers learn patterns from data."
    ids  = tokenizer.encode(text, add_special_tokens=True)

    print(f"Vocab size   : {tokenizer.vocab_size}")
    print(f"Original text: {text}")
    print(f"Token ids    : {ids}")
    print(f"Decoded      : {tokenizer.decode(ids)}")


if __name__ == "__main__":
    main()
