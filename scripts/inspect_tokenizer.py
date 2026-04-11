from edge_cloud_llm.data.tokenizer import BPETokenizer


def main():
    tokenizer = BPETokenizer.from_file("artifacts/tokenizer/bpe_tokenizer.json")

    text = "Machine learning helps computers learn patterns from data."
    ids = tokenizer.encode(text, add_special_tokens=True)
    decoded = tokenizer.decode(ids)

    print("Vocab size:", tokenizer.vocab_size)
    print("Original text:")
    print(text)

    print("\nToken ids:")
    print(ids)

    print("\nDecoded text:")
    print(decoded)


if __name__ == "__main__":
    main()