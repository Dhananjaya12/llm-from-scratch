from edge_cloud_llm.data import create_dataloaders


def main():
    block_size = 64
    batch_size = 4
    tokenizer_path = "artifacts/tokenizer/bpe_tokenizer.json"

    train_loader, val_loader, tokenizer = create_dataloaders(
        tokenizer_path=tokenizer_path,
        block_size=block_size,
        batch_size=batch_size,
    )

    print("Tokenizer vocab size:", tokenizer.vocab_size)
    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    x, y = next(iter(train_loader))

    print("\nInput batch shape:", x.shape)
    print("Target batch shape:", y.shape)

    print("\nFirst sample input ids:")
    print(x[0][:30].tolist())

    print("\nFirst sample target ids:")
    print(y[0][:30].tolist())

    print("\nDecoded input sample:")
    print(tokenizer.decode(x[0].tolist()))

    print("\nDecoded target sample:")
    print(tokenizer.decode(y[0].tolist()))


if __name__ == "__main__":
    main()