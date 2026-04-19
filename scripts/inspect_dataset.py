"""Sanity-check the dataloader. Config comes from config.py."""
from edge_cloud_llm.config import MODEL, DATA, BASE
from edge_cloud_llm.data import create_dataloaders


def main():
    train_loader, val_loader, tokenizer = create_dataloaders(
        tokenizer_path = DATA.tokenizer_path,
        block_size     = MODEL.block_size,
        batch_size     = BASE.batch_size,
        train_chars    = DATA.train_chars,
        val_chars      = DATA.val_chars,
    )

    print(f"Vocab size    : {tokenizer.vocab_size}")
    print(f"block_size    : {MODEL.block_size}")
    print(f"train_chars   : {DATA.train_chars}")
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")

    x, y = next(iter(train_loader))
    print(f"\nInput  shape : {x.shape}")
    print(f"Target shape : {y.shape}")
    print(f"\nFirst 30 input ids : {x[0][:30].tolist()}")
    print(f"Decoded        : {tokenizer.decode(x[0].tolist())}")


if __name__ == "__main__":
    main()
