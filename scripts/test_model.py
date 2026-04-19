"""Quick forward-pass smoke test. Config comes from config.py."""
import torch
from edge_cloud_llm.config import MODEL, DATA, build_model
from edge_cloud_llm.data.tokenizer import BPETokenizer


def main():
    tokenizer = BPETokenizer.from_file(DATA.tokenizer_path)
    model = build_model(tokenizer.vocab_size)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params  : {n_params:,}")
    print(f"Architecture  : n_embd={MODEL.n_embd}, n_head={MODEL.n_head}, "
          f"n_layer={MODEL.n_layer}, block_size={MODEL.block_size}")

    batch = torch.randint(0, tokenizer.vocab_size, (2, MODEL.block_size))
    logits, loss, _ = model(batch, targets=batch)

    print(f"Input shape   : {batch.shape}")
    print(f"Logits shape  : {logits.shape}")
    print(f"Loss          : {loss.item():.4f}")
    print("Forward pass OK")


if __name__ == "__main__":
    main()
