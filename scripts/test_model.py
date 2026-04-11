import torch

from edge_cloud_llm.model import MiniTransformerLM


def main():
    vocab_size = 100
    d_model = 32
    max_seq_len = 16
    num_heads = 4
    num_layers = 2
    ff_hidden_dim = 64

    model = MiniTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_hidden_dim=ff_hidden_dim,
    )

    input_ids = torch.randint(0, vocab_size, (2, 10))
    logits = model(input_ids)

    print("Input shape:", input_ids.shape)
    print("Logits shape:", logits.shape)


if __name__ == "__main__":
    main()