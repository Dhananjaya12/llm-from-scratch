# from __future__ import annotations

# import torch

# from edge_cloud_llm.data import create_dataloaders
# from edge_cloud_llm.training import Trainer
# from edge_cloud_llm.model import GPT


# def main():
#     tokenizer_path = "artifacts/tokenizer/bpe_tokenizer.json"

#     block_size = 128
#     batch_size = 16
#     n_layer = 4
#     n_head = 4
#     n_embd = 256
#     dropout = 0.1
#     learning_rate = 3e-4
#     epochs = 5
#     eval_batches = 50
#     output_dir = "outputs/base"

#     # block_size = 32
#     # batch_size = 4
#     # n_layer = 2
#     # n_head = 2
#     # n_embd = 64
#     # dropout = 0.1
#     # learning_rate = 3e-4
#     # epochs = 1
#     # eval_batches = 5
#     # output_dir = "outputs/base"

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print("Using device:", device)

#     train_loader, val_loader, tokenizer = create_dataloaders(
#         tokenizer_path=tokenizer_path,
#         block_size=block_size,
#         batch_size=batch_size,
#     )

#     print("Tokenizer vocab size:", tokenizer.vocab_size)
#     print("Train batches:", len(train_loader))
#     print("Val batches:", len(val_loader))

#     model = GPT(
#         vocab_size=tokenizer.vocab_size,
#         block_size=block_size,
#         n_layer=n_layer,
#         n_head=n_head,
#         n_embd=n_embd,
#         dropout=dropout,
#     )

#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#     trainer = Trainer(
#         model=model,
#         optimizer=optimizer,
#         device=device,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         output_dir=output_dir,
#     )

#     best_val_loss = float("inf")

#     for epoch in range(1, epochs + 1):
#         train_loss = trainer.train_epoch()
#         val_loss = trainer.evaluate(max_batches=eval_batches)

#         print(
#             f"Epoch {epoch}/{epochs} | "
#             f"train_loss={train_loss:.4f} | "
#             f"val_loss={val_loss:.4f}"
#         )

#         trainer.save_checkpoint(f"epoch_{epoch}.pt")

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             trainer.save_checkpoint("best.pt")
#             print("Saved new best checkpoint.")

#     print("Training finished.")
#     print("Best validation loss:", best_val_loss)


# if __name__ == "__main__":
#     main()



from __future__ import annotations

import torch

from edge_cloud_llm.data import create_dataloaders
from edge_cloud_llm.training import Trainer
from edge_cloud_llm.model import GPT


def main():
    tokenizer_path = "artifacts/tokenizer/bpe_tokenizer.json"

    # local-friendly config
    block_size = 32
    batch_size = 4

    n_layer = 2
    n_head = 2
    n_embd = 64
    dropout = 0.1

    # new Part 3 config
    rope_base = 10000.0
    window_size = None              # keep None for normal training
    attention_sink_tokens = 0       # keep 0 for now

    learning_rate = 3e-4
    epochs = 5
    eval_batches = 20
    output_dir = "outputs/base"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_loader, val_loader, tokenizer = create_dataloaders(
        tokenizer_path=tokenizer_path,
        block_size=block_size,
        batch_size=batch_size,
    )

    print("Tokenizer vocab size:", tokenizer.vocab_size)
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))

    model = GPT(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        rope_base=rope_base,
        window_size=window_size,
        attention_sink_tokens=attention_sink_tokens,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
    )

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        print(f"\nStarting epoch {epoch}/{epochs}")

        train_loss = trainer.train_epoch()
        val_loss = trainer.evaluate(max_batches=eval_batches)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )

        trainer.save_checkpoint(f"epoch_{epoch}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint("best.pt")
            print("Saved new best checkpoint.")

    print("Training finished.")
    print("Best validation loss:", best_val_loss)


if __name__ == "__main__":
    main()