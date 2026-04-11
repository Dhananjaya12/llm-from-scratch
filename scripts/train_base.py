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
#     epochs = 30
#     eval_batches = 50
#     output_dir = "outputs/base"
#     rope_base = 10000.0
#     window_size = None              # keep None for normal training
#     attention_sink_tokens = 0       # keep 0 for now

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
#         rope_base=rope_base,
#         window_size=window_size,
#         attention_sink_tokens=attention_sink_tokens,
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
#         print(f"\nStarting epoch {epoch}/{epochs}")

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

import math
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

from edge_cloud_llm.data import create_dataloaders
from edge_cloud_llm.training import Trainer
from edge_cloud_llm.model import GPT


def build_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def main():
    tokenizer_path = "artifacts/tokenizer/bpe_tokenizer.json"

    # -----------------------------
    # model config
    # -----------------------------
    block_size = 64
    batch_size = 8
    n_layer = 4
    n_head = 4
    n_embd = 128
    dropout = 0.1

    rope_base = 10000.0
    window_size = None
    attention_sink_tokens = 0

    # -----------------------------
    # training config
    # -----------------------------
    learning_rate = 3e-4
    epochs = 3
    eval_batches = 50
    output_dir = "outputs/base"

    grad_accum_steps = 4
    use_amp = True
    warmup_steps = 100
    resume_from = None  # example: "outputs/base/best.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # -----------------------------
    # data
    # -----------------------------
    train_loader, val_loader, tokenizer = create_dataloaders(
        tokenizer_path=tokenizer_path,
        block_size=block_size,
        batch_size=batch_size,
    )

    print("Tokenizer vocab size:", tokenizer.vocab_size)
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))

    # -----------------------------
    # model
    # -----------------------------
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

    total_optimizer_steps = (len(train_loader) * epochs) // grad_accum_steps
    scheduler = build_lr_scheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=max(1, total_optimizer_steps),
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        scheduler=scheduler,
        grad_accum_steps=grad_accum_steps,
        use_amp=use_amp,
    )

    start_epoch = 1
    if resume_from is not None and Path(resume_from).exists():
        last_epoch = trainer.load_checkpoint(resume_from)
        start_epoch = last_epoch + 1
        print(f"Resumed from checkpoint: {resume_from}")
        print(f"Starting at epoch {start_epoch}")

    best_val_loss = float("inf")

    for epoch in range(start_epoch, epochs + 1):
        print(f"\nStarting epoch {epoch}/{epochs}")

        train_loss = trainer.train_epoch()
        val_loss = trainer.evaluate(max_batches=eval_batches)

        trainer.log_metrics(epoch, train_loss, val_loss)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        trainer.save_checkpoint(f"epoch_{epoch}.pt", epoch=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint("best.pt", epoch=epoch)
            print("Saved new best checkpoint.")

    print("Training finished.")
    print("Best validation loss:", best_val_loss)
    print(f"Training log saved to: {output_dir}/training_log.csv")


if __name__ == "__main__":
    main()