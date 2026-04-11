# from __future__ import annotations

# from pathlib import Path
# import torch
# from tqdm import tqdm

# class Trainer:
#     def __init__(
#         self,
#         model,
#         optimizer,
#         device: str,
#         train_loader,
#         val_loader,
#         output_dir: str = "outputs/base",
#     ):
#         self.model = model
#         self.optimizer = optimizer
#         self.device = device
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)

#         self.model.to(self.device)

#     def train_epoch(self) -> float:
#         # self.model.train()
#         # total_loss = 0.0
#         # total_steps = 0

#         # for x, y in self.train_loader:
#         self.model.train()
#         total_loss = 0.0
#         total_steps = 0

#         pbar = tqdm(self.train_loader, desc="Training", leave=True)

#         for x, y in pbar:
#             x = x.to(self.device)
#             y = y.to(self.device)

#             self.optimizer.zero_grad()
#             _, loss = self.model(x, y)
#             loss.backward()
#             self.optimizer.step()

#             total_loss += loss.item()
#             total_steps += 1

#         return total_loss / max(total_steps, 1)

#     @torch.no_grad()
#     def evaluate(self, max_batches: int | None = None) -> float:
#         self.model.eval()
#         total_loss = 0.0
#         total_steps = 0

#         for batch_idx, (x, y) in enumerate(self.val_loader):
#             if max_batches is not None and batch_idx >= max_batches:
#                 break

#             x = x.to(self.device)
#             y = y.to(self.device)

#             _, loss = self.model(x, y)

#             total_loss += loss.item()
#             total_steps += 1

#         return total_loss / max(total_steps, 1)

#     def save_checkpoint(self, name: str):
#         ckpt_path = self.output_dir / name
#         torch.save(
#             {
#                 "model_state_dict": self.model.state_dict(),
#                 "optimizer_state_dict": self.optimizer.state_dict(),
#             },
#             ckpt_path,
#         )
#         return ckpt_path


from __future__ import annotations

from pathlib import Path
import torch
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        device: str,
        train_loader,
        val_loader,
        output_dir: str = "outputs/base",
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        total_steps = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=True)

        for x, y in pbar:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            logits, loss, _ = self.model(x, y, use_cache=False)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_steps += 1

            avg_loss = total_loss / total_steps
            pbar.set_postfix(
                batch_loss=f"{loss.item():.4f}",
                avg_loss=f"{avg_loss:.4f}",
            )

        return total_loss / max(total_steps, 1)

    @torch.no_grad()
    def evaluate(self, max_batches: int | None = None) -> float:
        self.model.eval()
        total_loss = 0.0
        total_steps = 0

        pbar = tqdm(self.val_loader, desc="Validation", leave=True)

        for batch_idx, (x, y) in enumerate(pbar, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break

            x = x.to(self.device)
            y = y.to(self.device)

            logits, loss, _ = self.model(x, y, use_cache=False)

            total_loss += loss.item()
            total_steps += 1

            avg_loss = total_loss / total_steps
            pbar.set_postfix(
                batch_loss=f"{loss.item():.4f}",
                avg_loss=f"{avg_loss:.4f}",
            )

        return total_loss / max(total_steps, 1)

    def save_checkpoint(self, name: str):
        ckpt_path = self.output_dir / name
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            ckpt_path,
        )
        return ckpt_path