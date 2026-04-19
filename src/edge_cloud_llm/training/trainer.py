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
#         self.model.train()
#         total_loss = 0.0
#         total_steps = 0

#         pbar = tqdm(self.train_loader, desc="Training", leave=True)

#         for x, y in pbar:
#             x = x.to(self.device)
#             y = y.to(self.device)

#             self.optimizer.zero_grad()

#             logits, loss, _ = self.model(x, y, use_cache=False)

#             loss.backward()
#             self.optimizer.step()

#             total_loss += loss.item()
#             total_steps += 1

#             avg_loss = total_loss / total_steps
#             pbar.set_postfix(
#                 batch_loss=f"{loss.item():.4f}",
#                 avg_loss=f"{avg_loss:.4f}",
#             )

#         return total_loss / max(total_steps, 1)

#     @torch.no_grad()
#     def evaluate(self, max_batches: int | None = None) -> float:
#         self.model.eval()
#         total_loss = 0.0
#         total_steps = 0

#         pbar = tqdm(self.val_loader, desc="Validation", leave=True)

#         for batch_idx, (x, y) in enumerate(pbar, start=1):
#             if max_batches is not None and batch_idx > max_batches:
#                 break

#             x = x.to(self.device)
#             y = y.to(self.device)

#             logits, loss, _ = self.model(x, y, use_cache=False)

#             total_loss += loss.item()
#             total_steps += 1

#             avg_loss = total_loss / total_steps
#             pbar.set_postfix(
#                 batch_loss=f"{loss.item():.4f}",
#                 avg_loss=f"{avg_loss:.4f}",
#             )

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
import csv
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
        scheduler=None,
        grad_accum_steps: int = 1,
        use_amp: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scheduler = scheduler
        self.grad_accum_steps = grad_accum_steps
        self.use_amp = use_amp and device.startswith("cuda")

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.model.to(self.device)

        self.log_file = self.output_dir / "training_log.csv"
        if not self.log_file.exists():
            with self.log_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate"])

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        total_steps = 0

        self.optimizer.zero_grad()

        # pbar = tqdm(self.train_loader, desc="Training", leave=True)

        import os

        disable_tqdm = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "Batch"

        pbar = tqdm(
            self.train_loader,
            "Training",
            leave=False,
            disable=disable_tqdm
        )

        for step_idx, (x, y) in enumerate(pbar, start=1):
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits, loss, _ = self.model(x, y, use_cache=False)
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if step_idx % self.grad_accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

            true_loss = loss.item() * self.grad_accum_steps
            total_loss += true_loss
            total_steps += 1

            avg_loss = total_loss / total_steps
            current_lr = self.optimizer.param_groups[0]["lr"]

            pbar.set_postfix(
                batch_loss=f"{true_loss:.4f}",
                avg_loss=f"{avg_loss:.4f}",
                lr=f"{current_lr:.6f}",
            )

        # flush leftover gradients if dataset size not divisible by grad_accum_steps
        if total_steps % self.grad_accum_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            if self.scheduler is not None:
                self.scheduler.step()

        return total_loss / max(total_steps, 1)

    @torch.no_grad()
    def evaluate(self, max_batches: int | None = None) -> float:
        self.model.eval()
        total_loss = 0.0
        total_steps = 0

        # pbar = tqdm(self.val_loader, desc="Validation", leave=True)

        import os

        disable_tqdm = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "Batch"

        pbar = tqdm(
            self.val_loader,
            "Validation",
            leave=False,
            disable=disable_tqdm
        )

        for batch_idx, (x, y) in enumerate(pbar, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break

            x = x.to(self.device)
            y = y.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits, loss, _ = self.model(x, y, use_cache=False)

            total_loss += loss.item()
            total_steps += 1

            avg_loss = total_loss / total_steps
            pbar.set_postfix(
                batch_loss=f"{loss.item():.4f}",
                avg_loss=f"{avg_loss:.4f}",
            )

        return total_loss / max(total_steps, 1)

    def save_checkpoint(self, name: str, epoch: int):
        ckpt_path = self.output_dir / name
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            },
            ckpt_path,
        )
        return ckpt_path

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.use_amp and checkpoint.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        return checkpoint.get("epoch", 0)

    def log_metrics(self, epoch: int, train_loss: float, val_loss: float):
        lr = self.optimizer.param_groups[0]["lr"]
        with self.log_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, lr])