# epizoo/train/seq.py

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import amp
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from epizoo.inference.utils import get_device
from epizoo.train.loss import CosineMSELogLoss


@dataclass
class EpiZooSeqTrainConfig:
    output_dir: str = "checkpoints_epizoo_seq"
    log_file: str = "training_log.csv"

    epochs: int = 20
    warmup_epochs: int = 1

    lr_head_warmup: float = 1e-4
    lr_head: float = 1e-4
    lr_backbone: float = 1e-4
    weight_decay: float = 0.01

    logging_steps: int = 200
    save_epochs: int = 1
    keep_last: int = 2

    warmup_ratio_phase1: float = 0.2
    warmup_ratio_phase2: float = 0.5

    use_amp: bool = True
    grad_clip: Optional[float] = None
    device: Optional[str] = None

    loss_max_weight: float = 1.0
    loss_multiplier: float = 1000.0


class EpiZooSeqTrainer:
    """
    Trainer for EpiZooSeq.

    Expected batch from SEAMDataset + collate_fn_seam:
        {
            "input_ids": LongTensor [batch_size, seq_len],
            "attention_mask": LongTensor [batch_size, seq_len],
            "signal": FloatTensor [batch_size, num_cell_types],
        }

    Training strategy:
        Phase 1:
            freeze SEAM backbone and train only prediction head.

        Phase 2:
            unfreeze SEAM backbone and train backbone/head with separate LRs.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        cfg: Optional[EpiZooSeqTrainConfig] = None,
        criterion: Optional[nn.Module] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.cfg = cfg or EpiZooSeqTrainConfig()

        self.device = get_device(self.cfg.device)
        self.global_step = 0
        self.recent_ckpts = []

        self._setup_dirs()
        self._init_log_file()

        self.model.to(self.device)

        self.criterion = criterion or CosineMSELogLoss(
            max_weight=self.cfg.loss_max_weight,
            multiplier=self.cfg.loss_multiplier,
        )
        self.criterion.to(self.device)

        self.optimizer = None
        self.scheduler = None

        self.scaler = amp.GradScaler(
            "cuda",
            enabled=self.cfg.use_amp and self.device.type == "cuda",
        )

    def train(self):
        for epoch in range(self.cfg.epochs):
            self._setup_phase(epoch)

            out = self._train_epoch(epoch)
            self._log_epoch(epoch, out)

            if (epoch + 1) % self.cfg.save_epochs == 0:
                self.save_checkpoint(
                    name=f"epoch_{epoch + 1}.pth",
                )

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        print("EpiZooSeq training finished.")
        return self.model

    def _train_epoch(
        self,
        epoch: int,
    ) -> Dict[str, float]:
        self.model.train()

        total_loss = 0.0
        num_steps = 0

        iterator = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.cfg.epochs}",
        )

        for batch in iterator:
            step_out = self.train_step(batch)

            self.global_step += 1
            num_steps += 1
            total_loss += step_out["loss"]

            iterator.set_postfix(
                {
                    "loss": f"{step_out['loss']:.4f}",
                    "lr": f"{self._current_lr():.2e}",
                }
            )

            if (
                self.cfg.logging_steps > 0
                and self.global_step % self.cfg.logging_steps == 0
            ):
                self._log_step(
                    epoch=epoch,
                    loss=total_loss / max(1, num_steps),
                )

        return {
            "loss": total_loss / max(1, num_steps),
        }

    def train_step(
        self,
        batch,
    ) -> Dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)

        batch = self._move_batch(batch)

        with amp.autocast(
            device_type=self.device.type,
            enabled=self.cfg.use_amp and self.device.type == "cuda",
        ):
            pred = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            loss = self.criterion(
                pred,
                batch["signal"],
            )

        self.scaler.scale(loss).backward()

        if self.cfg.grad_clip is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.grad_clip,
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return {
            "loss": float(loss.detach().cpu()),
        }

    def _setup_phase(
        self,
        epoch: int,
    ) -> None:
        if epoch == 0 and self.cfg.warmup_epochs > 0:
            self._freeze_backbone()

            total_steps = self.cfg.warmup_epochs * len(self.train_loader)
            warmup_steps = int(total_steps * self.cfg.warmup_ratio_phase1)

            self.optimizer = self._build_warmup_optimizer()
            self.scheduler = self._build_scheduler(
                total_steps=total_steps,
                warmup_steps=warmup_steps,
            )

            print(
                f"Epoch {epoch + 1}: phase 1. "
                "Frozen SEAM backbone."
            )
            return

        if epoch == self.cfg.warmup_epochs:
            self._unfreeze_backbone()

            total_steps = (
                self.cfg.epochs - self.cfg.warmup_epochs
            ) * len(self.train_loader)

            warmup_steps = int(total_steps * self.cfg.warmup_ratio_phase2)

            self.optimizer = self._build_finetune_optimizer()
            self.scheduler = self._build_scheduler(
                total_steps=total_steps,
                warmup_steps=warmup_steps,
            )

            print(
                f"Epoch {epoch + 1}: phase 2. "
                "Unfrozen SEAM backbone."
            )
            return

        if self.optimizer is None or self.scheduler is None:
            self._unfreeze_backbone()

            total_steps = self.cfg.epochs * len(self.train_loader)
            warmup_steps = int(total_steps * self.cfg.warmup_ratio_phase2)

            self.optimizer = self._build_finetune_optimizer()
            self.scheduler = self._build_scheduler(
                total_steps=total_steps,
                warmup_steps=warmup_steps,
            )

    def _build_warmup_optimizer(self):
        return torch.optim.AdamW(
            [
                {
                    "params": self._head_params(),
                    "lr": self.cfg.lr_head_warmup,
                }
            ],
            weight_decay=self.cfg.weight_decay,
        )

    def _build_finetune_optimizer(self):
        return torch.optim.AdamW(
            [
                {
                    "params": self._head_params(),
                    "lr": self.cfg.lr_head,
                },
                {
                    "params": self._backbone_params(),
                    "lr": self.cfg.lr_backbone,
                },
            ],
            weight_decay=self.cfg.weight_decay,
        )

    def _build_scheduler(
        self,
        total_steps: int,
        warmup_steps: int,
    ):
        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=max(1, warmup_steps),
            num_training_steps=max(1, total_steps),
        )

    def _head_params(self):
        return [
            p
            for p in self.model.head.parameters()
            if p.requires_grad
        ]

    def _backbone_params(self):
        return [
            p
            for name, p in self.model.named_parameters()
            if not name.startswith("head.")
            and p.requires_grad
        ]

    def _freeze_backbone(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = name.startswith("head.")

    def _unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def _move_batch(self, batch):
        return {
            "input_ids": batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device),
            "signal": batch["signal"].to(self.device).float(),
        }

    def save_checkpoint(
        self,
        name: Optional[str] = None,
    ) -> str:
        if name is None:
            name = f"step_{self.global_step}.pth"

        path = os.path.join(
            self.cfg.output_dir,
            name,
        )

        torch.save(
            self.model.state_dict(),
            path,
        )

        self.recent_ckpts.append(path)

        while len(self.recent_ckpts) > self.cfg.keep_last:
            old_path = self.recent_ckpts.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)

        print(f"Checkpoint saved to {path}")
        return path

    def _setup_dirs(self):
        os.makedirs(
            self.cfg.output_dir,
            exist_ok=True,
        )

    def _init_log_file(self):
        self.log_path = os.path.join(
            self.cfg.output_dir,
            self.cfg.log_file,
        )

        if os.path.exists(self.log_path):
            return

        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "time",
                    "global_step",
                    "epoch",
                    "lr",
                    "train_loss",
                ]
            )

    def _log_step(
        self,
        epoch: int,
        loss: float,
    ):
        print(
            f"Epoch {epoch + 1} | "
            f"Step {self.global_step} | "
            f"loss={loss:.4f} | "
            f"lr={self._current_lr():.3e}"
        )

        self._write_log(
            epoch=epoch,
            train_loss=loss,
        )

    def _log_epoch(
        self,
        epoch: int,
        out: Dict[str, float],
    ):
        print(
            f"Epoch {epoch + 1} finished | "
            f"train_loss={out['loss']:.4f}"
        )

        self._write_log(
            epoch=epoch,
            train_loss=out["loss"],
        )

    def _write_log(
        self,
        epoch: int,
        train_loss: float,
    ):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.global_step,
                    epoch + 1,
                    self._current_lr(),
                    train_loss,
                ]
            )

    def _current_lr(self) -> float:
        if self.optimizer is None:
            return 0.0

        return self.optimizer.param_groups[0]["lr"]