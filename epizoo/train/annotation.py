# epizoo/train/annotation.py

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import amp
from torch.optim.lr_scheduler import LambdaLR

from epizoo.metrics import compute_classification_metrics
from epizoo.models.lora import freeze_module


@dataclass
class AnnotationTrainConfig:
    output_dir: str = "checkpoints_annotation"
    log_file: str = "training_log.csv"

    max_steps: int = 100_000
    save_steps: int = 20_000
    log_steps: int = 500
    eval_steps: int = 5_000
    keep_last: int = 5

    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 10_000
    epoch_decay: float = 0.9

    use_amp: bool = True
    grad_clip: Optional[float] = None
    device: Optional[str] = None

    freeze_seq_emb: bool = True


class EpiZooAnnotationTrainer:
    """
    Trainer for EpiZoo cell type annotation.

    Expected batch format:
        input_ids, labels

    Model requirement:
        outputs = model(input_ids)
        logits = outputs["logits"]
        loss = model.compute_loss(logits, labels)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader=None,
        cfg: Optional[AnnotationTrainConfig] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg or AnnotationTrainConfig()

        self.device = torch.device(
            self.cfg.device
            if self.cfg.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.recent_ckpts = []

        self._setup_dirs()
        self._setup_trainable_modules()

        self.model.to(self.device)

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = amp.GradScaler(
            "cuda",
            enabled=self.cfg.use_amp and self.device.type == "cuda",
        )

        self._init_log_file()

    def train(self):
        self.model.train()

        while self.global_step < self.cfg.max_steps:
            epoch = self.global_step // max(1, len(self.train_loader))
            start_time = time.time()

            running = self._empty_running()

            for batch in self.train_loader:
                if self.global_step >= self.cfg.max_steps:
                    break

                step_out = self.train_step(batch)
                self.global_step += 1

                self._update_running(running, step_out)

                if self.global_step % self.cfg.log_steps == 0:
                    self._log_train(running)
                    running = self._empty_running()

                if self.val_loader is not None and self.global_step % self.cfg.eval_steps == 0:
                    self.evaluate_and_save_best()

                if self.global_step % self.cfg.save_steps == 0:
                    self.save_checkpoint()

            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1} finished. Time: {epoch_time:.2f}s")

        print("Annotation training finished.")
        return self.model

    def train_step(self, batch) -> Dict:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        input_ids, labels = self._move_batch(batch)

        with amp.autocast(
            device_type=self.device.type,
            enabled=self.cfg.use_amp and self.device.type == "cuda",
        ):
            outputs = self.model(
                input_ids=input_ids,
                return_cell_emb=False,
                return_transformer_out=False,
            )

            logits = outputs["logits"]

            loss = self.model.compute_loss(
                logits=logits,
                labels=labels,
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

        preds = logits.argmax(dim=1)

        return {
            "loss": float(loss.detach().cpu()),
            "labels": labels.detach().cpu().numpy().tolist(),
            "preds": preds.detach().cpu().numpy().tolist(),
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if self.val_loader is None:
            raise ValueError("`val_loader` is None.")

        self.model.eval()

        total_loss = 0.0
        num_steps = 0
        labels_all = []
        preds_all = []

        for batch in self.val_loader:
            input_ids, labels = self._move_batch(batch)

            with amp.autocast(
                device_type=self.device.type,
                enabled=self.cfg.use_amp and self.device.type == "cuda",
            ):
                outputs = self.model(
                    input_ids=input_ids,
                    return_cell_emb=False,
                    return_transformer_out=False,
                )

                logits = outputs["logits"]

                loss = self.model.compute_loss(
                    logits=logits,
                    labels=labels,
                )

            preds = logits.argmax(dim=1)

            total_loss += float(loss.detach().cpu())
            num_steps += 1

            labels_all.extend(labels.detach().cpu().numpy().tolist())
            preds_all.extend(preds.detach().cpu().numpy().tolist())

        metrics = compute_classification_metrics(
            labels=labels_all,
            preds=preds_all,
        )

        self.model.train()

        return {
            "val_loss": total_loss / max(1, num_steps),
            "val_acc": metrics["acc"],
            "val_kappa": metrics["kappa"],
            "val_macro_f1": metrics["macro_f1"],
        }

    def evaluate_and_save_best(self):
        metrics = self.evaluate()
        val_loss = metrics["val_loss"]

        print(
            f"Eval step {self.global_step} | "
            f"val_loss={val_loss:.4f} | "
            f"acc={metrics['val_acc']:.4f} | "
            f"kappa={metrics['val_kappa']:.4f} | "
            f"macro_f1={metrics['val_macro_f1']:.4f}"
        )

        self._append_log_row(
            split="val",
            loss=metrics["val_loss"],
            acc=metrics["val_acc"],
            kappa=metrics["val_kappa"],
            macro_f1=metrics["val_macro_f1"],
        )

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint(
                name="best_model.pth",
                keep_recent=False,
            )
            print(f"New best model saved. val_loss={val_loss:.4f}")

    def save_checkpoint(
        self,
        name: Optional[str] = None,
        keep_recent: bool = True,
    ):
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            name = f"{timestamp}_{self.global_step}.pth"

        path = os.path.join(self.cfg.output_dir, name)
        torch.save(self.model.state_dict(), path)

        if keep_recent:
            self.recent_ckpts.append(path)

            while len(self.recent_ckpts) > self.cfg.keep_last:
                old_path = self.recent_ckpts.pop(0)

                if os.path.exists(old_path):
                    os.remove(old_path)

        print(f"Checkpoint saved to {path}")

    def _move_batch(self, batch):
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            labels = batch["labels"]
        else:
            input_ids, labels = batch

        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device).long()

        return input_ids, labels

    def _build_optimizer(self):
        params = [
            param
            for param in self.model.parameters()
            if param.requires_grad
        ]

        return torch.optim.AdamW(
            params,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

    def _build_scheduler(self):
        steps_per_epoch = max(1, len(self.train_loader))

        def lr_lambda(step: int):
            if step < self.cfg.warmup_steps:
                return step / max(1, self.cfg.warmup_steps)

            epoch = step // steps_per_epoch
            return self.cfg.epoch_decay ** epoch

        return LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambda,
        )

    def _setup_dirs(self):
        os.makedirs(self.cfg.output_dir, exist_ok=True)

    def _setup_trainable_modules(self):
        """
        Freeze modules according to training config.
        """

        if self.cfg.freeze_seq_emb and hasattr(self.model, "seq_emb"):
            freeze_module(self.model.seq_emb)
            print("Frozen seq_emb.")

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
                    "global_step",
                    "split",
                    "lr",
                    "loss",
                    "acc",
                    "kappa",
                    "macro_f1",
                    "best_val_loss",
                ]
            )

    def _empty_running(self):
        return {
            "steps": 0,
            "loss": 0.0,
            "labels": [],
            "preds": [],
        }

    def _update_running(self, running: Dict, step_out: Dict):
        running["steps"] += 1
        running["loss"] += step_out["loss"]
        running["labels"].extend(step_out["labels"])
        running["preds"].extend(step_out["preds"])

    def _log_train(self, running: Dict):
        if running["steps"] == 0:
            return

        loss = running["loss"] / running["steps"]

        metrics = compute_classification_metrics(
            labels=running["labels"],
            preds=running["preds"],
        )

        print(
            f"Step {self.global_step} | "
            f"lr={self.scheduler.get_last_lr()[0]:.3e} | "
            f"loss={loss:.4f} | "
            f"acc={metrics['acc']:.4f} | "
            f"kappa={metrics['kappa']:.4f} | "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

        self._append_log_row(
            split="train",
            loss=loss,
            acc=metrics["acc"],
            kappa=metrics["kappa"],
            macro_f1=metrics["macro_f1"],
        )

    def _append_log_row(
        self,
        split: str,
        loss: float,
        acc: float,
        kappa: float,
        macro_f1: float,
    ):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.global_step,
                    split,
                    self.scheduler.get_last_lr()[0],
                    loss,
                    acc,
                    kappa,
                    macro_f1,
                    self.best_val_loss,
                ]
            )
