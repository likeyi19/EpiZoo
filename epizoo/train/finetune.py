# epizoo/train/finetune.py

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
from torch import amp
from torch.optim.lr_scheduler import LambdaLR

from epizoo.metrics import compute_cca_metrics
from epizoo.models.lora import (
    apply_lora_to_decoder,
    apply_lora_to_embedding,
    apply_lora_to_transformer,
    count_parameters,
    freeze_module,
)


TaskMode = Literal["sr", "sr_cca"]


@dataclass
class LoRAConfig:
    use_embedding: bool = False
    use_transformer: bool = False
    use_decoder: bool = False

    r: int = 8
    alpha: int = 32
    dropout: float = 0.0


@dataclass
class FineTuneConfig:
    mode: TaskMode = "sr"

    output_dir: str = "checkpoints"
    log_file: str = "training_log.csv"

    max_steps: int = 500_000
    save_steps: int = 10_000
    log_steps: int = 500
    keep_last: int = 5

    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1_000
    epoch_decay: float = 0.9

    sr_weight: float = 1.0
    cca_weight: float = 1.0

    use_amp: bool = True
    grad_clip: Optional[float] = None
    device: Optional[str] = None

    max_cca_metric_samples: int = 10_000

    freeze_seq_emb: bool = True


class EpiZooFinetuneTrainer:
    """
    Fine-tuning trainer for EpiZoo / EpiZooDI.

    Supported modes:
        "sr"      : signal reconstruction only
        "sr_cca"  : signal reconstruction + CCA

    Expected batch format:
        (
            input_ids,
            signals_human,
            signals_mouse,
            cca_ids,
            cca_labels,
            species,
        )
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        cfg: Optional[FineTuneConfig] = None,
        lora_cfg: Optional[LoRAConfig] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.cfg = cfg or FineTuneConfig()
        self.lora_cfg = lora_cfg or LoRAConfig()

        self.device = torch.device(
            self.cfg.device
            if self.cfg.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.global_step = 0
        self.recent_ckpts = []

        self._check_mode()
        self._setup_dirs()
        self._setup_trainable_modules()
        self._setup_lora()

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
                    self._log_step(running)
                    running = self._empty_running()

                if self.global_step % self.cfg.save_steps == 0:
                    self.save_checkpoint()

            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1} finished. Time: {epoch_time:.2f}s")

        print("Fine-tuning finished.")
        return self.model

    def train_step(self, batch) -> Dict:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        batch = self._move_batch(batch)

        with amp.autocast(
            device_type=self.device.type,
            enabled=self.cfg.use_amp and self.device.type == "cuda",
        ):
            outputs = self._forward_model(batch)
            cell_emb = outputs["cell_emb"]

            sr_out = self.model.compute_signal_loss(
                cell_emb=cell_emb,
                input_species=batch["species"],
                signals={
                    "human": batch["signals_human"],
                    "mouse": batch["signals_mouse"],
                },
            )
            sr_loss = sr_out["loss"]

            cca_loss = None
            cca_logits = None

            if self.cfg.mode == "sr_cca":
                cca_out = self.model.compute_cca_loss(
                    cell_emb=cell_emb,
                    ccre_ids=batch["cca_ids"],
                    accessibility=batch["cca_labels"],
                )
                cca_loss = cca_out["loss"]
                cca_logits = cca_out["logits"]

            loss = self._combine_loss(
                sr_loss=sr_loss,
                cca_loss=cca_loss,
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

        out = {
            "loss": float(loss.detach().cpu()),
            "sr_loss": float(sr_loss.detach().cpu()),
            "cca_loss": float(cca_loss.detach().cpu()) if cca_loss is not None else 0.0,
            "cca_labels": [],
            "cca_logits": [],
        }

        if cca_logits is not None:
            labels = batch["cca_labels"].detach().cpu().numpy().reshape(-1).tolist()
            logits = cca_logits.detach().cpu().numpy().reshape(-1).tolist()

            max_keep = self.cfg.max_cca_metric_samples
            out["cca_labels"] = labels[:max_keep]
            out["cca_logits"] = logits[:max_keep]

        return out
    
    def _forward_model(self, batch):
        """
        Forward hook.

        Subclasses can override this method to pass task-specific inputs.
        """

        return self.model(
            input_ids=batch["input_ids"],
            return_transformer_out=False,
        )

    def _combine_loss(
        self,
        sr_loss: torch.Tensor,
        cca_loss: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.cfg.mode == "sr":
            return self.cfg.sr_weight * sr_loss

        if self.cfg.mode == "sr_cca":
            if cca_loss is None:
                raise ValueError("`cca_loss` is required when mode='sr_cca'.")

            return self.cfg.sr_weight * sr_loss + self.cfg.cca_weight * cca_loss

        raise ValueError(f"Unknown mode: {self.cfg.mode}")

    def _move_batch(self, batch) -> Dict:
        (
            input_ids,
            signals_human,
            signals_mouse,
            cca_ids,
            cca_labels,
            species,
        ) = batch

        if signals_human is not None:
            signals_human = signals_human.to(self.device)

        if signals_mouse is not None:
            signals_mouse = signals_mouse.to(self.device)

        return {
            "input_ids": input_ids.to(self.device),
            "signals_human": signals_human,
            "signals_mouse": signals_mouse,
            "cca_ids": [x.to(self.device) for x in cca_ids],
            "cca_labels": cca_labels.to(self.device),
            "species": list(species),
        }

    def _setup_lora(self):
        if self.lora_cfg.use_embedding:
            self.model = apply_lora_to_embedding(
                self.model,
                r=self.lora_cfg.r,
                alpha=self.lora_cfg.alpha,
                dropout=self.lora_cfg.dropout,
            )

        if self.lora_cfg.use_transformer:
            self.model.encoder = apply_lora_to_transformer(
                self.model.encoder,
                r=self.lora_cfg.r,
                alpha=self.lora_cfg.alpha,
                dropout=self.lora_cfg.dropout,
            )

        if self.lora_cfg.use_decoder:
            self.model.signal_decoder = apply_lora_to_decoder(
                self.model.signal_decoder,
                r=self.lora_cfg.r,
                alpha=self.lora_cfg.alpha,
                dropout=self.lora_cfg.dropout,
            )

        count_parameters(self.model)

    def _setup_trainable_modules(self):
        """
        Freeze modules according to training config.
        """

        if self.cfg.freeze_seq_emb and hasattr(self.model, "seq_emb"):
            freeze_module(self.model.seq_emb)
            print("Frozen seq_emb.")

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

        return LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def save_checkpoint(self, name: Optional[str] = None):
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            name = f"{timestamp}_{self.global_step}.pth"

        path = os.path.join(self.cfg.output_dir, name)
        torch.save(self.model.state_dict(), path)

        self.recent_ckpts.append(path)

        while len(self.recent_ckpts) > self.cfg.keep_last:
            old_path = self.recent_ckpts.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)

        print(f"Checkpoint saved to {path}")

    def _setup_dirs(self):
        os.makedirs(self.cfg.output_dir, exist_ok=True)

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
                    "lr",
                    "loss",
                    "sr_loss",
                    "cca_loss",
                    "cca_pos_acc",
                    "cca_neg_acc",
                    "cca_auroc",
                    "cca_auprc",
                ]
            )

    def _empty_running(self):
        return {
            "steps": 0,
            "loss": 0.0,
            "sr_loss": 0.0,
            "cca_loss": 0.0,
            "cca_labels": [],
            "cca_logits": [],
        }

    def _update_running(self, running: Dict, step_out: Dict):
        running["steps"] += 1
        running["loss"] += step_out["loss"]
        running["sr_loss"] += step_out["sr_loss"]
        running["cca_loss"] += step_out["cca_loss"]

        remain = self.cfg.max_cca_metric_samples - len(running["cca_labels"])
        if remain > 0:
            running["cca_labels"].extend(step_out["cca_labels"][:remain])
            running["cca_logits"].extend(step_out["cca_logits"][:remain])

    def _log_step(self, running: Dict):
        steps = max(1, running["steps"])
        lr = self.scheduler.get_last_lr()[0]

        loss = running["loss"] / steps
        sr_loss = running["sr_loss"] / steps
        cca_loss = running["cca_loss"] / steps

        cca_metrics = compute_cca_metrics(
            labels=running["cca_labels"],
            logits=running["cca_logits"],
        )

        print(
            f"Step {self.global_step} | "
            f"lr={lr:.3e} | "
            f"loss={loss:.4f} | "
            f"sr={sr_loss:.4f} | "
            f"cca={cca_loss:.4f} | "
            f"cca_pos_acc={cca_metrics['pos_acc']:.4f} | "
            f"cca_neg_acc={cca_metrics['neg_acc']:.4f} | "
            f"auroc={cca_metrics['auroc']:.4f} | "
            f"auprc={cca_metrics['auprc']:.4f}"
        )

        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.global_step,
                    lr,
                    loss,
                    sr_loss,
                    cca_loss,
                    cca_metrics["pos_acc"],
                    cca_metrics["neg_acc"],
                    cca_metrics["auroc"],
                    cca_metrics["auprc"],
                ]
            )

    def _check_mode(self):
        if self.cfg.mode not in {"sr", "sr_cca"}:
            raise ValueError("`mode` should be either 'sr' or 'sr_cca'.")