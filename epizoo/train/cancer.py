# epizoo/train/cancer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from epizoo.train.finetune import (
    EpiZooFinetuneTrainer,
    FineTuneConfig,
    LoRAConfig,
    TaskMode,
)


@dataclass
class CancerTrainConfig(FineTuneConfig):
    """
    Training config for EpiZooCancer.

    Default mode is SR + CCA, matching the original cancer fine-tuning script.
    """

    mode: TaskMode = "sr_cca"


class EpiZooCancerTrainer(EpiZooFinetuneTrainer):
    """
    Trainer for EpiZooCancer.

    Expected batch format from `collate_fn_cancer`:
        (
            input_ids,
            signals_human,
            signals_mouse,
            cca_ids,
            cca_labels,
            species,
            cancer_type,
        )

    Difference from EpiZooFinetuneTrainer:
        - forward receives `cancer_type_ids`
        - all SR / CCA loss computation, logging, checkpointing, AMP,
          scheduler, and CCA metrics are inherited.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        cfg: Optional[CancerTrainConfig] = None,
        lora_cfg: Optional[LoRAConfig] = None,
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            cfg=cfg or CancerTrainConfig(),
            lora_cfg=lora_cfg,
        )

    def _move_batch(self, batch):
        (
            input_ids,
            signals_human,
            signals_mouse,
            cca_ids,
            cca_labels,
            species,
            cancer_type,
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
            "cancer_type": cancer_type.to(self.device).long(),
        }

    def _forward_model(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            cancer_type_ids=batch["cancer_type"],
            return_transformer_out=False,
        )