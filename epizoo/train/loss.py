# epizoo/train/loss.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineMSELogLoss(nn.Module):
    """
    Combined cosine loss and log-transformed MSE loss.

    This follows the original EpiZooSeq training objective.
    """

    def __init__(
        self,
        max_weight: float = 1.0,
        multiplier: float = 1000.0,
        reduction: str = "mean",
    ):
        super().__init__()

        self.max_weight = max_weight
        self.multiplier = multiplier
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred = pred.float()
        target = target.float()

        pred_pos = torch.clamp(pred, min=0.0)
        pred_neg = torch.clamp(pred, max=0.0)

        log_pred = (
            torch.log1p(self.multiplier * pred_pos)
            - torch.log1p(torch.abs(self.multiplier * pred_neg))
        )
        log_target = torch.log1p(self.multiplier * target)

        mse = torch.mean((log_pred - log_target) ** 2)

        weight = torch.clamp(
            torch.abs(mse),
            min=1.0,
            max=self.max_weight,
        )

        cosine = -F.cosine_similarity(
            pred,
            target,
            dim=-1,
        )

        loss = weight * cosine + mse

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss