# epizoo/metrics/classification.py

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


def compute_classification_metrics(
    labels,
    preds,
) -> Dict[str, float]:
    """
    Compute annotation classification metrics.

    Metrics:
        acc:
            Accuracy.

        kappa:
            Cohen's kappa.

        macro_f1:
            Macro-averaged F1 score.
    """

    labels = _to_numpy(labels)
    preds = _to_numpy(preds)

    return {
        "acc": accuracy_score(labels, preds),
        "kappa": cohen_kappa_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    return np.asarray(x)