# epizoo/metrics/cca.py

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_cca_metrics(labels, logits) -> Dict[str, float]:
    """
    Compute CCA metrics from binary labels and raw logits.

    Metrics:
        pos_acc:
            Fraction of positive samples with logits > 0.

        neg_acc:
            Fraction of negative samples with logits < 0.

        auroc:
            AUROC using raw logits as scores.

        auprc:
            AUPRC using raw logits as scores.
    """

    if len(labels) == 0 or len(logits) == 0:
        return {
            "pos_acc": float("nan"),
            "neg_acc": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
        }

    labels = np.asarray(labels)
    logits = np.asarray(logits)

    pos = labels > 0
    neg = labels == 0

    pos_acc = (
        np.sum(logits[pos] > 0) / np.sum(pos)
        if np.sum(pos) > 0
        else float("nan")
    )

    neg_acc = (
        np.sum(logits[neg] < 0) / np.sum(neg)
        if np.sum(neg) > 0
        else float("nan")
    )

    try:
        auroc = roc_auc_score(labels, logits)
    except ValueError:
        auroc = float("nan")

    try:
        auprc = average_precision_score(labels, logits)
    except ValueError:
        auprc = float("nan")

    return {
        "pos_acc": float(pos_acc),
        "neg_acc": float(neg_acc),
        "auroc": float(auroc),
        "auprc": float(auprc),
    }