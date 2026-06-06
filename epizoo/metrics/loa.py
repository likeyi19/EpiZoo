# epizoo/metrics/mutations.py

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


def compute_mutation_metrics(
    ref_pred: ArrayLike,
    alt_pred: ArrayLike,
    mut_idx: int,
    window_size: int = 10,
    eps: float = 1e-6,
    return_numpy: bool = True,
) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """
    Compute per-cell mutation impact metrics.

    Parameters
    ----------
    ref_pred:
        Reference predicted signals.
        Shape: [num_cells, num_ccres]

    alt_pred:
        Alternative predicted signals.
        Shape: [num_cells, num_ccres]

    mut_idx:
        Mutation cCRE index in signal space.
        This should be 0-based and should NOT include special-token offset.

    window_size:
        Number of cCREs on each side of mut_idx for local window score.

    eps:
        Small pseudocount for log fold-change.

    Returns
    -------
    metrics:
        {
            "point":  [num_cells],
            "window": [num_cells],
            "global": [num_cells],
        }
    """

    ref_pred = _to_tensor(ref_pred).float()
    alt_pred = _to_tensor(alt_pred).float().to(ref_pred.device)

    if ref_pred.shape != alt_pred.shape:
        raise ValueError(
            "`ref_pred` and `alt_pred` should have the same shape. "
            f"Got {tuple(ref_pred.shape)} and {tuple(alt_pred.shape)}."
        )

    if ref_pred.ndim != 2:
        raise ValueError(
            "`ref_pred` and `alt_pred` should have shape [num_cells, num_ccres]."
        )

    num_ccres = ref_pred.shape[1]

    if mut_idx < 0 or mut_idx >= num_ccres:
        raise IndexError(
            f"`mut_idx` out of range. Expected [0, {num_ccres}), got {mut_idx}."
        )

    lfc = torch.log2((alt_pred + eps) / (ref_pred + eps))

    point = lfc[:, mut_idx]

    start = max(0, mut_idx - window_size)
    end = min(num_ccres, mut_idx + window_size + 1)
    window = torch.abs(lfc[:, start:end]).sum(dim=1)

    global_score = torch.norm(lfc, p=2, dim=1)

    metrics = {
        "point": point,
        "window": window,
        "global": global_score,
    }

    if return_numpy:
        metrics = {
            key: value.detach().cpu().numpy()
            for key, value in metrics.items()
        }

    return metrics


def summarize_loa_score(
    mutation_metrics: Dict[str, ArrayLike],
    top_k: int = 200,
) -> float:
    """
    Summarize one mutation into a final LoA score.

    Final score:
        abs(mean(point score of top-k cells ranked by global score))
    """

    point = _to_tensor(mutation_metrics["point"]).float()
    global_score = _to_tensor(mutation_metrics["global"]).float()

    if point.ndim != 1 or global_score.ndim != 1:
        raise ValueError("`point` and `global` should be 1D arrays.")

    if point.shape[0] != global_score.shape[0]:
        raise ValueError(
            "`point` and `global` should have the same length. "
            f"Got {point.shape[0]} and {global_score.shape[0]}."
        )

    k = min(top_k, point.shape[0])
    top_idx = torch.topk(global_score, k=k, largest=True).indices

    score = torch.abs(point[top_idx].mean())

    return float(score.detach().cpu())


def compute_loa_score(
    ref_pred: ArrayLike,
    alt_pred: ArrayLike,
    mut_idx: int,
    window_size: int = 10,
    top_k: int = 200,
    eps: float = 1e-6,
) -> Dict[str, object]:
    """
    Compute mutation metrics and final LoA score.
    """

    metrics = compute_mutation_metrics(
        ref_pred=ref_pred,
        alt_pred=alt_pred,
        mut_idx=mut_idx,
        window_size=window_size,
        eps=eps,
        return_numpy=False,
    )

    score = summarize_loa_score(
        mutation_metrics=metrics,
        top_k=top_k,
    )

    metrics_np = {
        key: value.detach().cpu().numpy()
        for key, value in metrics.items()
    }

    return {
        "score": score,
        "metrics": metrics_np,
    }


def _to_tensor(x: ArrayLike) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x

    return torch.tensor(x)