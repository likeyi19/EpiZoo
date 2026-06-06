# epizoo/metrics/imputation.py

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch


def compute_imputation_correlations(
    adata,
    adata_imputed,
    cell_type_key: str = "cell_type",
    layer: Optional[str] = None,
    imputed_layer: Optional[str] = None,
) -> Dict[str, object]:
    """
    Compute correlation metrics for data imputation.

    Metrics
    -------
    1. cell_type_corr:
        For each cell, compute Pearson correlation between its imputed profile
        and the mean true TF-IDF profile of its cell type.

    2. cell_corr:
        For each cell, compute Pearson correlation between its true profile
        and imputed profile.

    3. feature_corr:
        For each feature/cCRE, compute Pearson correlation between its true
        values and imputed values across cells.

    Parameters
    ----------
    adata:
        AnnData containing true TF-IDF signals.

    adata_imputed:
        AnnData containing imputed signals.

    cell_type_key:
        Column in `adata.obs` used for cell type labels.

    layer:
        Optional layer key for true signals. If None, use `adata.X`.

    imputed_layer:
        Optional layer key for imputed signals. If None, use `adata_imputed.X`.

    Returns
    -------
    results:
        Dictionary containing:
            - "cell_type_corr": np.ndarray, shape [n_cells]
            - "cell_corr": np.ndarray, shape [n_cells]
            - "feature_corr": np.ndarray, shape [n_features]
            - "summary": pd.DataFrame
    """

    _check_inputs(
        adata=adata,
        adata_imputed=adata_imputed,
        cell_type_key=cell_type_key,
    )

    x_true = get_matrix(adata, layer)
    x_imputed = get_matrix(adata_imputed, imputed_layer)

    if x_true.shape != x_imputed.shape:
        raise ValueError(
            "`adata` and `adata_imputed` should have the same matrix shape. "
            f"Got {x_true.shape} and {x_imputed.shape}."
        )

    cell_types = np.asarray(adata.obs[cell_type_key])

    cell_type_corr = compute_cell_type_correlation(
        x_true=x_true,
        x_imputed=x_imputed,
        cell_types=cell_types,
    )

    cell_corr = rowwise_pearson(
        x_true,
        x_imputed,
    )

    feature_corr = rowwise_pearson(
        x_true.T,
        x_imputed.T,
    )

    summary = summarize_correlations(
        {
            "cell_type_corr": cell_type_corr,
            "cell_corr": cell_corr,
            "feature_corr": feature_corr,
        }
    )

    return {
        "cell_type_corr": cell_type_corr,
        "cell_corr": cell_corr,
        "feature_corr": feature_corr,
        "summary": summary,
    }


def compute_cell_type_correlation(
    x_true: np.ndarray,
    x_imputed: np.ndarray,
    cell_types,
) -> np.ndarray:
    """
    For each cell, correlate imputed profile with the mean true profile
    of its cell type.
    """

    cell_types = np.asarray(cell_types)
    correlations = np.full(x_true.shape[0], np.nan, dtype=np.float32)

    for cell_type in pd.unique(cell_types):
        mask = cell_types == cell_type
        idx = np.where(mask)[0]

        mean_profile = x_true[mask].mean(axis=0)

        correlations[idx] = rowwise_pearson(
            x_imputed[mask],
            np.broadcast_to(mean_profile, x_imputed[mask].shape),
        )

    return correlations


def rowwise_pearson(
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute Pearson correlation for each pair of rows in x and y.

    x and y should have the same shape.
    """

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if x.shape != y.shape:
        raise ValueError(f"`x` and `y` should have the same shape. Got {x.shape} and {y.shape}.")

    x_centered = x - x.mean(axis=1, keepdims=True)
    y_centered = y - y.mean(axis=1, keepdims=True)

    numerator = np.sum(x_centered * y_centered, axis=1)

    denominator = np.sqrt(
        np.sum(x_centered ** 2, axis=1)
        * np.sum(y_centered ** 2, axis=1)
    )

    corr = np.divide(
        numerator,
        denominator,
        out=np.full(x.shape[0], np.nan, dtype=np.float32),
        where=denominator > eps,
    )

    return corr.astype(np.float32, copy=False)


def summarize_correlations(
    correlations: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Summarize correlation arrays.
    """

    records = []

    for name, values in correlations.items():
        values = np.asarray(values, dtype=np.float32)

        records.append(
            {
                "metric": name,
                "n": int(values.size),
                "n_valid": int(np.sum(~np.isnan(values))),
                "mean": float(np.nanmean(values)),
                "median": float(np.nanmedian(values)),
                "std": float(np.nanstd(values)),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
            }
        )

    return pd.DataFrame(records)


def get_matrix(
    adata,
    layer: Optional[str] = None,
) -> np.ndarray:
    """
    Get dense matrix from adata.X or adata.layers[layer].
    """

    x = adata.X if layer is None else adata.layers[layer]

    if hasattr(x, "toarray"):
        x = x.toarray()

    return np.asarray(x, dtype=np.float32)


def _check_inputs(
    adata,
    adata_imputed,
    cell_type_key: str,
):
    """
    Validate AnnData inputs.
    """

    if adata.n_obs != adata_imputed.n_obs:
        raise ValueError(
            "`adata` and `adata_imputed` should have the same number of cells. "
            f"Got {adata.n_obs} and {adata_imputed.n_obs}."
        )

    if adata.n_vars != adata_imputed.n_vars:
        raise ValueError(
            "`adata` and `adata_imputed` should have the same number of features. "
            f"Got {adata.n_vars} and {adata_imputed.n_vars}."
        )

    if cell_type_key not in adata.obs:
        raise KeyError(f"`{cell_type_key}` not found in `adata.obs`.")


def pearson_corr_log(
    preds,
    targets,
    multiplier: float = 1000.0,
    eps: float = 1e-7,
) -> float:
    """
    Global Pearson correlation after log1p transformation.
    """

    preds = _to_tensor(preds).float()
    targets = _to_tensor(targets).float()

    preds = torch.log1p(torch.clamp(preds, min=0.0) * multiplier).flatten()
    targets = torch.log1p(torch.clamp(targets, min=0.0) * multiplier).flatten()

    preds = preds - preds.mean()
    targets = targets - targets.mean()

    cov = torch.sum(preds * targets)
    denom = torch.sqrt(torch.sum(preds ** 2)) * torch.sqrt(torch.sum(targets ** 2))

    return float((cov / (denom + eps)).detach().cpu())


def spearman_corr_per_class(
    preds,
    targets,
    eps: float = 1e-7,
) -> float:
    """
    Mean Spearman correlation across classes / cell types.
    """

    preds = _to_tensor(preds).float()
    targets = _to_tensor(targets).float()

    if preds.shape != targets.shape:
        raise ValueError(
            "`preds` and `targets` should have the same shape. "
            f"Got {tuple(preds.shape)} and {tuple(targets.shape)}."
        )

    n = preds.shape[0]

    pred_rank = torch.argsort(torch.argsort(preds, dim=0), dim=0).float()
    target_rank = torch.argsort(torch.argsort(targets, dim=0), dim=0).float()

    mean_rank = (n - 1) / 2.0

    pred_rank = pred_rank - mean_rank
    target_rank = target_rank - mean_rank

    cov = torch.sum(pred_rank * target_rank, dim=0)
    pred_var = torch.sqrt(torch.sum(pred_rank ** 2, dim=0))
    target_var = torch.sqrt(torch.sum(target_rank ** 2, dim=0))

    corr = cov / (pred_var * target_var + eps)

    return float(corr.mean().detach().cpu())


def compute_seq_correlations(
    preds,
    targets,
    multiplier: float = 1000.0,
) -> dict:
    """
    Compute EpiZooSeq prediction correlations.
    """

    return {
        "pearson_log": pearson_corr_log(
            preds=preds,
            targets=targets,
            multiplier=multiplier,
        ),
        "spearman_per_class": spearman_corr_per_class(
            preds=preds,
            targets=targets,
        ),
    }


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x

    return torch.tensor(np.asarray(x))