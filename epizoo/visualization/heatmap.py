# epizoo/visualization/heatmap.py

from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix


NormalizeMode = Optional[Literal["true", "pred", "all"]]


def confusion_matrix_frame(
    labels,
    preds,
    class_ids: Optional[Sequence[int]] = None,
    class_names: Optional[Sequence[str]] = None,
    normalize: NormalizeMode = "true",
) -> pd.DataFrame:
    """
    Build a confusion matrix DataFrame.

    Parameters
    ----------
    labels:
        True class ids.

    preds:
        Predicted class ids.

    class_ids:
        Class ids used to order rows and columns.
        If None, infer from labels and preds.

    class_names:
        Display names for rows and columns.
        If None, use class ids as names.

    normalize:
        - None: raw counts
        - "true": normalize each row by true class count
        - "pred": normalize each column by predicted class count
        - "all": normalize by total count
    """

    labels = _to_numpy(labels)
    preds = _to_numpy(preds)

    if class_ids is None:
        class_ids = sorted(set(labels.tolist()) | set(preds.tolist()))

    cm = confusion_matrix(
        labels,
        preds,
        labels=class_ids,
    ).astype(float)

    cm = _normalize_confusion_matrix(
        cm=cm,
        normalize=normalize,
    )

    if class_names is None:
        class_names = [str(x) for x in class_ids]

    return pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    )


def plot_confusion_heatmap(
    labels,
    preds,
    class_ids: Optional[Sequence[int]] = None,
    class_names: Optional[Sequence[str]] = None,
    normalize: NormalizeMode = "true",
    output_file: Optional[str] = None,
    title: Optional[str] = None,
    annot: bool = False,
    show: bool = False,
    figsize=None,
):
    """
    Plot confusion matrix heatmap for annotation results.

    Returns
    -------
    ax:
        Matplotlib axis.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    cm_df = confusion_matrix_frame(
        labels=labels,
        preds=preds,
        class_ids=class_ids,
        class_names=class_names,
        normalize=normalize,
    )

    if figsize is None:
        n = max(4, len(cm_df))
        figsize = (0.45 * n + 3, 0.45 * n + 3)

    fig, ax = plt.subplots(figsize=figsize)

    fmt = ".2f" if normalize is not None else ".0f"

    sns.heatmap(
        cm_df,
        annot=annot,
        fmt=fmt,
        square=True,
        cbar=True,
        ax=ax,
    )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    if title is None:
        title = "Confusion matrix"

    ax.set_title(title)

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ax


def _normalize_confusion_matrix(
    cm: np.ndarray,
    normalize: NormalizeMode,
) -> np.ndarray:
    if normalize is None:
        return cm

    if normalize == "true":
        denom = cm.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        denom = cm.sum(axis=0, keepdims=True)
    elif normalize == "all":
        denom = cm.sum()
    else:
        raise ValueError("`normalize` should be None, 'true', 'pred', or 'all'.")

    return np.divide(
        cm,
        denom,
        out=np.zeros_like(cm, dtype=float),
        where=denom != 0,
    )


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    return np.asarray(x)