# epizoo/visualization/attribution.py

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def plot_attribution_logo(
    sequence: str,
    scores,
    chrom: Optional[str] = None,
    start_pos: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (30, 2),
):
    """
    Plot base-level attribution logo.

    Parameters
    ----------
    sequence:
        DNA sequence.

    scores:
        Base-level attribution scores with length equal to sequence length.

    chrom:
        Optional chromosome name for x-axis label.

    start_pos:
        Optional genomic start position.
        If provided, x-axis uses absolute genomic coordinates.

    title:
        Plot title.

    save_path:
        Optional output path.

    Returns
    -------
    fig, ax, logo_matrix
    """

    try:
        import logomaker
    except ImportError as exc:
        raise ImportError(
            "`logomaker` is required for attribution logo plotting. "
            "Please install it with `pip install logomaker`."
        ) from exc

    scores = _to_numpy(scores).reshape(-1)

    if len(sequence) != len(scores):
        raise ValueError(
            "`sequence` and `scores` should have the same length. "
            f"Got {len(sequence)} and {len(scores)}."
        )

    logo_matrix = attribution_logo_matrix(
        sequence=sequence,
        scores=scores,
        start_pos=start_pos,
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )

    fig, ax = plt.subplots(figsize=figsize)

    colors = {
        "A": "#CC0000",
        "C": "#0000CC",
        "G": "#FFB300",
        "T": "#008000",
    }

    logo = logomaker.Logo(
        logo_matrix,
        ax=ax,
        color_scheme=colors,
        vpad=0,
    )

    logo.style_spines(visible=False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    if title is not None:
        ax.set_title(title, fontsize=14, pad=15)

    ax.set_ylabel("Contribution score", fontsize=12)

    if chrom is not None:
        ax.set_xlabel(f"Genomic position on {chrom}", fontsize=12)
    else:
        ax.set_xlabel("Sequence position", fontsize=12)

    ax.set_xlim(
        logo_matrix.index.min() - 0.5,
        logo_matrix.index.max() + 0.5,
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=500,
            bbox_inches="tight",
        )

    return fig, ax, logo_matrix


def attribution_logo_matrix(
    sequence: str,
    scores,
    start_pos: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build logomaker matrix from sequence and attribution scores.
    """

    scores = _to_numpy(scores).reshape(-1)

    if len(sequence) != len(scores):
        raise ValueError(
            "`sequence` and `scores` should have the same length."
        )

    if start_pos is None:
        index = np.arange(len(sequence))
    else:
        index = np.arange(start_pos, start_pos + len(sequence))

    matrix = pd.DataFrame(
        0.0,
        columns=list("ACGT"),
        index=index,
    )

    for pos, base, score in zip(index, sequence.upper(), scores):
        if base in matrix.columns:
            matrix.loc[pos, base] = score

    return matrix


def plot_attribution_distribution(
    scores,
    title: str = "Attribution score distribution",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (5, 2),
):
    """
    Plot histogram and boxplot for attribution scores.
    """

    scores = _to_numpy(scores).reshape(-1)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
    )

    sns.histplot(
        scores,
        kde=True,
        color="skyblue",
        ax=axes[0],
    )
    axes[0].set_xlabel("Attribution score")

    sns.boxplot(
        x=scores,
        color="lightcoral",
        ax=axes[1],
    )
    axes[1].set_xlabel("Attribution score")

    fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=500,
            bbox_inches="tight",
        )

    return fig, axes


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return np.asarray(x)