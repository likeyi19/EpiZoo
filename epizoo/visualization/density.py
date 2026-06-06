# epizoo/visualization/density.py

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_density_scatter(
    preds,
    targets,
    title: str = "EpiZoo",
    cmap: str = "GnBu",
    downsample_size: int = 5000,
    save_path: Optional[str] = None,
):
    """
    Plot density scatter of observed vs predicted signals.

    Signals are transformed by:
        log1p(max(x, 0))
    """

    preds = _to_numpy(preds)
    targets = _to_numpy(targets)

    pred_log = np.log1p(np.maximum(preds, 0)).flatten()
    target_log = np.log1p(np.maximum(targets, 0)).flatten()

    mask = target_log > 0
    pred_log = pred_log[mask]
    target_log = target_log[mask]

    density = _fit_kde_with_jitter(
        x=target_log,
        y=pred_log,
        downsample_size=downsample_size,
    )

    fig, ax = plt.subplots(figsize=(1.5, 1.5))

    max_val = max(target_log.max(), pred_log.max())
    ax.plot(
        [0, max_val],
        [0, max_val],
        "--",
        color="#7f8c8d",
        linewidth=1,
        alpha=0.8,
        zorder=1,
    )

    sc = ax.scatter(
        target_log,
        pred_log,
        c=density,
        cmap=cmap,
        s=2,
        alpha=0.5,
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(
        "right",
        size="5%",
        pad="7%",
    )

    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Density", fontsize=10)
    cbar.outline.set_linewidth(0.8)
    cbar.ax.tick_params(
        labelsize=9,
        width=0.8,
        direction="out",
    )

    ax.set_title(title, fontsize=10, pad=10)
    ax.set_xlabel("Observed signal (log1p)", fontsize=10)
    ax.set_ylabel("Predicted signal (log1p)", fontsize=10)

    ax.set_xlim(-0.2, max_val)
    ax.set_ylim(-0.2, max_val)
    ax.set_box_aspect(1)

    sns.despine()
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10,
        direction="out",
        length=3,
    )

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=500,
            bbox_inches="tight",
        )

    return fig, ax


def _fit_kde_with_jitter(
    x,
    y,
    downsample_size: int = 5000,
):
    xy = np.vstack([x, y])
    xy = xy + np.random.normal(
        loc=0,
        scale=1e-8,
        size=xy.shape,
    )

    if downsample_size < xy.shape[1]:
        idx = np.random.choice(
            xy.shape[1],
            downsample_size,
            replace=False,
        )
        kde = gaussian_kde(xy[:, idx])
    else:
        kde = gaussian_kde(xy)

    return kde(xy)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return np.asarray(x)