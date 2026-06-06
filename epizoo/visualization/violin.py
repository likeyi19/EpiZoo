# epizoo/visualization/imputation.py

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def plot_correlation_violin(
    correlations: Dict[str, np.ndarray],
    output_file: Optional[str] = None,
    title: str = "Imputation correlation",
    show: bool = False,
    figsize=(6, 5),
):
    """
    Plot violin plot for imputation correlation metrics.

    Parameters
    ----------
    correlations:
        Dictionary from metric name to correlation array.

        Example:
            {
                "cell_type_corr": ...,
                "cell_corr": ...,
                "feature_corr": ...,
            }

    output_file:
        Optional path to save the figure.

    title:
        Plot title.

    show:
        Whether to show the figure.

    figsize:
        Figure size.

    Returns
    -------
    ax:
        Matplotlib axis.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    df = correlation_dict_to_frame(correlations)

    fig, ax = plt.subplots(figsize=figsize)

    sns.violinplot(
        data=df,
        x="metric",
        y="correlation",
        inner="box",
        cut=0,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Pearson r")
    ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ax


def correlation_dict_to_frame(
    correlations: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Convert correlation dictionary to long-format DataFrame for plotting.
    """

    frames = []

    for metric, values in correlations.items():
        values = np.asarray(values, dtype=np.float32)
        values = values[~np.isnan(values)]

        frames.append(
            pd.DataFrame(
                {
                    "metric": metric,
                    "correlation": values,
                }
            )
        )

    if len(frames) == 0:
        return pd.DataFrame(columns=["metric", "correlation"])

    return pd.concat(frames, axis=0, ignore_index=True)