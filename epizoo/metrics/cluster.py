# epizoo/inference/cluster.py

from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_score,
    normalized_mutual_info_score,
)


MetricName = Literal["NMI", "ARI", "Homo", "AMI"]


def compute_cluster_scores(
    labels,
    clusters,
) -> dict:
    """
    Compute clustering metrics.
    """

    return {
        "NMI": normalized_mutual_info_score(labels, clusters),
        "ARI": adjusted_rand_score(labels, clusters),
        "Homo": homogeneity_score(labels, clusters),
        "AMI": adjusted_mutual_info_score(labels, clusters),
    }


def run_louvain(
    adata,
    label_key: str,
    cluster_key: str = "cluster",
    range_min: float = 0.0,
    range_max: float = 3.0,
    max_steps: int = 30,
    opt_metric: MetricName = "NMI",
    resolutions: Optional[Sequence[float]] = None,
    use_rep: Optional[str] = None,
    inplace: bool = True,
    plot: bool = False,
    force: bool = True,
    verbose: bool = True,
):
    """
    Run Louvain clustering and select the best resolution.

    This keeps the core behavior of the original function:
        1. test candidate resolutions
        2. select the resolution with best metric
        3. additionally run a binary search to find a resolution whose
           cluster number matches the number of labels, if possible

    Returns
    -------
    If inplace=True:
        score_df

    If inplace=False:
        score_df, clustering
    """

    if opt_metric not in {"NMI", "ARI", "Homo", "AMI"}:
        raise ValueError("`opt_metric` should be one of: NMI, ARI, Homo, AMI.")

    if cluster_key in adata.obs.columns:
        if force:
            print(f"Warning: `{cluster_key}` already exists and will be overwritten.")
            del adata.obs[cluster_key]
        else:
            raise ValueError(
                f"`{cluster_key}` already exists in adata.obs. "
                "Set `force=True` to overwrite it."
            )

    if resolutions is None:
        # Keep the old default behavior: [1.0, 2.0]
        resolutions = [1.0, 2.0]

    if "neighbors" not in adata.uns:
        if verbose:
            print("Computing neighbors...")
        sc.pp.neighbors(adata, use_rep=use_rep)

    records = []
    best_score = -np.inf
    best_res = resolutions[0]
    best_scores = None
    best_clustering = None

    if verbose:
        print("Running Louvain clustering...")

    for res in resolutions:
        sc.tl.louvain(
            adata,
            resolution=res,
            key_added=cluster_key,
        )

        scores = compute_cluster_scores(
            labels=adata.obs[label_key],
            clusters=adata.obs[cluster_key],
        )

        record = {"resolution": res, **scores}
        records.append(record)

        if verbose:
            print(
                f"resolution={res}, "
                f"NMI={scores['NMI']:.4f}, "
                f"ARI={scores['ARI']:.4f}, "
                f"Homo={scores['Homo']:.4f}, "
                f"AMI={scores['AMI']:.4f}"
            )

        score = scores[opt_metric]

        if score > best_score:
            best_score = score
            best_res = res
            best_scores = scores
            best_clustering = adata.obs[cluster_key].copy()

        del adata.obs[cluster_key]

    binary_record, binary_clustering = _binary_search_louvain(
        adata=adata,
        label_key=label_key,
        cluster_key=cluster_key,
        range_min=range_min,
        range_max=range_max,
        max_steps=max_steps,
        opt_metric=opt_metric,
        verbose=verbose,
    )

    if binary_record is not None:
        records.insert(0, binary_record)

        if binary_record[opt_metric] > best_score:
            best_score = binary_record[opt_metric]
            best_res = binary_record["resolution"]
            best_scores = {
                "NMI": binary_record["NMI"],
                "ARI": binary_record["ARI"],
                "Homo": binary_record["Homo"],
                "AMI": binary_record["AMI"],
            }
            best_clustering = binary_clustering

    records.insert(0, {"resolution": best_res, **best_scores})
    score_df = pd.DataFrame(records)

    if verbose:
        print(f"Optimized clustering against `{label_key}`.")
        print(f"Best resolution: {best_res}")
        print(f"Best {opt_metric}: {best_score:.4f}")
        print(
            f"NMI={best_scores['NMI']:.4f}, "
            f"ARI={best_scores['ARI']:.4f}, "
            f"Homo={best_scores['Homo']:.4f}, "
            f"AMI={best_scores['AMI']:.4f}"
        )

    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.lineplot(data=score_df, x="resolution", y=opt_metric)
        plt.title("Optimal cluster resolution profile")
        plt.show()

    if inplace:
        adata.obs[cluster_key] = best_clustering
        return score_df

    return score_df, best_clustering


def _binary_search_louvain(
    adata,
    label_key: str,
    cluster_key: str,
    range_min: float,
    range_max: float,
    max_steps: int,
    opt_metric: MetricName,
    verbose: bool,
) -> Tuple[Optional[dict], Optional[pd.Series]]:
    """
    Binary search for a resolution that gives the same number of clusters
    as the number of labels.
    """

    target_clusters = np.unique(adata.obs[label_key]).shape[0]

    low = float(range_min)
    high = float(range_max)

    for _ in range(max_steps):
        res = low + (high - low) / 2

        sc.tl.louvain(
            adata,
            resolution=res,
            key_added=cluster_key,
        )

        n_clusters = adata.obs[cluster_key].nunique()

        if n_clusters > target_clusters:
            high = res
            del adata.obs[cluster_key]
            continue

        if n_clusters < target_clusters:
            low = res
            del adata.obs[cluster_key]
            continue

        scores = compute_cluster_scores(
            labels=adata.obs[label_key],
            clusters=adata.obs[cluster_key],
        )

        record = {"resolution": res, **scores}
        clustering = adata.obs[cluster_key].copy()

        if verbose:
            print("Louvain clustering with binary search")
            print(
                f"resolution={res}, "
                f"NMI={scores['NMI']:.4f}, "
                f"ARI={scores['ARI']:.4f}, "
                f"Homo={scores['Homo']:.4f}, "
                f"AMI={scores['AMI']:.4f}"
            )

        del adata.obs[cluster_key]

        return record, clustering

    return None, None