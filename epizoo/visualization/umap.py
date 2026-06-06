# epizoo/inference/umap.py

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import scanpy as sc


def build_embedding_adata(
    embeddings,
    labels: Optional[Sequence] = None,
    label_key: str = "label",
):
    """
    Build AnnData from cell embeddings.
    """

    adata = sc.AnnData(np.asarray(embeddings))

    if labels is not None:
        adata.obs[label_key] = list(labels)

    return adata


def compute_umap(
    adata,
    use_rep: str = "X",
    n_neighbors: int = 15,
    random_state: int = 2020,
    copy: bool = False,
    **neighbors_kwargs,
):
    """
    Compute neighbors and UMAP.
    """

    out = adata.copy() if copy else adata

    np.random.seed(random_state)

    sc.pp.neighbors(
        out,
        n_neighbors=n_neighbors,
        use_rep=use_rep,
        **neighbors_kwargs,
    )

    sc.tl.umap(out)

    return out


def plot_umap(
    adata,
    color: Union[str, Sequence[str]] = "label",
    output_file: Optional[str] = None,
    show: bool = False,
    **plot_kwargs,
):
    """
    Plot UMAP and optionally save to file.
    """

    import matplotlib.pyplot as plt

    sc.pl.umap(
        adata,
        color=color,
        show=show,
        **plot_kwargs,
    )

    if output_file is not None:
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()

    return adata


def run_umap(
    embeddings,
    labels: Optional[Sequence] = None,
    label_key: str = "label",
    output_file: Optional[str] = None,
    color: Optional[Union[str, Sequence[str]]] = None,
    n_neighbors: int = 15,
    random_state: int = 2020,
    show: bool = False,
    **plot_kwargs,
):
    """
    Convenience pipeline:
        embeddings -> AnnData -> neighbors -> UMAP -> optional plot
    """

    adata = build_embedding_adata(
        embeddings=embeddings,
        labels=labels,
        label_key=label_key,
    )

    compute_umap(
        adata,
        use_rep="X",
        n_neighbors=n_neighbors,
        random_state=random_state,
    )

    if output_file is not None or show:
        plot_umap(
            adata,
            color=color or label_key,
            output_file=output_file,
            show=show,
            **plot_kwargs,
        )

    return adata