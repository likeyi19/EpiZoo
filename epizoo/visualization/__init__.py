# epizoo/visualization/__init__.py

from .umap import (
    build_embedding_adata,
    compute_umap,
    plot_umap,
    run_umap,
)

from .violin import (
    plot_correlation_violin,
    correlation_dict_to_frame,
)

from .heatmap import (
    confusion_matrix_frame,
    plot_confusion_heatmap,
)

from .density import plot_density_scatter

from .attribution import (
    plot_attribution_logo,
    attribution_logo_matrix,
    plot_attribution_distribution,
)

__all__ = [
    "build_embedding_adata",
    "compute_umap",
    "plot_umap",
    "run_umap",
    "plot_correlation_violin",
    "correlation_dict_to_frame",
    "confusion_matrix_frame",
    "plot_confusion_heatmap",
    "plot_density_scatter",
    "plot_attribution_logo",
    "attribution_logo_matrix",
    "plot_attribution_distribution",
]