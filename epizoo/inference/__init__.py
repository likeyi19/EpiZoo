# epizoo/inference/__init__.py

from .utils import get_device
from .embeddings import (
    extract_cell_embeddings,
    extract_seq_embeddings,
    compute_cell_type_embeddings,
)
from .signals import predict_signals
from .annotation import predict_cell_types

from .mutations import (
    compute_seq_delta_embedding,
    predict_mutation_signals,
    score_mutation_loa,
)

from .seq import (
    predict_seq_accessibility,
    run_seq_attribution,
    compute_base_attributions,
    token_scores_to_base_scores,
    smooth_attribution_scores,
    extract_top_motifs_for_meme,
)

__all__ = [
    "get_device",
    "extract_cell_embeddings",
    "extract_seq_embeddings",
    "compute_cell_type_embeddings",
    "predict_signals",
    "predict_cell_types",
    "compute_seq_delta_embedding",
    "predict_mutation_signals",
    "score_mutation_loa",
    "predict_seq_accessibility",
    "run_seq_attribution",
    "compute_base_attributions",
    "token_scores_to_base_scores",
    "smooth_attribution_scores",
    "extract_top_motifs_for_meme",
]