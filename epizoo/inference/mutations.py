# epizoo/inference/mutations.py

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from epizoo.data import SEAMDataset, collate_fn_seam
from epizoo.inference.embeddings import extract_seq_embeddings
from epizoo.inference.signals import predict_signal_delta
from epizoo.inference.utils import get_device
from epizoo.metrics.loa import (
    compute_mutation_metrics,
    summarize_loa_score,
)


ArrayLike = Union[np.ndarray, torch.Tensor]


@torch.no_grad()
def compute_seq_delta_embedding(
    seam_model,
    ref_sequence: str,
    alt_sequence: str,
    dnabert_path: str,
    device: Optional[str] = None,
    max_length: int = 512,
    batch_size: int = 2,
    num_workers: int = 0,
    use_amp: bool = True,
) -> torch.Tensor:
    """
    Compute sequence-level mutation embedding.

    delta_embedding = SEAM(alt_sequence) - SEAM(ref_sequence)

    Parameters
    ----------
    seam_model:
        SEAM model.

    ref_sequence:
        Reference DNA sequence.

    alt_sequence:
        Mutated DNA sequence.

    dnabert_path:
        DNABERT tokenizer path.

    Returns
    -------
    delta_embedding:
        Tensor with shape [emb_dim].
    """

    dataset = SEAMDataset(
        sequences=[ref_sequence, alt_sequence],
        dnabert_path=dnabert_path,
        max_length=max_length,
        return_index=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_seam,
    )

    seq_emb = extract_seq_embeddings(
        model=seam_model,
        dataloader=dataloader,
        device=device,
        use_amp=use_amp,
        return_numpy=False,
        show_progress=False,
    )

    ref_emb = seq_emb[0]
    alt_emb = seq_emb[1]

    return alt_emb - ref_emb


@torch.no_grad()
def predict_mutation_signals(
    model,
    dataloader,
    delta_embedding: ArrayLike,
    device: Optional[str] = None,
    use_amp: bool = True,
    apply_sigmoid: bool = True,
    return_numpy: bool = False,
    show_progress: bool = True,
):
    """
    Predict reference and mutation-perturbed signals.

    This is a mutation-specific wrapper around `predict_signal_delta`.

    For each cell:
        ref_pred = decoder(cell_emb)
        alt_pred = decoder(cell_emb + delta_embedding)
        loa = alt_pred - ref_pred

    Parameters
    ----------
    model:
        EpiZooCancer or compatible EpiZoo model.

    dataloader:
        Dataloader from InferenceCellDatasetCancer + inference_collate_fn_cancer,
        or any dataloader supported by predict_signal_delta.

    delta_embedding:
        Mutation embedding with shape [emb_dim] or [1, emb_dim].

    apply_sigmoid:
        Whether to apply sigmoid to decoder logits.
        For BCEWithLogits-trained signal decoders, this should usually be True.

    Returns
    -------
    outputs:
        {
            "ref_pred": Tensor or ndarray [num_cells, num_ccres],
            "alt_pred": Tensor or ndarray [num_cells, num_ccres],
            "loa":      Tensor or ndarray [num_cells, num_ccres],
        }
    """

    return predict_signal_delta(
        model=model,
        dataloader=dataloader,
        delta_emb=delta_embedding,
        device=device,
        use_amp=use_amp,
        apply_sigmoid=apply_sigmoid,
        return_numpy=return_numpy,
        show_progress=show_progress,
    )


@torch.no_grad()
def score_mutation_loa(
    model,
    seam_model,
    dataloader,
    ref_sequence: str,
    alt_sequence: str,
    mut_idx: int,
    dnabert_path: str,
    device: Optional[str] = None,
    max_length: int = 512,
    window_size: int = 10,
    top_k: int = 200,
    eps: float = 1e-6,
    use_amp: bool = True,
    return_predictions: bool = False,
    show_progress: bool = True,
) -> Dict[str, object]:
    """
    End-to-end LoA scoring for one mutation.

    Steps:
        1. Compute delta_embedding = SEAM(alt) - SEAM(ref)
        2. Predict ref_pred and alt_pred for all cells
        3. Compute per-cell point / window / global mutation metrics
        4. Final LoA score =
           abs(mean(point score of top-k cells ranked by global score))

    Parameters
    ----------
    mut_idx:
        The mutation cCRE index in decoder signal space.
        This should be 0-based and should NOT include CCRE_TOKEN_OFFSET.

    Returns
    -------
    result:
        {
            "score": float,
            "metrics": {
                "point": ndarray [num_cells],
                "window": ndarray [num_cells],
                "global": ndarray [num_cells],
            },
            "delta_embedding": ndarray [emb_dim],
            optionally:
                "ref_pred", "alt_pred", "loa"
        }
    """

    device = get_device(device)

    delta_embedding = compute_seq_delta_embedding(
        seam_model=seam_model,
        ref_sequence=ref_sequence,
        alt_sequence=alt_sequence,
        dnabert_path=dnabert_path,
        device=device,
        max_length=max_length,
        use_amp=use_amp,
    )

    pred = predict_mutation_signals(
        model=model,
        dataloader=dataloader,
        delta_embedding=delta_embedding,
        device=device,
        use_amp=use_amp,
        apply_sigmoid=True,
        return_numpy=False,
        show_progress=show_progress,
    )

    metrics = compute_mutation_metrics(
        ref_pred=pred["ref_pred"].to(device),
        alt_pred=pred["alt_pred"].to(device),
        mut_idx=mut_idx,
        window_size=window_size,
        eps=eps,
        return_numpy=False,
    )

    score = summarize_loa_score(
        mutation_metrics=metrics,
        top_k=top_k,
    )

    result = {
        "score": score,
        "metrics": {
            key: value.detach().cpu().numpy()
            for key, value in metrics.items()
        },
        "delta_embedding": delta_embedding.detach().cpu().numpy(),
    }

    if return_predictions:
        result["ref_pred"] = pred["ref_pred"].detach().cpu().numpy()
        result["alt_pred"] = pred["alt_pred"].detach().cpu().numpy()
        result["loa"] = pred["loa"].detach().cpu().numpy()

    return result