# epizoo/inference/seq.py

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import amp
from tqdm import tqdm

from epizoo.inference.utils import get_device


@torch.no_grad()
def predict_seq_accessibility(
    model,
    dataloader,
    device: Optional[str] = None,
    use_amp: bool = True,
    return_numpy: bool = True,
    show_progress: bool = True,
):
    """
    Predict sequence accessibility across cell types.

    Expected batch:
        {
            "input_ids": LongTensor,
            "attention_mask": LongTensor,
            optional "signal": FloatTensor
        }
    """

    device = get_device(device)

    model = model.to(device)
    model.eval()

    preds = []
    targets = []

    iterator = dataloader
    if show_progress:
        iterator = tqdm(dataloader, desc="Predicting sequence accessibility")

    for step, batch in enumerate(iterator):
        if step % 10 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with amp.autocast(
            device_type=device.type,
            enabled=use_amp and device.type == "cuda",
        ):
            pred = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        preds.append(pred.detach().cpu())

        if "signal" in batch:
            targets.append(batch["signal"].detach().cpu())

    out = {
        "preds": torch.cat(preds, dim=0),
    }

    if len(targets) > 0:
        out["targets"] = torch.cat(targets, dim=0)

    if return_numpy:
        out = {
            key: value.numpy()
            for key, value in out.items()
        }

    return out


def run_seq_attribution(
    model,
    tokenizer,
    sequence: str,
    target_cell_type: Union[int, str],
    cell_type_names: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
    max_length: int = 512,
    n_steps: int = 50,
    smooth_sigma: Optional[float] = 1.0,
    crop: Optional[Tuple[int, int]] = None,
) -> Dict[str, object]:
    """
    Run base-level integrated-gradient attribution for EpiZooSeq.

    Parameters
    ----------
    model:
        EpiZooSeq model.

    tokenizer:
        DNABERT tokenizer.

    sequence:
        Input DNA sequence.

    target_cell_type:
        Target cell type index or cell type name.

    cell_type_names:
        Required when target_cell_type is a string.

    max_length:
        Maximum tokenizer length.

    n_steps:
        Integrated gradients interpolation steps.

    smooth_sigma:
        Optional Gaussian smoothing sigma.
        If None or 0, no smoothing is applied.

    crop:
        Optional region to return from the full sequence.
        Format: (start, end), relative to the input sequence.

    Returns
    -------
    result:
        {
            "scores": cropped or full attribution scores,
            "sequence": cropped or full sequence,
            "full_scores": full attribution scores,
            "full_sequence": full sequence,
            "target_cell_type_idx": int,
        }
    """

    target_idx = resolve_cell_type_idx(
        target_cell_type=target_cell_type,
        cell_type_names=cell_type_names,
    )

    scores = compute_base_attributions(
        model=model,
        tokenizer=tokenizer,
        sequence=sequence,
        target_cell_type_idx=target_idx,
        device=device,
        max_length=max_length,
        n_steps=n_steps,
    )

    if smooth_sigma is not None and smooth_sigma > 0:
        scores = smooth_attribution_scores(
            scores=scores,
            sigma=smooth_sigma,
        )

    full_scores = scores
    full_sequence = sequence

    if crop is not None:
        start, end = crop
        scores = full_scores[start:end]
        sequence = full_sequence[start:end]

    return {
        "scores": scores,
        "sequence": sequence,
        "full_scores": full_scores,
        "full_sequence": full_sequence,
        "target_cell_type_idx": target_idx,
    }


def compute_base_attributions(
    model,
    tokenizer,
    sequence: str,
    target_cell_type_idx: int,
    device: Optional[str] = None,
    max_length: int = 512,
    n_steps: int = 50,
) -> np.ndarray:
    """
    Compute base-level integrated-gradient attribution scores.

    This follows the notebook logic:
        1. Tokenize sequence.
        2. Get DNABERT input embeddings.
        3. Run Integrated Gradients on input embeddings.
        4. Sum attribution over embedding dimension.
        5. Map token-level scores back to base-level scores.
    """

    try:
        from captum.attr import IntegratedGradients
    except ImportError as exc:
        raise ImportError(
            "`captum` is required for gradient attribution. "
            "Please install it with `pip install captum`."
        ) from exc

    device = get_device(device)

    model = model.to(device)
    model.eval()

    encoded = tokenizer(
        sequence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    backbone = model.seam.backbone
    embed_layer = backbone.get_input_embeddings()
    input_embeds = embed_layer(input_ids)

    def forward_from_embeds(inputs_embeds: torch.Tensor) -> torch.Tensor:
        batch_size = inputs_embeds.size(0)
        mask = attention_mask.expand(batch_size, -1)

        seq_emb = model.seam.forward_from_embeds(
            inputs_embeds=inputs_embeds,
            attention_mask=mask,
        )

        pred = _predict_one_cell_type(
            model=model,
            seq_emb=seq_emb,
            target_cell_type_idx=target_cell_type_idx,
        )

        return pred

    ig = IntegratedGradients(forward_from_embeds)

    baseline = torch.zeros_like(input_embeds)

    attributions = ig.attribute(
        inputs=input_embeds,
        baselines=baseline,
        n_steps=n_steps,
    )

    token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    base_scores = token_scores_to_base_scores(
        tokens=tokens,
        token_scores=token_scores,
        sequence=sequence,
    )

    return base_scores


def token_scores_to_base_scores(
    tokens: Sequence[str],
    token_scores: np.ndarray,
    sequence: str,
) -> np.ndarray:
    """
    Map token attribution scores back to base-level scores.

    This keeps the original behavior:
        each non-special token contributes its score to the bases it covers,
        using a left-to-right cursor.
    """

    base_scores = np.zeros(len(sequence), dtype=np.float32)

    cursor = 0

    special_tokens = {
        "[CLS]",
        "[SEP]",
        "[PAD]",
        "<pad>",
        "<unk>",
        "[UNK]",
    }

    for token, score in zip(tokens, token_scores):
        if token in special_tokens:
            continue

        clean_token = clean_dna_token(token)
        token_len = len(clean_token)

        if token_len == 0:
            continue

        if cursor + token_len > len(sequence):
            break

        base_scores[cursor: cursor + token_len] = score
        cursor += token_len

    return base_scores


def clean_dna_token(token: str) -> str:
    """
    Clean tokenizer-specific token prefixes.
    """

    token = token.replace(" ", "")
    token = token.replace("##", "")
    token = token.replace("Ġ", "")
    token = token.replace("▁", "")

    return token


def smooth_attribution_scores(
    scores,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Smooth attribution scores with Gaussian filter.
    """

    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(
        np.asarray(scores, dtype=np.float32),
        sigma=sigma,
    )


def resolve_cell_type_idx(
    target_cell_type: Union[int, str],
    cell_type_names: Optional[Sequence[str]] = None,
) -> int:
    """
    Resolve target cell type index.
    """

    if isinstance(target_cell_type, int):
        return target_cell_type

    if cell_type_names is None:
        raise ValueError(
            "`cell_type_names` is required when `target_cell_type` is a string."
        )

    if target_cell_type not in cell_type_names:
        raise KeyError(
            f"Cell type `{target_cell_type}` not found in `cell_type_names`."
        )

    return list(cell_type_names).index(target_cell_type)


def extract_top_motifs_for_meme(
    sequence: str,
    scores,
    top_k: int = 5,
    window_size: int = 15,
    mode: str = "activator",
) -> str:
    """
    Extract top attribution windows and return MEME-compatible FASTA text.

    Parameters
    ----------
    mode:
        "activator" uses positive scores.
        "repressor" uses negative scores.
    """

    scores = np.asarray(scores, dtype=np.float32)

    if mode not in {"activator", "repressor"}:
        raise ValueError("`mode` should be 'activator' or 'repressor'.")

    work_scores = scores.copy()

    if mode == "repressor":
        work_scores = -work_scores

    fasta = ""
    half = window_size // 2

    for rank in range(top_k):
        peak_idx = int(np.argmax(work_scores))
        max_score = work_scores[peak_idx]

        if max_score <= 0:
            break

        start = max(0, peak_idx - half)
        end = min(len(sequence), peak_idx + half + 1)

        motif_seq = list(sequence[start:end])
        window_scores = work_scores[start:end]

        for i, score in enumerate(window_scores):
            if score < 0:
                motif_seq[i] = "N"

        motif_seq = "".join(motif_seq)
        real_score = scores[peak_idx]

        fasta += f">Rank_{rank + 1}_Pos_{peak_idx}_Score_{real_score:.4f}\n"
        fasta += f"{motif_seq}\n"

        work_scores[start:end] = -np.inf

    return fasta


def _predict_one_cell_type(
    model,
    seq_emb: torch.Tensor,
    target_cell_type_idx: int,
) -> torch.Tensor:
    """
    Predict accessibility for one target cell type.
    """

    batch_size = seq_emb.size(0)

    cell_type_emb = model.cell_type_emb[target_cell_type_idx]
    cell_type_emb = cell_type_emb.unsqueeze(0).expand(batch_size, -1)

    fused = torch.cat(
        [seq_emb, cell_type_emb],
        dim=-1,
    )

    pred = model.head(fused).squeeze(-1)
    pred = model.out_act(pred)

    return pred