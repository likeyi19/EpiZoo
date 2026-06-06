# epizoo/inference/embeddings.py

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from torch import amp
from tqdm import tqdm
import numpy as np

from epizoo.inference.utils import get_device, get_input_ids


ArrayLike = Union[np.ndarray, torch.Tensor]


@torch.no_grad()
def extract_cell_embeddings(
    model,
    dataloader,
    device: Optional[str] = None,
    use_amp: bool = True,
    return_numpy: bool = True,
    show_progress: bool = True,
):
    """
    Extract cell embeddings from EpiZoo-style models.

    Supported dataloader batch formats:
        input_ids
        input_ids, species
        {"input_ids": input_ids, ...}

    Returns:
        embeddings with shape [n_cells, emb_dim]
    """

    device = get_device(device)

    model = model.to(device)
    model.eval()

    embeddings = []

    iterator = dataloader
    if show_progress:
        iterator = tqdm(dataloader, desc="Extracting cell embeddings")

    for step, batch in enumerate(iterator):
        if step % 10 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

        input_ids = get_input_ids(batch).to(device)

        with amp.autocast(
            device_type=device.type,
            enabled=use_amp and device.type == "cuda",
        ):
            outputs = model(
                input_ids=input_ids,
                return_transformer_out=False,
            )

        embeddings.append(outputs["cell_emb"].detach().cpu())

    embeddings = torch.cat(embeddings, dim=0)

    return embeddings.numpy() if return_numpy else embeddings


@torch.no_grad()
def extract_seq_embeddings(
    model,
    dataloader,
    device: Optional[str] = None,
    use_amp: bool = True,
    return_numpy: bool = True,
    show_progress: bool = True,
):
    """
    Extract sequence embeddings with SEAM.

    Expected dataloader batch from SEAMDataset + collate_fn_seam:
        {
            "input_ids": LongTensor [batch_size, seq_len],
            "attention_mask": LongTensor [batch_size, seq_len],
            "index": LongTensor [batch_size], optional
        }

    If `index` is provided, embeddings are reordered by index before returning.
    This is useful when DataLoader order is not guaranteed.

    Returns:
        embeddings with shape [n_sequences, emb_dim]
    """

    device = get_device(device)

    model = model.to(device)
    model.eval()

    embeddings = []
    indices = []

    iterator = dataloader
    if show_progress:
        iterator = tqdm(dataloader, desc="Extracting sequence embeddings")

    for step, batch in enumerate(iterator):
        if step % 10 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with amp.autocast(
            device_type=device.type,
            enabled=use_amp and device.type == "cuda",
        ):
            seq_emb = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        embeddings.append(seq_emb.detach().cpu())

        if "index" in batch:
            indices.append(batch["index"].detach().cpu())

    embeddings = torch.cat(embeddings, dim=0)

    if len(indices) > 0:
        indices = torch.cat(indices, dim=0)
        order = torch.argsort(indices)
        embeddings = embeddings[order]

    return embeddings.numpy() if return_numpy else embeddings


def compute_cell_type_embeddings(
    cell_embeddings: ArrayLike,
    labels: Sequence,
    label_order: Optional[Sequence] = None,
    return_labels: bool = False,
    return_numpy: bool = False,
):
    """
    Compute cell type embeddings by averaging cell embeddings within each label.

    Parameters
    ----------
    cell_embeddings:
        Cell embedding matrix with shape [num_cells, emb_dim].

    labels:
        Cell type labels with length num_cells.

    label_order:
        Optional ordered cell type labels.

        If provided, the output matrix follows this order.
        If None, labels are ordered by first appearance.

    return_labels:
        Whether to also return the label order.

    return_numpy:
        Whether to return numpy array instead of torch.Tensor.

    Returns
    -------
    cell_type_embeddings:
        Tensor or ndarray with shape [num_cell_types, emb_dim].

    label_order:
        Returned only when return_labels=True.
    """

    if isinstance(cell_embeddings, torch.Tensor):
        emb = cell_embeddings.detach().cpu().float()
    else:
        emb = torch.tensor(
            np.asarray(cell_embeddings),
            dtype=torch.float32,
        )

    labels = np.asarray(labels)

    if emb.ndim != 2:
        raise ValueError(
            "`cell_embeddings` should have shape [num_cells, emb_dim]."
        )

    if emb.shape[0] != len(labels):
        raise ValueError(
            "`cell_embeddings` and `labels` should have the same length. "
            f"Got {emb.shape[0]} and {len(labels)}."
        )

    if label_order is None:
        label_order = list(dict.fromkeys(labels.tolist()))
    else:
        label_order = list(label_order)

    ct_embs = []

    for label in label_order:
        idx = labels == label

        if not np.any(idx):
            raise ValueError(
                f"Label `{label}` was provided in `label_order`, "
                "but no cells have this label."
            )

        ct_emb = emb[idx].mean(dim=0)
        ct_embs.append(ct_emb)

    ct_embs = torch.stack(ct_embs, dim=0)

    if return_numpy:
        ct_embs = ct_embs.numpy()

    if return_labels:
        return ct_embs, label_order

    return ct_embs