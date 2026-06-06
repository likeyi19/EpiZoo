# epizoo/inference/annotation.py

from __future__ import annotations

from typing import Optional

import torch
from torch import amp

from epizoo.inference.utils import get_device, get_input_ids


@torch.no_grad()
def predict_cell_types(
    model,
    dataloader,
    device: Optional[str] = None,
    use_amp: bool = True,
    return_cell_emb: bool = True,
    return_numpy: bool = True,
):
    """
    Predict cell types with EpiZooAnno.

    Supported dataloader batch formats:
        input_ids
        input_ids, species
        {"input_ids": ...}

    Returns
    -------
    outputs:
        {
            "predicted_probabilities": array or tensor,
            "predicted_labels": array or tensor,
            "cell_embeddings": array or tensor, optional
        }
    """

    device = get_device(device)

    model = model.to(device)
    model.eval()

    all_probs = []
    all_labels = []
    all_cell_emb = [] if return_cell_emb else None

    for step, batch in enumerate(dataloader):
        if step % 10 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

        input_ids = _get_input_ids(batch).to(device)

        with amp.autocast(
            device_type=device.type,
            enabled=use_amp and device.type == "cuda",
        ):
            outputs = model(
                input_ids=input_ids,
                return_cell_emb=return_cell_emb,
                return_transformer_out=False,
            )

            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)
            labels = probs.argmax(dim=1)

        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())

        if return_cell_emb:
            all_cell_emb.append(outputs["cell_emb"].detach().cpu())

    predicted_probabilities = torch.cat(all_probs, dim=0)
    predicted_labels = torch.cat(all_labels, dim=0)

    outputs = {
        "predicted_probabilities": (
            predicted_probabilities.numpy()
            if return_numpy
            else predicted_probabilities
        ),
        "predicted_labels": (
            predicted_labels.numpy()
            if return_numpy
            else predicted_labels
        ),
    }

    if return_cell_emb:
        cell_embeddings = torch.cat(all_cell_emb, dim=0)
        outputs["cell_embeddings"] = (
            cell_embeddings.numpy()
            if return_numpy
            else cell_embeddings
        )

    return outputs
