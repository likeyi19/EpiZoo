# # epizoo/inference/signals.py

# from __future__ import annotations

# from typing import Optional

# import torch
# from torch import amp

# from epizoo.inference.utils import get_device


# @torch.no_grad()
# def predict_signals(
#     model,
#     dataloader,
#     device: Optional[str] = None,
#     use_amp: bool = True,
#     return_cell_emb: bool = True,
#     return_numpy: bool = True,
# ):
#     """
#     Predict reconstructed signals.

#     Expected dataloader batch:
#         input_ids, species

#     Returns:
#         {
#             "predicted_signals": array or tensor,
#             "cell_embeddings": array or tensor, optional
#         }
#     """

#     device = get_device(device)

#     model = model.to(device)
#     model.eval()

#     all_signals = []
#     all_cell_emb = [] if return_cell_emb else None

#     for step, batch in enumerate(dataloader):
#         if step % 10 == 0 and device.type == "cuda":
#             torch.cuda.empty_cache()

#         input_ids, species = batch
#         input_ids = input_ids.to(device)

#         with amp.autocast(
#             device_type=device.type,
#             enabled=use_amp and device.type == "cuda",
#         ):
#             outputs = model(input_ids=input_ids)
#             cell_emb = outputs["cell_emb"]
#             species_name = get_single_species(species)

#             predicted = model.predict_signal(
#                 cell_emb=cell_emb,
#                 species=species_name,
#             )

#         all_signals.append(predicted.detach().cpu())

#         if return_cell_emb:
#             all_cell_emb.append(cell_emb.detach().cpu())

#     predicted_signals = torch.cat(all_signals, dim=0)

#     outputs = {
#         "predicted_signals": predicted_signals.numpy()
#         if return_numpy
#         else predicted_signals
#     }

#     if return_cell_emb:
#         cell_embeddings = torch.cat(all_cell_emb, dim=0)
#         outputs["cell_embeddings"] = (
#             cell_embeddings.numpy()
#             if return_numpy
#             else cell_embeddings
#         )

#     return outputs


# def get_single_species(species):
#     """
#     Get the unique species from one batch.

#     Signal prediction currently assumes one species per batch.
#     """

#     if isinstance(species, torch.Tensor):
#         species = species.detach().cpu().tolist()

#     if isinstance(species, (int, str)):
#         return species

#     unique_species = list(dict.fromkeys(species))

#     if len(unique_species) != 1:
#         raise ValueError(
#             "Signal prediction expects one species per batch. "
#             f"Got mixed species: {unique_species}."
#         )

#     return unique_species[0]


# epizoo/inference/signals.py

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import torch
from torch import amp
from tqdm import tqdm

from epizoo.inference.utils import get_device


ArrayLike = Union[np.ndarray, torch.Tensor]


@torch.no_grad()
def predict_signals(
    model,
    dataloader,
    device: Optional[str] = None,
    use_amp: bool = True,
    apply_sigmoid: bool = False,
    delta_emb: Optional[ArrayLike] = None,
    return_cell_emb: bool = True,
    return_numpy: bool = True,
    show_progress: bool = True,
):
    """
    Predict reconstructed signals.

    Supports:
        1. EpiZoo / EpiZooDI
        2. EpiZooCancer
        3. EpiZooX

    Supported batch formats:
        (input_ids, species)
        (input_ids, species, cancer_type)
        dict with input_ids / species / cancer_type
    """

    device = get_device(device)

    model = model.to(device)
    model.eval()

    delta_emb = _prepare_delta_emb(delta_emb, device=device)

    all_signals = []
    all_cell_emb = [] if return_cell_emb else None

    iterator = dataloader
    if show_progress:
        iterator = tqdm(dataloader, desc="Predicting signals")

    for step, batch in enumerate(iterator):
        if step % 10 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

        batch = parse_signal_batch(batch)
        input_ids = batch["input_ids"].to(device)
        cancer_type = batch.get("cancer_type")

        if cancer_type is not None:
            cancer_type = cancer_type.to(device).long()

        species = get_single_species(batch.get("species"))

        with amp.autocast(
            device_type=device.type,
            enabled=use_amp and device.type == "cuda",
        ):
            outputs = _forward_model(
                model=model,
                input_ids=input_ids,
                cancer_type=cancer_type,
            )

            cell_emb = outputs["cell_emb"]

            if delta_emb is not None:
                cell_emb = cell_emb + delta_emb

            logits = decode_signals(
                model=model,
                cell_emb=cell_emb,
                species=species,
            )

            pred = torch.sigmoid(logits) if apply_sigmoid else logits

        all_signals.append(pred.detach().cpu())

        if return_cell_emb:
            all_cell_emb.append(cell_emb.detach().cpu())

    signals = torch.cat(all_signals, dim=0)

    out = {
        "predicted_signals": signals.numpy() if return_numpy else signals,
    }

    if return_cell_emb:
        cell_emb = torch.cat(all_cell_emb, dim=0)
        out["cell_embeddings"] = cell_emb.numpy() if return_numpy else cell_emb

    return out


@torch.no_grad()
def predict_signal_delta(
    model,
    dataloader,
    delta_emb: ArrayLike,
    device: Optional[str] = None,
    use_amp: bool = True,
    apply_sigmoid: bool = True,
    return_numpy: bool = False,
    show_progress: bool = True,
):
    """
    Predict reference and delta-perturbed signals.

    For each batch:
        cell_emb = model(input_ids, cancer_type_ids).cell_emb
        ref_pred = decoder(cell_emb)
        alt_pred = decoder(cell_emb + delta_emb)
        loa = alt_pred - ref_pred

    This is the common inference primitive for mutation LoA.
    """

    device = get_device(device)

    model = model.to(device)
    model.eval()

    delta_emb = _prepare_delta_emb(
        delta_emb=delta_emb,
        device=device,
    )

    ref_all = []
    alt_all = []

    iterator = dataloader
    if show_progress:
        iterator = tqdm(dataloader, desc="Predicting signal delta")

    for step, batch in enumerate(iterator):
        if step % 10 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

        batch = parse_signal_batch(batch)
        input_ids = batch["input_ids"].to(device)
        cancer_type = batch.get("cancer_type")

        if cancer_type is not None:
            cancer_type = cancer_type.to(device).long()

        species = get_single_species(batch.get("species"))

        with amp.autocast(
            device_type=device.type,
            enabled=use_amp and device.type == "cuda",
        ):
            outputs = _forward_model(
                model=model,
                input_ids=input_ids,
                cancer_type=cancer_type,
            )

            cell_emb = outputs["cell_emb"]

            ref_logits = decode_signals(
                model=model,
                cell_emb=cell_emb,
                species=species,
            )

            alt_logits = decode_signals(
                model=model,
                cell_emb=cell_emb + delta_emb,
                species=species,
            )

            if apply_sigmoid:
                ref_pred = torch.sigmoid(ref_logits)
                alt_pred = torch.sigmoid(alt_logits)
            else:
                ref_pred = ref_logits
                alt_pred = alt_logits

        ref_all.append(ref_pred.detach().cpu())
        alt_all.append(alt_pred.detach().cpu())

    ref_pred = torch.cat(ref_all, dim=0)
    alt_pred = torch.cat(alt_all, dim=0)
    loa = alt_pred - ref_pred

    out = {
        "ref_pred": ref_pred,
        "alt_pred": alt_pred,
        "loa": loa,
    }

    if return_numpy:
        out = {
            key: value.numpy()
            for key, value in out.items()
        }

    return out


def decode_signals(
    model,
    cell_emb: torch.Tensor,
    species: Optional[Union[int, str]] = None,
) -> torch.Tensor:
    """
    Decode cell embeddings into signal logits.

    Supports:
        1. model.predict_signal(cell_emb, species)
        2. model.signal_decoder with species-specific decoders
        3. model.signal_decoder as a single Linear decoder
        4. legacy model.signal_decoder / model.signal_decoder_mouse
    """

    species = normalize_species(species)

    if hasattr(model, "predict_signal"):
        return model.predict_signal(
            cell_emb=cell_emb,
            species=species,
        )

    decoder = model.signal_decoder

    if hasattr(decoder, "decoders"):
        if species is None:
            raise ValueError("`species` is required for species-specific decoder.")
        return decoder.decoders[species](cell_emb)

    return decoder(cell_emb)


def parse_signal_batch(batch) -> Dict:
    """
    Parse batch for signal inference.

    Supported formats:
        (input_ids, species)
        (input_ids, species, cancer_type)
        cancer training batch:
            input_ids, signals_human, signals_mouse, cca_ids, cca_labels, species, cancer_type
        dict:
            {"input_ids": ..., "species": ..., "cancer_type": ...}
    """

    if isinstance(batch, dict):
        return {
            "input_ids": batch["input_ids"],
            "species": batch.get("species"),
            "cancer_type": batch.get("cancer_type", batch.get("cancer_type_ids")),
        }

    if isinstance(batch, torch.Tensor):
        return {
            "input_ids": batch,
            "species": None,
            "cancer_type": None,
        }

    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            input_ids, species = batch
            return {
                "input_ids": input_ids,
                "species": species,
                "cancer_type": None,
            }

        if len(batch) == 3:
            input_ids, species, cancer_type = batch
            return {
                "input_ids": input_ids,
                "species": species,
                "cancer_type": cancer_type,
            }

        if len(batch) == 7:
            return {
                "input_ids": batch[0],
                "species": batch[5],
                "cancer_type": batch[6],
            }

    raise ValueError(
        "Unsupported batch format for signal prediction."
    )


def get_single_species(species):
    """
    Get the unique species from one batch.

    Returns:
        "human", "mouse", or None
    """

    if species is None:
        return None

    if isinstance(species, torch.Tensor):
        species = species.detach().cpu().tolist()

    if isinstance(species, (int, str)):
        return normalize_species(species)

    species = [
        normalize_species(x)
        for x in list(species)
    ]

    unique_species = list(dict.fromkeys(species))

    if len(unique_species) != 1:
        raise ValueError(
            "Signal prediction expects one species per batch. "
            f"Got mixed species: {unique_species}."
        )

    return unique_species[0]


def normalize_species(species):
    if species is None:
        return None

    if species in {0, "human", "Human", "HUMAN"}:
        return "human"

    if species in {1, "mouse", "Mouse", "MOUSE"}:
        return "mouse"

    raise ValueError(
        "`species` should be 0/'human', 1/'mouse', or None."
    )


def _forward_model(
    model,
    input_ids: torch.Tensor,
    cancer_type: Optional[torch.Tensor] = None,
):
    if cancer_type is not None:
        return model(
            input_ids=input_ids,
            cancer_type_ids=cancer_type,
            return_transformer_out=False,
        )

    return model(
        input_ids=input_ids,
        return_transformer_out=False,
    )


def _prepare_delta_emb(
    delta_emb: Optional[ArrayLike],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if delta_emb is None:
        return None

    if not isinstance(delta_emb, torch.Tensor):
        delta_emb = torch.tensor(delta_emb, dtype=torch.float32)

    delta_emb = delta_emb.to(device).float()

    if delta_emb.ndim == 1:
        delta_emb = delta_emb.unsqueeze(0)

    if delta_emb.ndim != 2 or delta_emb.size(0) != 1:
        raise ValueError(
            "`delta_emb` should have shape [emb_dim] or [1, emb_dim]."
        )

    return delta_emb