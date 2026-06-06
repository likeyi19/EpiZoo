# epizoo/models/transfer.py

from __future__ import annotations

from typing import Dict, Literal, Mapping, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from epizoo.data.ccre import get_joint_ccre_count


ArrayLike = Union[np.ndarray, torch.Tensor]
SourceSpecies = Optional[Literal["human", "mouse"]]


def transfer_epizoox_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    seq_embeddings: ArrayLike,
    num_ccres: Optional[int] = None,
    ccre_offset: int = 4,
) -> Dict[str, torch.Tensor]:
    """
    Transfer a pretrained EpiZoo-style state_dict to EpiZooX without cCRE mapping.

    This randomly initializes:
        - new cCRE embedding rows after special tokens
        - new signal decoder

    It copies:
        - all compatible backbone / rank / CCA parameters
        - special-token rows in ccre_emb and seq_emb
        - SEAM-computed seq embeddings into seq_emb[ccre_offset:]
    """

    source, seq_embeddings, num_ccres, emb_dim = _prepare_inputs(
        state_dict=state_dict,
        seq_embeddings=seq_embeddings,
        num_ccres=num_ccres,
    )

    _check_embedding_keys(source)

    vocab_size = num_ccres + ccre_offset
    new_state_dict = _remove_signal_decoder(source)

    new_state_dict["ccre_emb.weight"] = _init_ccre_emb(
        source_weight=source["ccre_emb.weight"],
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        ccre_offset=ccre_offset,
    )

    new_state_dict["seq_emb.weight"] = _init_seq_emb(
        source_weight=source["seq_emb.weight"],
        seq_embeddings=seq_embeddings,
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        ccre_offset=ccre_offset,
    )

    new_state_dict.update(
        _init_signal_decoder(
            emb_dim=emb_dim,
            num_ccres=num_ccres,
        )
    )

    return new_state_dict


def transfer_epizoox_state_dict_with_map(
    state_dict: Mapping[str, torch.Tensor],
    seq_embeddings: ArrayLike,
    ccre_map: Mapping[int, int],
    source_species: SourceSpecies = "human",
    num_ccres: Optional[int] = None,
    ccre_offset: int = 4,
    human_vocab_size: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Transfer a pretrained EpiZoo-style state_dict to EpiZooX with cCRE mapping.

    Parameters
    ----------
    state_dict:
        Source state_dict using the refactored EpiZoo naming.

    seq_embeddings:
        SEAM-computed sequence embeddings for the new species.
        Shape: [num_ccres, emb_dim].
        It should NOT include special-token embeddings.

    ccre_map:
        Mapping from new species cCRE index to reference species cCRE index.

        Format:
            {
                new_idx: ref_idx
            }

        Both indices are 0-based positions in their own cCRE lists.
        They do not include the special-token offset.

    source_species:
        Reference species in the source EpiZoo model.

        - "human":
            ccre_emb source row = ref_idx + 4
            decoder source row  = ref_idx

        - "mouse":
            ccre_emb source row = ref_idx + 4 + human_vocab_size
            decoder source row  = ref_idx

        - None:
            Source model is assumed to be EpiZooX-style single-vocabulary:
            ccre_emb source row = ref_idx + 4
            decoder source row  = ref_idx

    num_ccres:
        Number of new species cCREs.
        If None, inferred from seq_embeddings.shape[0].

    ccre_offset:
        Number of reserved special tokens.
        Default: 4.

    human_vocab_size:
        Required only when source_species="mouse" and it cannot be inferred
        from state_dict.

    Returns
    -------
    new_state_dict:
        State dict compatible with EpiZooX.
    """

    source, seq_embeddings, num_ccres, emb_dim = _prepare_inputs(
        state_dict=state_dict,
        seq_embeddings=seq_embeddings,
        num_ccres=num_ccres,
    )

    _check_embedding_keys(source)

    source_token_offset = _get_source_token_offset(
        state_dict=source,
        source_species=source_species,
        ccre_offset=ccre_offset,
        human_vocab_size=human_vocab_size,
    )

    decoder_weight_key, decoder_bias_key = _get_source_decoder_keys(
        source_species=source_species,
    )

    if decoder_weight_key not in source or decoder_bias_key not in source:
        raise KeyError(
            "Missing source signal decoder keys: "
            f"{decoder_weight_key}, {decoder_bias_key}"
        )

    vocab_size = num_ccres + ccre_offset
    new_state_dict = _remove_signal_decoder(source)

    ccre_weight = _init_ccre_emb(
        source_weight=source["ccre_emb.weight"],
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        ccre_offset=ccre_offset,
    )

    seq_weight = _init_seq_emb(
        source_weight=source["seq_emb.weight"],
        seq_embeddings=seq_embeddings,
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        ccre_offset=ccre_offset,
    )

    decoder_state = _init_signal_decoder(
        emb_dim=emb_dim,
        num_ccres=num_ccres,
    )

    new_idx, ref_idx = _parse_idx_map(ccre_map)

    _check_idx_range(
        name="new_idx",
        values=new_idx,
        upper=num_ccres,
    )

    _check_idx_range(
        name="ref_idx",
        values=ref_idx,
        upper=source[decoder_weight_key].shape[0],
    )

    source_emb_rows = ref_idx + source_token_offset
    target_emb_rows = new_idx + ccre_offset

    _check_idx_range(
        name="source embedding rows",
        values=source_emb_rows,
        upper=source["ccre_emb.weight"].shape[0],
    )

    ccre_weight[target_emb_rows] = source["ccre_emb.weight"][source_emb_rows]

    decoder_state["signal_decoder.weight"][new_idx] = source[decoder_weight_key][ref_idx]
    decoder_state["signal_decoder.bias"][new_idx] = source[decoder_bias_key][ref_idx]

    new_state_dict["ccre_emb.weight"] = ccre_weight
    new_state_dict["seq_emb.weight"] = seq_weight
    new_state_dict.update(decoder_state)

    return new_state_dict


def _prepare_inputs(
    state_dict: Mapping[str, torch.Tensor],
    seq_embeddings: ArrayLike,
    num_ccres: Optional[int],
):
    source = {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
    }

    seq_embeddings = _to_tensor(seq_embeddings)

    if seq_embeddings.ndim != 2:
        raise ValueError(
            "`seq_embeddings` should have shape [num_ccres, emb_dim]. "
            f"Got {tuple(seq_embeddings.shape)}."
        )

    inferred_num_ccres, emb_dim = seq_embeddings.shape

    if num_ccres is None:
        num_ccres = inferred_num_ccres

    if num_ccres != inferred_num_ccres:
        raise ValueError(
            "`num_ccres` should match seq_embeddings.shape[0]. "
            f"Got num_ccres={num_ccres}, seq_embeddings.shape[0]={inferred_num_ccres}."
        )

    return source, seq_embeddings, num_ccres, emb_dim


def _to_tensor(x: ArrayLike) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float()

    return torch.tensor(x, dtype=torch.float32)


def _check_embedding_keys(state_dict: Mapping[str, torch.Tensor]) -> None:
    for key in ("ccre_emb.weight", "seq_emb.weight"):
        if key not in state_dict:
            raise KeyError(f"Missing required key in state_dict: {key}")


def _remove_signal_decoder(
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("signal_decoder.")
    }


def _init_ccre_emb(
    source_weight: torch.Tensor,
    vocab_size: int,
    emb_dim: int,
    ccre_offset: int,
) -> torch.Tensor:
    emb = nn.Embedding(vocab_size, emb_dim)
    emb.weight.data[:ccre_offset] = source_weight[:ccre_offset]
    return emb.weight.detach().cpu().clone()


def _init_seq_emb(
    source_weight: torch.Tensor,
    seq_embeddings: torch.Tensor,
    vocab_size: int,
    emb_dim: int,
    ccre_offset: int,
) -> torch.Tensor:
    emb = nn.Embedding(vocab_size, emb_dim)
    emb.weight.data[:ccre_offset] = source_weight[:ccre_offset]
    emb.weight.data[ccre_offset:] = seq_embeddings
    return emb.weight.detach().cpu().clone()


def _init_signal_decoder(
    emb_dim: int,
    num_ccres: int,
) -> Dict[str, torch.Tensor]:
    decoder = nn.Linear(emb_dim, num_ccres)

    return {
        "signal_decoder.weight": decoder.weight.detach().cpu().clone(),
        "signal_decoder.bias": decoder.bias.detach().cpu().clone(),
    }


def _get_source_decoder_keys(
    source_species: SourceSpecies,
):
    if source_species is None:
        return "signal_decoder.weight", "signal_decoder.bias"

    if source_species not in {"human", "mouse"}:
        raise ValueError("`source_species` should be 'human', 'mouse', or None.")

    return (
        f"signal_decoder.decoders.{source_species}.weight",
        f"signal_decoder.decoders.{source_species}.bias",
    )


def _get_source_token_offset(
    state_dict: Mapping[str, torch.Tensor],
    source_species: SourceSpecies,
    ccre_offset: int,
    human_vocab_size: Optional[int],
) -> int:
    if source_species is None:
        return ccre_offset

    if source_species == "human":
        return ccre_offset

    if source_species == "mouse":
        if human_vocab_size is None:
            human_key = "signal_decoder.decoders.human.weight"

            if human_key not in state_dict:
                raise ValueError(
                    "`human_vocab_size` is required for mouse mapping when "
                    "`signal_decoder.decoders.human.weight` is not found."
                )

            human_vocab_size = state_dict[human_key].shape[0]

        return ccre_offset + int(human_vocab_size)

    raise ValueError("`source_species` should be 'human', 'mouse', or None.")


def _parse_idx_map(
    ccre_map: Mapping[int, int],
):
    if len(ccre_map) == 0:
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )

    new_idx = torch.tensor(
        list(ccre_map.keys()),
        dtype=torch.long,
    )

    ref_idx = torch.tensor(
        list(ccre_map.values()),
        dtype=torch.long,
    )

    return new_idx, ref_idx


def _check_idx_range(
    name: str,
    values: torch.Tensor,
    upper: int,
) -> None:
    if values.numel() == 0:
        return

    min_value = int(values.min())
    max_value = int(values.max())

    if min_value < 0 or max_value >= upper:
        raise IndexError(
            f"`{name}` out of range. "
            f"Expected values in [0, {upper}), got min={min_value}, max={max_value}."
        )
    

def transfer_epizoox_joint_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    new_seq_embeddings: ArrayLike,
    ref_to_joint: Mapping[int, int],
    new_to_joint: Mapping[int, int],
    source_species: SourceSpecies,
    ccre_offset: int = 4,
    human_vocab_size: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Transfer a base EpiZoo state_dict to a joint-vocabulary EpiZooX state_dict.

    This function is used when the reference species and the new species are
    merged into one joint cCRE vocabulary.

    Expected maps:
        ref_to_joint: {ref_idx: joint_token_id}
        new_to_joint: {new_idx: joint_token_id}

    Both ref_idx and new_idx are 0-based indices in their original cCRE lists.
    joint_token_id should already include `ccre_offset`.

    Transfer logic:
        1. Keep compatible backbone / rank / CCA parameters.
        2. Randomly initialize joint cCRE embedding.
        3. Randomly initialize joint signal decoder.
        4. Copy special-token cCRE and seq embeddings.
        5. Copy all reference-species cCRE / seq / decoder parameters
           into their joint vocabulary positions.
        6. Fill new-only cCRE seq embeddings with `new_seq_embeddings`.
           Their cCRE embedding and decoder rows remain randomly initialized.
        7. For overlap cCREs, reference parameters take priority.
    """

    source, new_seq_embeddings, new_num_ccres, emb_dim = _prepare_inputs(
        state_dict=state_dict,
        seq_embeddings=new_seq_embeddings,
        num_ccres=None,
    )

    _check_embedding_keys(source)

    decoder_weight_key, decoder_bias_key = _get_source_decoder_keys(
        source_species=source_species,
    )

    if decoder_weight_key not in source or decoder_bias_key not in source:
        raise KeyError(
            "Missing source signal decoder keys: "
            f"{decoder_weight_key}, {decoder_bias_key}"
        )

    source_token_offset = _get_source_token_offset(
        state_dict=source,
        source_species=source_species,
        ccre_offset=ccre_offset,
        human_vocab_size=human_vocab_size,
    )

    ref_num_ccres = source[decoder_weight_key].shape[0]

    _check_complete_idx_map(
        idx_map=ref_to_joint,
        expected_size=ref_num_ccres,
        name="ref_to_joint",
    )

    _check_complete_idx_map(
        idx_map=new_to_joint,
        expected_size=new_num_ccres,
        name="new_to_joint",
    )

    joint_num_ccres = get_joint_ccre_count(
        ref_to_joint=ref_to_joint,
        new_to_joint=new_to_joint,
        ccre_offset=ccre_offset,
    )

    joint_vocab_size = joint_num_ccres + ccre_offset

    new_state_dict = _remove_signal_decoder(source)

    ccre_weight = _init_ccre_emb(
        source_weight=source["ccre_emb.weight"],
        vocab_size=joint_vocab_size,
        emb_dim=emb_dim,
        ccre_offset=ccre_offset,
    )

    seq_weight = _init_seq_emb_for_joint(
        source_weight=source["seq_emb.weight"],
        vocab_size=joint_vocab_size,
        emb_dim=emb_dim,
        ccre_offset=ccre_offset,
    )

    decoder_state = _init_signal_decoder(
        emb_dim=emb_dim,
        num_ccres=joint_num_ccres,
    )

    _copy_reference_params_to_joint(
        source=source,
        ccre_weight=ccre_weight,
        seq_weight=seq_weight,
        decoder_state=decoder_state,
        ref_to_joint=ref_to_joint,
        source_token_offset=source_token_offset,
        decoder_weight_key=decoder_weight_key,
        decoder_bias_key=decoder_bias_key,
        ccre_offset=ccre_offset,
    )

    _copy_new_seq_embeddings_to_joint(
        seq_weight=seq_weight,
        new_seq_embeddings=new_seq_embeddings,
        new_to_joint=new_to_joint,
        ref_joint_tokens=set(ref_to_joint.values()),
        ccre_offset=ccre_offset,
    )

    new_state_dict["ccre_emb.weight"] = ccre_weight
    new_state_dict["seq_emb.weight"] = seq_weight
    new_state_dict.update(decoder_state)

    return new_state_dict


def _init_seq_emb_for_joint(
    source_weight: torch.Tensor,
    vocab_size: int,
    emb_dim: int,
    ccre_offset: int,
) -> torch.Tensor:
    """
    Initialize joint seq embedding.

    Special-token rows are copied from the source model.
    Other rows are randomly initialized first and will be filled later.
    """

    emb = nn.Embedding(vocab_size, emb_dim)
    emb.weight.data[:ccre_offset] = source_weight[:ccre_offset]

    return emb.weight.detach().cpu().clone()


def _copy_reference_params_to_joint(
    source: Mapping[str, torch.Tensor],
    ccre_weight: torch.Tensor,
    seq_weight: torch.Tensor,
    decoder_state: Dict[str, torch.Tensor],
    ref_to_joint: Mapping[int, int],
    source_token_offset: int,
    decoder_weight_key: str,
    decoder_bias_key: str,
    ccre_offset: int,
) -> None:
    """
    Copy reference-species parameters into joint vocabulary positions.

    This function modifies ccre_weight, seq_weight, and decoder_state in place.
    """

    ref_idx = torch.tensor(
        list(ref_to_joint.keys()),
        dtype=torch.long,
    )

    joint_token_ids = torch.tensor(
        list(ref_to_joint.values()),
        dtype=torch.long,
    )

    joint_signal_idx = joint_token_ids - ccre_offset
    source_token_ids = ref_idx + source_token_offset

    _check_idx_range(
        name="ref_idx",
        values=ref_idx,
        upper=source[decoder_weight_key].shape[0],
    )

    _check_idx_range(
        name="joint_token_ids",
        values=joint_token_ids,
        upper=ccre_weight.shape[0],
    )

    _check_idx_range(
        name="joint_signal_idx",
        values=joint_signal_idx,
        upper=decoder_state["signal_decoder.weight"].shape[0],
    )

    _check_idx_range(
        name="source_token_ids",
        values=source_token_ids,
        upper=source["ccre_emb.weight"].shape[0],
    )

    ccre_weight[joint_token_ids] = source["ccre_emb.weight"][source_token_ids]
    seq_weight[joint_token_ids] = source["seq_emb.weight"][source_token_ids]

    decoder_state["signal_decoder.weight"][joint_signal_idx] = source[decoder_weight_key][ref_idx]
    decoder_state["signal_decoder.bias"][joint_signal_idx] = source[decoder_bias_key][ref_idx]


def _copy_new_seq_embeddings_to_joint(
    seq_weight: torch.Tensor,
    new_seq_embeddings: torch.Tensor,
    new_to_joint: Mapping[int, int],
    ref_joint_tokens: set,
    ccre_offset: int,
) -> None:
    """
    Copy new-species SEAM embeddings into joint seq embedding.

    For overlap cCREs, reference-species seq embeddings are kept.
    For new-only cCREs, seq embeddings come from new_seq_embeddings.
    """

    new_idx = torch.tensor(
        list(new_to_joint.keys()),
        dtype=torch.long,
    )

    joint_token_ids = torch.tensor(
        list(new_to_joint.values()),
        dtype=torch.long,
    )

    _check_idx_range(
        name="new_idx",
        values=new_idx,
        upper=new_seq_embeddings.shape[0],
    )

    _check_idx_range(
        name="joint_token_ids",
        values=joint_token_ids,
        upper=seq_weight.shape[0],
    )

    for i, token_id in zip(new_idx.tolist(), joint_token_ids.tolist()):
        if token_id in ref_joint_tokens:
            continue

        seq_weight[token_id] = new_seq_embeddings[i]


def _check_complete_idx_map(
    idx_map: Mapping[int, int],
    expected_size: int,
    name: str,
) -> None:
    """
    Check whether a map contains exactly all indices [0, expected_size).
    """

    keys = set(int(x) for x in idx_map.keys())
    expected = set(range(expected_size))

    if keys != expected:
        missing = sorted(expected - keys)[:10]
        extra = sorted(keys - expected)[:10]

        raise ValueError(
            f"`{name}` should contain exactly indices [0, {expected_size}). "
            f"Missing examples: {missing}; extra examples: {extra}."
        )