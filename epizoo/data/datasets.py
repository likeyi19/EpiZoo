# epizoo/data/datasets.py

from __future__ import annotations

import json
from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from epizoo.data.ccre import get_joint_ccre_count


PAD_TOKEN_ID = 0
CLS_TOKEN_ID = 1
SEP_TOKEN_ID = 2
CCRE_TOKEN_OFFSET = 4


def parse_cell_sentence(cell_sentence: Union[str, Sequence[int]]) -> List[int]:
    """
    Parse one cell sentence into a list of cCRE token ids.
    """

    if isinstance(cell_sentence, str):
        try:
            cell_sentence = json.loads(cell_sentence)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                "Error parsing cell_sentences. "
                "Ensure all string elements are JSON-serializable lists."
            ) from exc

    if not isinstance(cell_sentence, list):
        cell_sentence = list(cell_sentence)

    if not all(isinstance(x, (int, np.integer)) for x in cell_sentence):
        raise ValueError(
            "Invalid format for cell_sentences. "
            "Each element must be a list of integers."
        )

    return [int(x) for x in cell_sentence]


def truncate_cell(
    cell: np.ndarray,
    max_length: int,
    random_sample: bool,
) -> np.ndarray:
    """
    Truncate or randomly sample cCRE tokens.

    max_length includes [CLS] and [SEP], so the maximum number of cCRE tokens
    is max_length - 2.
    """

    max_tokens = max_length - 2

    if len(cell) <= max_tokens:
        return cell

    if random_sample:
        sampled_idx = np.sort(
            np.random.choice(
                len(cell),
                size=max_tokens,
                replace=False,
            )
        )
        return cell[sampled_idx]

    return cell[:max_tokens]


def add_special_tokens(cell: Sequence[int]) -> torch.Tensor:
    """
    Add [CLS] and [SEP] tokens.
    """

    input_ids = [CLS_TOKEN_ID] + list(cell) + [SEP_TOKEN_ID]
    return torch.tensor(input_ids, dtype=torch.long)


def pad_input_ids(
    input_ids: Sequence[torch.Tensor],
    pad_token_id: int = PAD_TOKEN_ID,
) -> torch.Tensor:
    """
    Pad input ids to the maximum sequence length in the batch.
    """

    max_len = max(x.size(0) for x in input_ids)

    return torch.stack([
        F.pad(
            x,
            pad=(0, max_len - x.size(0)),
            mode="constant",
            value=pad_token_id,
        )
        for x in input_ids
    ])


class CellDataset(Dataset):
    """
    Dataset for EpiZoo training.

    For each cell, this dataset generates:
        1. input_ids with [CLS] and [SEP]
        2. species-specific binary signal vector
        3. sampled cCRE ids for CCA
        4. CCA accessibility labels
        5. species id

    Token convention:
        0 = [PAD]
        1 = [CLS]
        2 = [SEP]
        cCRE token ids start from 4

    Species convention:
        0 = human
        1 = mouse
    """

    def __init__(
        self,
        cell_sentences,
        species,
        max_length: int = 8192,
        cca_alpha: int = 1,
        human_num_ccres: int = 1_355_445,
        mouse_num_ccres: int = 1_341_077,
        random_sample: bool = True,
    ):
        self.cell_sentences = [
            parse_cell_sentence(x) for x in list(cell_sentences)
        ]
        self.species = list(species)

        self.max_length = max_length
        self.cca_alpha = cca_alpha
        self.human_num_ccres = human_num_ccres
        self.mouse_num_ccres = mouse_num_ccres
        self.random_sample = random_sample

        if len(self.cell_sentences) != len(self.species):
            raise ValueError(
                "`cell_sentences` and `species` must have the same length. "
                f"Got {len(self.cell_sentences)} and {len(self.species)}."
            )

    def __len__(self) -> int:
        return len(self.cell_sentences)

    def __getitem__(self, idx: int):
        cell = np.asarray(self.cell_sentences[idx], dtype=int)
        species = self.species[idx]

        signal, inaccessible_ids = self._build_signal(cell, species)

        cell = truncate_cell(
            cell=cell,
            max_length=self.max_length,
            random_sample=self.random_sample,
        )

        input_ids = add_special_tokens(cell)
        cca_ids, cca_labels = self._build_cca_samples(
            accessible_ids=cell,
            inaccessible_ids=inaccessible_ids,
        )

        return {
            "input_ids": input_ids,
            "signal": torch.tensor(signal, dtype=torch.float32),
            "cca_ids": torch.tensor(cca_ids, dtype=torch.long),
            "cca_labels": torch.tensor(cca_labels, dtype=torch.float32),
            "species": species,
        }

    def _build_signal(
        self,
        cell: np.ndarray,
        species: int,
    ):
        """
        Build signal vector and inaccessible cCRE ids.

        This follows the original indexing logic:
            human adjusted id = token_id - 4
            mouse adjusted id = token_id - 4 - human_num_ccres
        """

        if species == 0:
            num_ccres = self.human_num_ccres
            token_offset = CCRE_TOKEN_OFFSET

        elif species == 1:
            num_ccres = self.mouse_num_ccres
            token_offset = CCRE_TOKEN_OFFSET + self.human_num_ccres

        else:
            raise ValueError("Species should be 0 or 1. 0 for human, 1 for mouse.")

        adjusted_ids = cell - token_offset

        signal = np.zeros(num_ccres, dtype=np.float32)
        inaccessible_mask = np.ones(num_ccres, dtype=bool)

        signal[adjusted_ids] = 1.0
        inaccessible_mask[adjusted_ids] = False

        inaccessible_ids = np.where(inaccessible_mask)[0] + token_offset

        return signal, inaccessible_ids

    def _build_cca_samples(
        self,
        accessible_ids: np.ndarray,
        inaccessible_ids: np.ndarray,
    ):
        """
        Build CCA positive and negative samples.

        Positive samples:
            accessible cCREs in the truncated cell sentence

        Negative samples:
            sampled inaccessible cCREs

        Original behavior:
            num_negatives = int(len(accessible_ids) * cca_alpha)
        """

        num_negatives = int(len(accessible_ids) * self.cca_alpha)

        if num_negatives <= len(inaccessible_ids):
            sampled_inaccessible_ids = np.random.choice(
                inaccessible_ids,
                size=num_negatives,
                replace=False,
            )
        else:
            sampled_inaccessible_ids = np.random.choice(
                inaccessible_ids,
                size=num_negatives,
                replace=True,
            )

        cca_ids = np.concatenate([
            accessible_ids,
            sampled_inaccessible_ids,
        ])

        cca_labels = np.concatenate([
            np.ones_like(accessible_ids),
            np.zeros_like(sampled_inaccessible_ids),
        ])

        return cca_ids, cca_labels


def collate_fn(batch):
    """
    Collate function for EpiZoo training.

    It only packs samples generated by CellDataset.

    Returns:
        input_ids:
            LongTensor [batch_size, max_seq_len]

        signals_human:
            FloatTensor [num_human_cells, human_num_ccres] or None

        signals_mouse:
            FloatTensor [num_mouse_cells, mouse_num_ccres] or None

        cca_ids:
            List[LongTensor], one tensor per cell

        cca_labels:
            FloatTensor [total_num_cca_samples]

        species:
            List[int]
    """

    input_ids = [item["input_ids"] for item in batch]
    signals = [item["signal"] for item in batch]
    cca_ids = [item["cca_ids"] for item in batch]
    cca_labels = [item["cca_labels"] for item in batch]
    species = [item["species"] for item in batch]

    input_ids = pad_input_ids(input_ids, pad_token_id=PAD_TOKEN_ID)

    species_array = np.asarray(species)
    human_idx = np.where(species_array == 0)[0]
    mouse_idx = np.where(species_array == 1)[0]

    signals_human = (
        torch.stack([signals[i] for i in human_idx])
        if len(human_idx) > 0
        else None
    )

    signals_mouse = (
        torch.stack([signals[i] for i in mouse_idx])
        if len(mouse_idx) > 0
        else None
    )

    cca_labels = torch.cat(cca_labels)

    return (
        input_ids,
        signals_human,
        signals_mouse,
        cca_ids,
        cca_labels,
        list(species),
    )


class InferenceCellDataset(Dataset):
    """
    Dataset for EpiZoo inference/test.

    It only returns:
        1. input_ids with [CLS] and [SEP]
        2. species id

    No signal reconstruction target or CCA samples are generated.
    """

    def __init__(
        self,
        cell_sentences,
        species,
        max_length: int = 8192,
        random_sample: bool = True,
    ):
        self.cell_sentences = [
            parse_cell_sentence(x) for x in list(cell_sentences)
        ]
        self.species = list(species)

        self.max_length = max_length
        self.random_sample = random_sample

        if len(self.cell_sentences) != len(self.species):
            raise ValueError(
                "`cell_sentences` and `species` must have the same length. "
                f"Got {len(self.cell_sentences)} and {len(self.species)}."
            )

    def __len__(self) -> int:
        return len(self.cell_sentences)

    def __getitem__(self, idx: int):
        cell = np.asarray(self.cell_sentences[idx], dtype=int)
        species = self.species[idx]

        cell = truncate_cell(
            cell=cell,
            max_length=self.max_length,
            random_sample=self.random_sample,
        )

        input_ids = add_special_tokens(cell)

        return input_ids, species


def inference_collate_fn(batch):
    """
    Collate function for inference/test.

    Returns:
        input_ids:
            LongTensor [batch_size, max_seq_len]

        species:
            List[int]
    """

    input_ids, species = zip(*batch)

    input_ids = pad_input_ids(input_ids, pad_token_id=PAD_TOKEN_ID)

    return input_ids, list(species)


class CellDatasetDI(Dataset):
    """
    Dataset for EpiZoo data imputation.

    Difference from CellDataset:
        - CellDataset builds binary 0/1 signal targets.
        - CellDatasetDI uses continuous TF-IDF signals from adata.X.

    This dataset currently assumes a single species for the whole dataset,
    matching the original DI implementation.

    Token convention:
        0 = [PAD]
        1 = [CLS]
        2 = [SEP]
        cCRE token ids start from 4

    Species convention:
        0 / "Human" / "human" = human
        1 / "Mouse" / "mouse" = mouse
    """

    def __init__(
        self,
        adata,
        cell_sentences,
        species,
        max_length: int = 8192,
        cca_alpha: float = 1,
        human_num_ccres: int = 1_355_445,
        mouse_num_ccres: int = 1_341_077,
        random_sample: bool = True,
    ):
        self.adata = adata.copy()
        self.cell_sentences = [
            parse_cell_sentence(x) for x in list(cell_sentences)
        ]

        self.species = self._normalize_species(species)

        self.max_length = max_length
        self.cca_alpha = cca_alpha
        self.human_num_ccres = human_num_ccres
        self.mouse_num_ccres = mouse_num_ccres
        self.random_sample = random_sample

        if len(self.cell_sentences) != self.adata.n_obs:
            raise ValueError(
                "`cell_sentences` length must match `adata.n_obs`. "
                f"Got {len(self.cell_sentences)} and {self.adata.n_obs}."
            )

    def __len__(self) -> int:
        return len(self.cell_sentences)

    def __getitem__(self, idx: int):
        cell = np.asarray(self.cell_sentences[idx], dtype=int)
        species = self.species

        signal = self._get_signal(idx)
        inaccessible_ids = self._get_inaccessible_ids(
            cell=cell,
            species=species,
        )

        cell = truncate_cell(
            cell=cell,
            max_length=self.max_length,
            random_sample=self.random_sample,
        )

        input_ids = add_special_tokens(cell)

        cca_ids, cca_labels = self._build_cca_samples(
            accessible_ids=cell,
            inaccessible_ids=inaccessible_ids,
        )

        return {
            "input_ids": input_ids,
            "signal": torch.tensor(signal, dtype=torch.float32),
            "cca_ids": torch.tensor(cca_ids, dtype=torch.long),
            "cca_labels": torch.tensor(cca_labels, dtype=torch.float32),
            "species": species,
        }

    def _get_signal(self, idx: int) -> np.ndarray:
        """
        Get continuous TF-IDF signal from adata.X.
        """

        x = self.adata[idx].X

        if hasattr(x, "toarray"):
            x = x.toarray()

        return np.asarray(x, dtype=np.float32).reshape(-1)

    def _get_inaccessible_ids(
        self,
        cell: np.ndarray,
        species: int,
    ) -> np.ndarray:
        """
        Get inaccessible cCRE token ids for CCA negative sampling.

        This follows the original DI indexing logic:
            human adjusted id = token_id - 4
            mouse adjusted id = token_id - 4 - human_num_ccres
        """

        if species == 0:
            num_ccres = self.human_num_ccres
            token_offset = CCRE_TOKEN_OFFSET

        elif species == 1:
            num_ccres = self.mouse_num_ccres
            token_offset = CCRE_TOKEN_OFFSET + self.human_num_ccres

        else:
            raise ValueError("Species should be 0 or 1.")

        adjusted_ids = cell - token_offset

        inaccessible_mask = np.ones(num_ccres, dtype=bool)
        inaccessible_mask[adjusted_ids] = False

        return np.where(inaccessible_mask)[0] + token_offset

    def _build_cca_samples(
        self,
        accessible_ids: np.ndarray,
        inaccessible_ids: np.ndarray,
    ):
        """
        Build CCA positive and negative samples.
        """

        num_negatives = int(len(accessible_ids) * self.cca_alpha)

        if num_negatives <= len(inaccessible_ids):
            sampled_inaccessible_ids = np.random.choice(
                inaccessible_ids,
                size=num_negatives,
                replace=False,
            )
        else:
            sampled_inaccessible_ids = np.random.choice(
                inaccessible_ids,
                size=num_negatives,
                replace=True,
            )

        cca_ids = np.concatenate([
            accessible_ids,
            sampled_inaccessible_ids,
        ])

        cca_labels = np.concatenate([
            np.ones_like(accessible_ids),
            np.zeros_like(sampled_inaccessible_ids),
        ])

        return cca_ids, cca_labels

    @staticmethod
    def _normalize_species(species) -> int:
        """
        Normalize species input to integer id.
        """

        if species in {0, "human", "Human"}:
            return 0

        if species in {1, "mouse", "Mouse"}:
            return 1

        raise ValueError(
            "`species` should be 0, 1, 'Human', 'Mouse', 'human', or 'mouse'."
        )
    

class CellDatasetAnno(Dataset):
    """
    Dataset for EpiZoo cell type annotation.

    Each item returns:
        input_ids: LongTensor
        label: LongTensor

    Token convention:
        0 = [PAD]
        1 = [CLS]
        2 = [SEP]
    """

    def __init__(
        self,
        cell_sentences,
        labels,
        max_length: int = 8192,
        random_sample: bool = True,
    ):
        self.cell_sentences = [
            parse_cell_sentence(x) for x in list(cell_sentences)
        ]
        self.labels = list(labels)

        self.max_length = max_length
        self.random_sample = random_sample

        if len(self.cell_sentences) != len(self.labels):
            raise ValueError(
                "`cell_sentences` and `labels` must have the same length. "
                f"Got {len(self.cell_sentences)} and {len(self.labels)}."
            )

    def __len__(self) -> int:
        return len(self.cell_sentences)

    def __getitem__(self, idx: int):
        cell = np.asarray(self.cell_sentences[idx], dtype=int)
        label = self.labels[idx]

        cell = truncate_cell(
            cell=cell,
            max_length=self.max_length,
            random_sample=self.random_sample,
        )

        input_ids = add_special_tokens(cell)
        label = torch.tensor(label, dtype=torch.long)

        return input_ids, label


def collate_fn_anno(batch):
    """
    Collate function for cell type annotation.

    Returns:
        input_ids:
            LongTensor [batch_size, max_seq_len]

        labels:
            LongTensor [batch_size]
    """

    input_ids, labels = zip(*batch)

    input_ids = pad_input_ids(
        input_ids,
        pad_token_id=PAD_TOKEN_ID,
    )

    labels = torch.stack(labels).long()

    return input_ids, labels


class CellDatasetX(Dataset):
    """
    Dataset for EpiZooX training.

    This is the single-vocabulary version of CellDataset.

    For each cell, it generates:
        1. input_ids with [CLS] and [SEP]
        2. binary signal reconstruction target
        3. sampled cCRE ids for CCA
        4. CCA accessibility labels

    Token convention:
        0 = [PAD]
        1 = [CLS]
        2 = [SEP]
        cCRE token ids start from 4
    """

    def __init__(
        self,
        cell_sentences,
        max_length: int = 8192,
        cca_alpha: float = 1.0,
        num_ccres: int = 1_355_445,
        random_sample: bool = True,
    ):
        self.cell_sentences = [
            parse_cell_sentence(x)
            for x in list(cell_sentences)
        ]

        self.max_length = max_length
        self.cca_alpha = cca_alpha
        self.num_ccres = num_ccres
        self.random_sample = random_sample

    def __len__(self) -> int:
        return len(self.cell_sentences)

    def __getitem__(self, idx: int):
        cell = np.asarray(
            self.cell_sentences[idx],
            dtype=int,
        )

        signal, inaccessible_ids = self._build_signal(cell)

        # Keep original behavior:
        # signal and inaccessible ids are built from the full cell sentence first,
        # then the transformer input and CCA positives are truncated/sampled.
        cell = truncate_cell(
            cell=cell,
            max_length=self.max_length,
            random_sample=self.random_sample,
        )

        input_ids = add_special_tokens(cell)

        cca_ids, cca_labels = self._build_cca_samples(
            accessible_ids=cell,
            inaccessible_ids=inaccessible_ids,
        )

        return {
            "input_ids": input_ids,
            "signal": torch.tensor(signal, dtype=torch.float32),
            "cca_ids": torch.tensor(cca_ids, dtype=torch.long),
            "cca_labels": torch.tensor(cca_labels, dtype=torch.float32),
        }

    def _build_signal(self, cell: np.ndarray):
        """
        Build binary signal vector and inaccessible cCRE ids.

        Original indexing logic:
            adjusted_id = token_id - 4
        """

        adjusted_ids = cell - CCRE_TOKEN_OFFSET

        signal = np.zeros(self.num_ccres, dtype=np.float32)
        inaccessible_mask = np.ones(self.num_ccres, dtype=bool)

        signal[adjusted_ids] = 1.0
        inaccessible_mask[adjusted_ids] = False

        inaccessible_ids = np.where(inaccessible_mask)[0] + CCRE_TOKEN_OFFSET

        return signal, inaccessible_ids

    def _build_cca_samples(
        self,
        accessible_ids: np.ndarray,
        inaccessible_ids: np.ndarray,
    ):
        """
        Build CCA positive and negative samples.

        Original behavior:
            num_negatives = int(len(accessible_ids) * cca_alpha)
        """

        num_negatives = int(len(accessible_ids) * self.cca_alpha)

        if num_negatives <= len(inaccessible_ids):
            sampled_inaccessible_ids = np.random.choice(
                inaccessible_ids,
                size=num_negatives,
                replace=False,
            )
        else:
            sampled_inaccessible_ids = np.random.choice(
                inaccessible_ids,
                size=num_negatives,
                replace=True,
            )

        cca_ids = np.concatenate([
            accessible_ids,
            sampled_inaccessible_ids,
        ])

        cca_labels = np.concatenate([
            np.ones_like(accessible_ids),
            np.zeros_like(sampled_inaccessible_ids),
        ])

        return cca_ids, cca_labels


def collate_fn_x(batch):
    """
    Collate function for EpiZooX training.

    Returns:
        input_ids:
            LongTensor [batch_size, max_seq_len]

        signals:
            FloatTensor [batch_size, num_ccres]

        cca_ids:
            List[LongTensor], one tensor per cell

        cca_labels:
            FloatTensor [total_num_cca_samples]
    """

    input_ids = [item["input_ids"] for item in batch]
    signals = [item["signal"] for item in batch]
    cca_ids = [item["cca_ids"] for item in batch]
    cca_labels = [item["cca_labels"] for item in batch]

    input_ids = pad_input_ids(
        input_ids,
        pad_token_id=PAD_TOKEN_ID,
    )

    signals = torch.stack(signals)
    cca_labels = torch.cat(cca_labels)

    return (
        input_ids,
        signals,
        cca_ids,
        cca_labels,
    )


class InferenceCellDatasetX(Dataset):
    """
    Dataset for EpiZooX inference.

    It only returns input_ids with [CLS] and [SEP].
    """

    def __init__(
        self,
        cell_sentences,
        max_length: int = 8192,
        random_sample: bool = True,
    ):
        self.cell_sentences = [
            parse_cell_sentence(x)
            for x in list(cell_sentences)
        ]

        self.max_length = max_length
        self.random_sample = random_sample

    def __len__(self) -> int:
        return len(self.cell_sentences)

    def __getitem__(self, idx: int):
        cell = np.asarray(
            self.cell_sentences[idx],
            dtype=int,
        )

        cell = truncate_cell(
            cell=cell,
            max_length=self.max_length,
            random_sample=self.random_sample,
        )

        return add_special_tokens(cell)


def inference_collate_fn_x(batch):
    """
    Collate function for EpiZooX inference.

    Returns:
        input_ids:
            LongTensor [batch_size, max_seq_len]
    """

    return pad_input_ids(
        batch,
        pad_token_id=PAD_TOKEN_ID,
    )


class SEAMDataset(Dataset):
    """
    Dataset for tokenizing DNA sequences with DNABERT tokenizer.

    It can be used for:
        1. SEAM embedding extraction:
            sequences only

        2. EpiZooSeq training:
            sequences + signals
    """

    def __init__(
        self,
        sequences,
        dnabert_path: str,
        signals=None,
        max_length: int = 512,
        trust_remote_code: bool = True,
        return_index: bool = False,
    ):
        self.sequences = list(sequences)
        self.max_length = max_length
        self.return_index = return_index

        self.signals = None
        if signals is not None:
            self.signals = np.asarray(signals, dtype=np.float32)

            if len(self.signals) != len(self.sequences):
                raise ValueError(
                    "`signals` and `sequences` must have the same length. "
                    f"Got {len(self.signals)} and {len(self.sequences)}."
                )

        self.tokenizer = AutoTokenizer.from_pretrained(
            dnabert_path,
            trust_remote_code=trust_remote_code,
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        encoded = self.tokenizer(
            self.sequences[idx],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0).long(),
            "attention_mask": encoded["attention_mask"].squeeze(0).long(),
        }

        if self.signals is not None:
            item["signal"] = torch.tensor(
                self.signals[idx],
                dtype=torch.float32,
            )

        if self.return_index:
            item["index"] = torch.tensor(idx, dtype=torch.long)

        return item


def collate_fn_seam(batch):
    """
    Collate function for SEAMDataset.

    It pads tokenized DNA sequences and optionally stacks signals.
    """

    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]

    out = {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=PAD_TOKEN_ID,
        ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        ),
    }

    if "signal" in batch[0]:
        out["signal"] = torch.stack([
            item["signal"] for item in batch
        ])

    if "index" in batch[0]:
        out["index"] = torch.stack([
            item["index"] for item in batch
        ])

    return out


class CellDatasetJoint(Dataset):
    """
    Dataset for joint-vocabulary EpiZooX training.

    Difference from CellDatasetX:
        - input cell sentences are first mapped to joint-vocabulary token ids.
        - species is used only internally to choose ref_to_joint or new_to_joint.
        - species is not returned.

    Expected maps:
        ref_to_joint: {ref_idx: joint_token_id}
        new_to_joint: {new_idx: joint_token_id}

    Both ref_idx and new_idx are 0-based cCRE indices in their original
    cCRE lists. joint_token_id should already include CCRE_TOKEN_OFFSET.
    """

    def __init__(
        self,
        cell_sentences,
        species,
        ref_to_joint,
        new_to_joint,
        max_length: int = 8192,
        cca_alpha: float = 1.0,
        random_sample: bool = True,
    ):
        self.cell_sentences = [
            parse_cell_sentence(x)
            for x in list(cell_sentences)
        ]

        self.species = self._expand_species(
            species=species,
            n=len(self.cell_sentences),
        )

        self.ref_to_joint = {
            int(k): int(v)
            for k, v in ref_to_joint.items()
        }
        self.new_to_joint = {
            int(k): int(v)
            for k, v in new_to_joint.items()
        }

        self.max_length = max_length
        self.cca_alpha = cca_alpha
        self.random_sample = random_sample

        self.num_ccres = get_joint_ccre_count(
            ref_to_joint=self.ref_to_joint,
            new_to_joint=self.new_to_joint,
            ccre_offset=CCRE_TOKEN_OFFSET,
            ids_include_offset=True,
        )

    def __len__(self) -> int:
        return len(self.cell_sentences)

    def __getitem__(self, idx: int):
        cell = np.asarray(
            self.cell_sentences[idx],
            dtype=int,
        )

        joint_map = self._get_joint_map(self.species[idx])
        cell = self._map_cell_to_joint(
            cell=cell,
            joint_map=joint_map,
        )

        signal, inaccessible_ids = self._build_signal(cell)

        cell = truncate_cell(
            cell=cell,
            max_length=self.max_length,
            random_sample=self.random_sample,
        )

        input_ids = add_special_tokens(cell)

        cca_ids, cca_labels = self._build_cca_samples(
            accessible_ids=cell,
            inaccessible_ids=inaccessible_ids,
        )

        return {
            "input_ids": input_ids,
            "signal": torch.tensor(signal, dtype=torch.float32),
            "cca_ids": torch.tensor(cca_ids, dtype=torch.long),
            "cca_labels": torch.tensor(cca_labels, dtype=torch.float32),
        }

    def _map_cell_to_joint(
        self,
        cell: np.ndarray,
        joint_map,
    ) -> np.ndarray:
        """
        Map original cCRE token ids to joint-vocabulary token ids.
        """

        ccre_idx = cell - CCRE_TOKEN_OFFSET

        if np.any(ccre_idx < 0):
            raise ValueError(
                "Cell sentence should contain cCRE token ids with offset. "
                f"Found token ids smaller than {CCRE_TOKEN_OFFSET}."
            )

        try:
            mapped = [
                joint_map[int(idx)]
                for idx in ccre_idx
            ]
        except KeyError as exc:
            raise KeyError(
                f"cCRE index {exc.args[0]} not found in joint map. "
                "Please make sure ref_to_joint/new_to_joint is complete."
            ) from exc

        return np.asarray(mapped, dtype=int)

    def _build_signal(self, cell: np.ndarray):
        """
        Build binary signal vector and inaccessible cCRE token ids
        in the joint vocabulary.
        """

        adjusted_ids = cell - CCRE_TOKEN_OFFSET

        signal = np.zeros(self.num_ccres, dtype=np.float32)
        inaccessible_mask = np.ones(self.num_ccres, dtype=bool)

        signal[adjusted_ids] = 1.0
        inaccessible_mask[adjusted_ids] = False

        inaccessible_ids = np.where(inaccessible_mask)[0] + CCRE_TOKEN_OFFSET

        return signal, inaccessible_ids

    def _build_cca_samples(
        self,
        accessible_ids: np.ndarray,
        inaccessible_ids: np.ndarray,
    ):
        """
        Build CCA positive and negative samples.
        """

        num_negatives = int(len(accessible_ids) * self.cca_alpha)

        if num_negatives <= len(inaccessible_ids):
            sampled_inaccessible_ids = np.random.choice(
                inaccessible_ids,
                size=num_negatives,
                replace=False,
            )
        else:
            sampled_inaccessible_ids = np.random.choice(
                inaccessible_ids,
                size=num_negatives,
                replace=True,
            )

        cca_ids = np.concatenate([
            accessible_ids,
            sampled_inaccessible_ids,
        ])

        cca_labels = np.concatenate([
            np.ones_like(accessible_ids),
            np.zeros_like(sampled_inaccessible_ids),
        ])

        return cca_ids, cca_labels

    def _get_joint_map(self, species):
        species = self._normalize_species(species)

        if species == "ref":
            return self.ref_to_joint

        if species == "new":
            return self.new_to_joint

        raise ValueError("Unknown species type.")

    @staticmethod
    def _expand_species(species, n: int):
        """
        Species is used only internally and will not be returned.

        Supported:
            "ref" / "reference" / 0
            "new" / "target" / 1
            list of the above
        """

        if isinstance(species, (str, int, np.integer)):
            return [
                CellDatasetJoint._normalize_species(species)
                for _ in range(n)
            ]

        species = list(species)

        if len(species) != n:
            raise ValueError(
                "`species` length must match `cell_sentences`. "
                f"Got {len(species)} and {n}."
            )

        return [
            CellDatasetJoint._normalize_species(x)
            for x in species
        ]

    @staticmethod
    def _normalize_species(species) -> str:
        if species in {0, "ref", "reference", "Reference"}:
            return "ref"

        if species in {1, "new", "target", "New", "Target"}:
            return "new"

        raise ValueError(
            "`species` should be 0/'ref'/'reference' or 1/'new'/'target'."
        )
    

class InferenceCellDatasetJoint(Dataset):
    """
    Dataset for joint-vocabulary EpiZooX inference.

    It maps original cCRE token ids to joint-vocabulary token ids,
    then returns input_ids only.
    """

    def __init__(
        self,
        cell_sentences,
        species,
        ref_to_joint,
        new_to_joint,
        max_length: int = 8192,
        random_sample: bool = True,
    ):
        self.cell_sentences = [
            parse_cell_sentence(x)
            for x in list(cell_sentences)
        ]

        self.species = CellDatasetJoint._expand_species(
            species=species,
            n=len(self.cell_sentences),
        )

        self.ref_to_joint = {
            int(k): int(v)
            for k, v in ref_to_joint.items()
        }
        self.new_to_joint = {
            int(k): int(v)
            for k, v in new_to_joint.items()
        }

        self.max_length = max_length
        self.random_sample = random_sample

    def __len__(self) -> int:
        return len(self.cell_sentences)

    def __getitem__(self, idx: int):
        cell = np.asarray(
            self.cell_sentences[idx],
            dtype=int,
        )

        joint_map = self._get_joint_map(self.species[idx])

        cell = self._map_cell_to_joint(
            cell=cell,
            joint_map=joint_map,
        )

        cell = truncate_cell(
            cell=cell,
            max_length=self.max_length,
            random_sample=self.random_sample,
        )

        return add_special_tokens(cell)

    def _map_cell_to_joint(
        self,
        cell: np.ndarray,
        joint_map,
    ) -> np.ndarray:
        ccre_idx = cell - CCRE_TOKEN_OFFSET

        if np.any(ccre_idx < 0):
            raise ValueError(
                "Cell sentence should contain cCRE token ids with offset. "
                f"Found token ids smaller than {CCRE_TOKEN_OFFSET}."
            )

        try:
            mapped = [
                joint_map[int(idx)]
                for idx in ccre_idx
            ]
        except KeyError as exc:
            raise KeyError(
                f"cCRE index {exc.args[0]} not found in joint map. "
                "Please make sure ref_to_joint/new_to_joint is complete."
            ) from exc

        return np.asarray(mapped, dtype=int)

    def _get_joint_map(self, species):
        species = CellDatasetJoint._normalize_species(species)

        if species == "ref":
            return self.ref_to_joint

        if species == "new":
            return self.new_to_joint

        raise ValueError("Unknown species type.")


class CellDatasetCancer(CellDataset):
    """
    Dataset for EpiZooCancer training.

    Difference from CellDataset:
        - Adds cancer_type_id for each cell.

    Each item returns:
        {
            "input_ids": LongTensor
            "signal": FloatTensor
            "cca_ids": LongTensor
            "cca_labels": FloatTensor
            "species": int
            "cancer_type": int
        }
    """

    def __init__(
        self,
        cell_sentences,
        species,
        cancer_type,
        max_length: int = 8192,
        cca_alpha: float = 1.0,
        human_num_ccres: int = 1_355_445,
        mouse_num_ccres: int = 1_341_077,
        random_sample: bool = True,
    ):
        super().__init__(
            cell_sentences=cell_sentences,
            species=species,
            max_length=max_length,
            cca_alpha=cca_alpha,
            human_num_ccres=human_num_ccres,
            mouse_num_ccres=mouse_num_ccres,
            random_sample=random_sample,
        )

        self.cancer_type = list(cancer_type)

        if len(self.cell_sentences) != len(self.cancer_type):
            raise ValueError(
                "`cell_sentences` and `cancer_type` must have the same length. "
                f"Got {len(self.cell_sentences)} and {len(self.cancer_type)}."
            )

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        item["cancer_type"] = int(self.cancer_type[idx])

        return item
    

def collate_fn_cancer(batch):
    """
    Collate function for EpiZooCancer training.

    Returns:
        input_ids:
            LongTensor [batch_size, max_seq_len]

        signals_human:
            FloatTensor [num_human_cells, human_num_ccres] or None

        signals_mouse:
            FloatTensor [num_mouse_cells, mouse_num_ccres] or None

        cca_ids:
            List[LongTensor], one tensor per cell

        cca_labels:
            FloatTensor [total_num_cca_samples]

        species:
            List[int]

        cancer_type:
            LongTensor [batch_size]
    """

    cancer_type = torch.tensor(
        [item["cancer_type"] for item in batch],
        dtype=torch.long,
    )

    input_ids, signals_human, signals_mouse, cca_ids, cca_labels, species = collate_fn(batch)

    return (
        input_ids,
        signals_human,
        signals_mouse,
        cca_ids,
        cca_labels,
        species,
        cancer_type,
    )


class InferenceCellDatasetCancer(InferenceCellDataset):
    """
    Dataset for EpiZooCancer inference.

    Difference from InferenceCellDataset:
        - Adds cancer_type_id for each cell.

    Each item returns:
        input_ids, species, cancer_type
    """

    def __init__(
        self,
        cell_sentences,
        species,
        cancer_type,
        max_length: int = 8192,
        random_sample: bool = True,
    ):
        super().__init__(
            cell_sentences=cell_sentences,
            species=species,
            max_length=max_length,
            random_sample=random_sample,
        )

        self.cancer_type = list(cancer_type)

        if len(self.cell_sentences) != len(self.cancer_type):
            raise ValueError(
                "`cell_sentences` and `cancer_type` must have the same length. "
                f"Got {len(self.cell_sentences)} and {len(self.cancer_type)}."
            )

    def __getitem__(self, idx: int):
        input_ids, species = super().__getitem__(idx)
        cancer_type = int(self.cancer_type[idx])

        return input_ids, species, cancer_type
    

def inference_collate_fn_cancer(batch):
    """
    Collate function for EpiZooCancer inference.

    Returns:
        input_ids:
            LongTensor [batch_size, max_seq_len]

        species:
            List[int]

        cancer_type:
            LongTensor [batch_size]
    """

    input_ids, species, cancer_type = zip(*batch)

    input_ids = pad_input_ids(
        input_ids,
        pad_token_id=PAD_TOKEN_ID,
    )

    cancer_type = torch.tensor(
        cancer_type,
        dtype=torch.long,
    )

    return input_ids, list(species), cancer_type