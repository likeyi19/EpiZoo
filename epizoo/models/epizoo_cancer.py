# epizoo/models/epizoo_cancer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from epizoo.models.epizoo import (
    EpiZoo,
    EpiZooConfig,
    init_weights,
)


@dataclass
class EpiZooCancerConfig(EpiZooConfig):
    """
    Configuration for EpiZooCancer.

    EpiZooCancer reuses the full EpiZoo architecture and adds a learnable
    cancer-type embedding to the cell embedding.
    """

    num_cancer_types: int = 8
    cancer_dropout: float = 0.1
    cancer_keep_prob: float = 0.9


class EpiZooCancer(EpiZoo):
    """
    EpiZoo model with cancer-type context.

    Difference from EpiZoo:
        EpiZoo:
            input_ids -> cell_emb

        EpiZooCancer:
            input_ids -> cell_emb
            cancer_type_ids -> cancer_emb
            final_cell_emb = cell_emb + cancer_emb

    Notes
    -----
    During training, cancer embeddings are randomly zeroed at the sample level
    with probability `1 - cancer_keep_prob`, matching the original implementation.

    This model does not compute loss in forward.
    SR and CCA losses are inherited from EpiZoo:
        - compute_signal_loss()
        - compute_cca_loss()
    """

    def __init__(
        self,
        cfg: Optional[EpiZooCancerConfig] = None,
    ):
        super().__init__(cfg or EpiZooCancerConfig())

        self.cfg: EpiZooCancerConfig

        self.cancer_emb = nn.Embedding(
            self.cfg.num_cancer_types,
            self.cfg.emb_dim,
        )

        self.cancer_drop = nn.Dropout(
            self.cfg.cancer_dropout,
        )

        self.cancer_emb.apply(
            lambda module: init_weights(
                module,
                init_range=self.cfg.init_range,
            )
        )

    def add_cancer_context(
        self,
        cell_emb: torch.Tensor,
        cancer_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add cancer-type context to cell embeddings.

        Parameters
        ----------
        cell_emb:
            Tensor [batch_size, emb_dim]

        cancer_type_ids:
            LongTensor [batch_size]

        Returns
        -------
        cell_emb:
            Cancer-context-aware cell embeddings.
        """

        cancer_type_ids = cancer_type_ids.to(
            device=cell_emb.device,
            dtype=torch.long,
        )

        if cancer_type_ids.dim() != 1:
            raise ValueError(
                "`cancer_type_ids` should be a 1D tensor with shape [batch_size]."
            )

        if cancer_type_ids.size(0) != cell_emb.size(0):
            raise ValueError(
                "`cancer_type_ids` length should match batch size. "
                f"Got {cancer_type_ids.size(0)} and {cell_emb.size(0)}."
            )

        cancer_emb = self.cancer_emb(cancer_type_ids)
        cancer_emb = self.cancer_drop(cancer_emb)

        if self.training:
            keep_mask = torch.bernoulli(
                torch.full(
                    size=(cell_emb.size(0), 1),
                    fill_value=self.cfg.cancer_keep_prob,
                    device=cell_emb.device,
                    dtype=cell_emb.dtype,
                )
            )

            cancer_emb = cancer_emb * keep_mask

        return cell_emb + cancer_emb

    def forward(
        self,
        input_ids: torch.Tensor,
        cancer_type_ids: Optional[torch.Tensor] = None,
        rank_ids: Optional[torch.Tensor] = None,
        return_transformer_out: bool = True,
    ):
        """
        Forward pass.

        This method only returns embeddings/logits needed by downstream losses.
        It does not compute SR or CCA loss.
        """

        transformer_out = self.encode(input_ids=input_ids, rank_ids=rank_ids)
        cell_emb = self.get_cell_emb(transformer_out)

        if cancer_type_ids is not None:
            cell_emb = self.add_cancer_context(cell_emb=cell_emb, cancer_type_ids=cancer_type_ids)

        outputs = {
            "cell_emb": cell_emb,
        }

        if return_transformer_out:
            transformer_out = transformer_out.clone()
            transformer_out[:, 0, :] = cell_emb
            outputs["transformer_out"] = transformer_out

        return outputs