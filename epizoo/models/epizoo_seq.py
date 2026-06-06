# epizoo/models/epizoo_seq.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from epizoo.models.seam import SEAM, SEAMConfig


@dataclass
class EpiZooSeqConfig:
    """
    Configuration for EpiZooSeq.
    """

    dnabert_path: str
    seq_emb_dim: int = 512
    hidden_dim: int = 256
    dropout: float = 0.1
    trust_remote_code: bool = True


class EpiZooSeq(nn.Module):
    """
    Predict sequence accessibility across cell types.

    Input:
        DNA sequence tokens:
            input_ids, attention_mask

        Fixed cell type embeddings:
            cell_type_emb [num_cell_types, cell_type_emb_dim]

    Output:
        accessibility scores:
            [batch_size, num_cell_types]

    Architecture:
        SEAM(sequence) -> seq_emb
        concat(seq_emb, cell_type_emb)
        -> MLP
        -> Softplus
    """

    def __init__(
        self,
        cell_type_emb: torch.Tensor,
        cfg: EpiZooSeqConfig,
    ):
        super().__init__()

        self.cfg = cfg

        if cell_type_emb.ndim != 2:
            raise ValueError(
                "`cell_type_emb` should have shape "
                "[num_cell_types, cell_type_emb_dim]."
            )

        self.num_cell_types = cell_type_emb.shape[0]
        self.cell_type_emb_dim = cell_type_emb.shape[1]

        self.register_buffer(
            "cell_type_emb",
            cell_type_emb.float(),
            persistent=False,
        )

        self.seam = SEAM(
            SEAMConfig(
                dnabert_path=cfg.dnabert_path,
                emb_dim=cfg.seq_emb_dim,
                trust_remote_code=cfg.trust_remote_code,
            )
        )

        self.head = nn.Sequential(
            nn.Linear(
                cfg.seq_emb_dim + self.cell_type_emb_dim,
                cfg.hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

        self.out_act = nn.Softplus()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict accessibility for each sequence across all cell types.

        Returns
        -------
        pred:
            Tensor [batch_size, num_cell_types]
        """

        seq_emb = self.seam(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        fused = self._fuse_seq_and_cell_type(seq_emb)

        pred = self.head(fused).squeeze(-1)
        pred = self.out_act(pred)

        return pred

    def _fuse_seq_and_cell_type(
        self,
        seq_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse sequence embeddings with all cell type embeddings.

        seq_emb:
            [batch_size, seq_emb_dim]

        output:
            [batch_size, num_cell_types, seq_emb_dim + cell_type_emb_dim]
        """

        batch_size = seq_emb.size(0)

        seq_emb = seq_emb.unsqueeze(1).expand(
            -1,
            self.num_cell_types,
            -1,
        )

        cell_type_emb = self.cell_type_emb.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )

        return torch.cat(
            [seq_emb, cell_type_emb],
            dim=-1,
        )