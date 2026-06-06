# epizoo/models/epizoo_x.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from epizoo.models.epizoo import (
    CCAHead,
    build_bce_loss,
    init_weights,
)
from epizoo.models.moe_transformer import (
    BertEncoderMoE,
    build_bert_moe_config,
)


@dataclass
class EpiZooXConfig:
    # Vocabulary
    vocab_size: int = 1_355_449
    ccre_offset: int = 4

    # Embedding
    emb_dim: int = 512
    max_rank: int = 8192
    pad_token_id: int = 0

    # Transformer
    num_layers: int = 18
    num_heads: int = 8
    hidden_dropout: float = 0.1
    attn_dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    hidden_act: str = "gelu"

    # MoE
    use_flash_attn: bool = True
    use_moe: bool = True
    num_experts: int = 4
    top_k: int = 2
    normalize_topk: bool = True

    # Heads
    cca_hidden_dim: int = 128

    # Loss
    cca_pos_weight: Optional[float] = None
    signal_pos_weight: float = 100.0

    # Initialization
    init_range: float = 0.02

    @property
    def signal_vocab_size(self) -> int:
        return self.vocab_size - self.ccre_offset


class EpiZooX(nn.Module):
    """
    Cross-species / new-species EpiZoo model.

    This is a single-vocabulary version of EpiZoo.

    Tasks:
        1. CCA: cell-cCRE alignment
        2. SR: signal reconstruction with a single decoder

    Difference from base EpiZoo:
        EpiZoo  : human decoder + mouse decoder
        EpiZooX : one shared decoder for one target vocabulary
    """

    def __init__(self, cfg: Optional[EpiZooXConfig] = None):
        super().__init__()

        self.cfg = cfg or EpiZooXConfig()

        # Embeddings
        self.ccre_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.emb_dim, padding_idx=self.cfg.pad_token_id)
        self.seq_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.emb_dim,padding_idx=self.cfg.pad_token_id)
        self.rank_emb = nn.Embedding(self.cfg.max_rank, self.cfg.emb_dim, padding_idx=self.cfg.pad_token_id,)

        # Transformer encoder
        bert_cfg = build_bert_moe_config(
            vocab_size=self.cfg.vocab_size,
            num_layers=self.cfg.num_layers,
            hidden_size=self.cfg.emb_dim,
            num_attention_heads=self.cfg.num_heads,
            intermediate_size=4 * self.cfg.emb_dim,
            max_position_embeddings=self.cfg.max_rank,
            use_flash_attn=self.cfg.use_flash_attn,
            use_moe=self.cfg.use_moe,
            num_experts=self.cfg.num_experts,
            top_k=self.cfg.top_k,
            normalize_topk=self.cfg.normalize_topk,
            hidden_dropout_prob=self.cfg.hidden_dropout,
            attention_probs_dropout_prob=self.cfg.attn_dropout,
            layer_norm_eps=self.cfg.layer_norm_eps,
            hidden_act=self.cfg.hidden_act,
            pad_token_id=self.cfg.pad_token_id,
        )

        self.encoder = BertEncoderMoE(bert_cfg)

        # Heads
        self.cca_head = CCAHead(emb_dim=self.cfg.emb_dim, hidden_dim=self.cfg.cca_hidden_dim)
        self.signal_decoder = nn.Linear(self.cfg.emb_dim, self.cfg.signal_vocab_size)

        # Loss
        self.cca_loss_fn = build_bce_loss(self.cfg.cca_pos_weight)
        self.signal_loss_fn = build_bce_loss(self.cfg.signal_pos_weight)

        # Init
        self.apply(
            lambda module: init_weights(
                module,
                init_range=self.cfg.init_range,
            )
        )

    # ---------------------------------------
    # Embedding & Encoding
    # ---------------------------------------
    def embed_tokens(self, input_ids: torch.Tensor, rank_ids: Optional[torch.Tensor] = None):
        batch_size, seq_len = input_ids.shape
        if rank_ids is None:
            rank_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        return self.ccre_emb(input_ids) + self.seq_emb(input_ids) + self.rank_emb(rank_ids)

    def encode(self, input_ids: torch.Tensor, rank_ids: Optional[torch.Tensor] = None):
        x = self.embed_tokens(input_ids, rank_ids)
        mask = input_ids.ne(self.cfg.pad_token_id)
        return self.encoder(x, key_padding_mask=mask)

    @staticmethod
    def get_cell_emb(transformer_out: torch.Tensor) -> torch.Tensor:
        return transformer_out[:, 0, :]

    # ---------------------------------------
    # Loss Utilities
    # ---------------------------------------
    def compute_cca_loss(self, cell_emb: torch.Tensor, ccre_ids: List[torch.Tensor], accessibility: torch.Tensor):
        repeat_counts = torch.tensor([len(ids) for ids in ccre_ids], device=cell_emb.device, dtype=torch.long)
        expanded = torch.repeat_interleave(cell_emb, repeat_counts, dim=0)
        sampled_ids = torch.cat([ids.to(cell_emb.device) for ids in ccre_ids], dim=0)
        sampled_emb = self.ccre_emb(sampled_ids)
        labels = accessibility.to(cell_emb.device).float().view(-1)
        logits = self.cca_head(expanded, sampled_emb)
        loss = self.cca_loss_fn(logits, labels)
        return {"loss": loss, "logits": logits, "prob": torch.sigmoid(logits)}

    def compute_signal_loss(self, cell_emb: torch.Tensor, signals: torch.Tensor):
        signals = signals.to(cell_emb.device).float()
        logits = self.signal_decoder(cell_emb)
        loss = self.signal_loss_fn(logits, signals)

        return {"loss": loss, "logits": logits}

    # ---------------------------------------
    # Forward
    # ---------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        rank_ids: Optional[torch.Tensor] = None,
        return_transformer_out: bool = True,
    ):
        transformer_out = self.encode(input_ids=input_ids, rank_ids=rank_ids)
        cell_emb = self.get_cell_emb(transformer_out)

        outputs = {
            "cell_emb": cell_emb,
        }

        if return_transformer_out:
            outputs["transformer_out"] = transformer_out

        return outputs