# epizoo/models/epizoo_anno.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from epizoo.models.epizoo import init_weights
from epizoo.models.moe_transformer import BertEncoderMoE, build_bert_moe_config


@dataclass
class EpiZooAnnoConfig:
    # Vocabulary
    vocab_size: int = 2_696_526

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

    # Classifier
    num_classes: int = 10
    classifier_hidden_dim: int = 256
    classifier_mid_dim: int = 128
    classifier_dropout: float = 0.25

    # Focal loss
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0

    # Initialization
    init_range: float = 0.02


class FocalLoss(nn.Module):
    """
    Focal loss for cell type classification.

    This follows the original implementation:
        CE = cross_entropy(logits, targets)
        pt = exp(-CE)
        loss = alpha * (1 - pt)^gamma * CE
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
        )

        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1.0 - pt) ** self.gamma * ce_loss

        return loss.mean()


class CellTypeClassifier(nn.Module):
    """
    Cell type classifier on top of EpiZoo cell embeddings.

    Architecture matches the original classifier:
        Linear -> LayerNorm -> Dropout -> Linear -> GELU -> LayerNorm -> Linear
    """

    def __init__(
        self,
        emb_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        mid_dim: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, mid_dim)
        self.norm2 = nn.LayerNorm(mid_dim)

        self.fc3 = nn.Linear(mid_dim, num_classes)
        self.gelu = nn.GELU()

    def forward(self, cell_emb: torch.Tensor) -> torch.Tensor:
        x = self.fc1(cell_emb)
        x = self.norm1(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.gelu(x)
        x = self.norm2(x)

        logits = self.fc3(x)

        return logits


class EpiZooAnno(nn.Module):
    """
    EpiZoo model for cell type annotation.

    Architecture:
        cCRE embedding + sequence embedding + rank embedding
        -> MoE transformer encoder
        -> [CLS] cell embedding
        -> cell type classifier

    This model does not include:
        - CCA head
        - signal reconstruction decoder
    """

    def __init__(self, cfg: Optional[EpiZooAnnoConfig] = None):
        super().__init__()
        self.cfg = cfg or EpiZooAnnoConfig()

        # Embeddings
        self.ccre_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.emb_dim, padding_idx=self.cfg.pad_token_id)
        self.seq_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.emb_dim, padding_idx=self.cfg.pad_token_id)
        self.rank_emb = nn.Embedding(self.cfg.max_rank, self.cfg.emb_dim, padding_idx=self.cfg.pad_token_id)

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

        # Classifier
        self.classifier = CellTypeClassifier(
            emb_dim=self.cfg.emb_dim,
            num_classes=self.cfg.num_classes,
            hidden_dim=self.cfg.classifier_hidden_dim,
            mid_dim=self.cfg.classifier_mid_dim,
            dropout=self.cfg.classifier_dropout,
        )

        # Loss
        self.loss_fn = FocalLoss(
            alpha=self.cfg.focal_alpha,
            gamma=self.cfg.focal_gamma,
        )

        # Init
        self.apply(lambda m: init_weights(m, self.cfg.init_range))

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
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor,) -> torch.Tensor:
        return self.loss_fn(logits, labels.long())

    # ---------------------------------------
    # Forward
    # ---------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        rank_ids: Optional[torch.Tensor] = None,
        return_cell_emb: bool = True,
        return_transformer_out: bool = False,
    ):
        transformer_out = self.encode(input_ids=input_ids, rank_ids=rank_ids)
        cell_emb = self.get_cell_emb(transformer_out)
        logits = self.classifier(cell_emb)

        outputs = {
            "logits": logits,
        }

        if return_cell_emb:
            outputs["cell_emb"] = cell_emb

        if return_transformer_out:
            outputs["transformer_out"] = transformer_out

        return outputs