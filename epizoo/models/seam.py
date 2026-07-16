# epizoo/models/seam.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer


@dataclass
class SEAMConfig:
    """
    Configuration for SEAM.
    """

    cfg_path: str
    emb_dim: int = 512
    trust_remote_code: bool = True


class SEAM(nn.Module):
    """
    Sequence Embedding Alignment Module.

    This is the cleaned version of EmbeddingAligner.

    Input:
        input_ids, attention_mask

    Output:
        sequence embeddings with shape [batch_size, emb_dim]
    """

    def __init__(self, cfg: SEAMConfig):
        super().__init__()

        self.cfg = cfg

        bert_cfg = AutoConfig.from_pretrained(
            cfg.cfg_path,
            trust_remote_code=cfg.trust_remote_code,
        )

        self.backbone = AutoModel.from_config(
            bert_cfg,
            trust_remote_code=cfg.trust_remote_code,
        )

        hidden_size = getattr(bert_cfg, "hidden_size", 768)

        self.proj = nn.Linear(
            hidden_size,
            cfg.emb_dim,
        )

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden = self._get_last_hidden_state(outputs)
        cls_emb = hidden[:, 0, :]

        return self.proj(cls_emb)
    

    def forward_from_embeds(
        self,
        inputs_embeds,
        attention_mask,
    ):
        backbone = self.backbone

        token_type_ids = torch.zeros(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=inputs_embeds.device,
        )

        embedding_output = backbone.embeddings(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
        )

        outputs = backbone.encoder(
            embedding_output,
            attention_mask=attention_mask,
        )

        hidden = self._get_last_hidden_state(outputs)

        if hidden.dim() == 2:
            hidden = hidden.view(inputs_embeds.shape[0], inputs_embeds.shape[1], -1)

        cls_emb = hidden[:, 0, :]

        return self.proj(cls_emb)

    @staticmethod
    def _get_last_hidden_state(outputs) -> torch.Tensor:
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state

        return outputs[0]