# epizoo/models/epizoo.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from epizoo.models.moe_transformer import BertEncoderMoE, build_bert_moe_config


@dataclass
class EpiZooConfig:

    # Vocabulary
    vocab_size: int = 2_696_526
    human_vocab_size: int = 1_355_445
    mouse_vocab_size: int = 1_341_077

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
    cca_pos_weight: float = 1.0
    signal_pos_weight: float = 100.0

    # Initialization
    init_range: float = 0.02


def init_weights(module: nn.Module, init_range: float = 0.02):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=init_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=init_range)
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def build_bce_loss(pos_weight: Optional[float] = None):
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()

    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32)
    )


class CCAHead(nn.Module):
    """Cell-cCRE Alignment Head"""
    def __init__(self, emb_dim: int, hidden_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cell_emb: torch.Tensor, ccre_emb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([cell_emb, ccre_emb], dim=-1)).squeeze(-1)


class SignalDecoder(nn.Module):
    """Species-specific Signal Reconstruction Decoder"""
    def __init__(self, emb_dim: int, species_vocab: Dict[str, int]):
        super().__init__()

        self.vocab_sizes = dict(species_vocab)

        self.decoders = nn.ModuleDict({
            sp: nn.Linear(emb_dim, vocab_size)
            for sp, vocab_size in species_vocab.items()
        })

    def forward(self, cell_emb: torch.Tensor, species: str) -> torch.Tensor:
        return self.decoders[species](cell_emb)


class EpiZoo(nn.Module):
    """
    EpiZoo model.
    """
    species_map = {0: "human", 1: "mouse"}

    def __init__(self, cfg: Optional[EpiZooConfig] = None):
        super().__init__()
        self.cfg = cfg or EpiZooConfig()

        # Embeddings
        self.ccre_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.emb_dim, padding_idx=self.cfg.pad_token_id)
        self.seq_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.emb_dim, padding_idx=self.cfg.pad_token_id)
        self.rank_emb = nn.Embedding(self.cfg.max_rank, self.cfg.emb_dim, padding_idx=self.cfg.pad_token_id)

        # Transformer Encoder
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
        self.signal_decoder = SignalDecoder(
            emb_dim=self.cfg.emb_dim,
            species_vocab={
                "human": self.cfg.human_vocab_size,
                "mouse": self.cfg.mouse_vocab_size,
            },
        )

        # Loss
        self.cca_loss_fn = build_bce_loss(self.cfg.cca_pos_weight)
        self.signal_loss_fn = build_bce_loss(self.cfg.signal_pos_weight)

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
    def compute_cca_loss(self, cell_emb: torch.Tensor, ccre_ids: List[torch.Tensor], accessibility: torch.Tensor):
        repeat_counts = torch.tensor([len(ids) for ids in ccre_ids], device=cell_emb.device, dtype=torch.long)
        expanded = torch.repeat_interleave(cell_emb, repeat_counts, dim=0)
        sampled_ids = torch.cat([ids.to(cell_emb.device) for ids in ccre_ids], dim=0)
        sampled_emb = self.ccre_emb(sampled_ids)
        labels = accessibility.to(cell_emb.device).float().view(-1)
        logits = self.cca_head(expanded, sampled_emb)
        loss = self.cca_loss_fn(logits, labels)
        return {"loss": loss, "logits": logits, "prob": torch.sigmoid(logits)}

    def compute_signal_loss(self, cell_emb: torch.Tensor, input_species: Union[List[int], torch.Tensor], signals: Dict[str, torch.Tensor]):
        if isinstance(input_species, torch.Tensor):
            input_species = input_species.detach().cpu().tolist()
        species_names = [self.species_map[i] if isinstance(i, int) else i for i in input_species]
        total_loss = torch.zeros([], device=cell_emb.device)
        total_weight = 0
        outputs = {}
        for sp, vocab_size in self.signal_decoder.decoders.items():
            mask = torch.tensor([s==sp for s in species_names], device=cell_emb.device, dtype=torch.bool)
            if mask.any():
                sp_emb = cell_emb[mask]
                pred = self.signal_decoder(sp_emb, sp)
                target = signals[sp].to(cell_emb.device).float()
                loss = self.signal_loss_fn(pred, target)
                outputs[f"pred_{sp}"] = pred
                outputs[f"loss_{sp}"] = loss
                total_loss += loss * vocab_size
                total_weight += vocab_size
            else:
                outputs[f"pred_{sp}"] = None
                outputs[f"loss_{sp}"] = torch.zeros([], device=cell_emb.device)
        outputs["loss"] = total_loss / total_weight if total_weight>0 else torch.tensor(0.0, device=cell_emb.device)
        return outputs

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