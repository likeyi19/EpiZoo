# EpiZoo_v3/epizoo/models/moe_transformer.py

from __future__ import annotations

from functools import partial
from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig

from flash_attn.bert_padding import (
    index_first_axis_residual,
    pad_input,
    unpad_input,
)
from flash_attn.modules.block import Block
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import FusedMLP, Mlp

try:
    from flash_attn.ops.fused_dense import FusedDense
except ImportError:
    FusedDense = None


class SparseMoEFeedForward(nn.Module):
    """
    Sparse top-k MoE feed-forward network.

    Input shape:
        [num_tokens, hidden_size]

    Output shape:
        [num_tokens, hidden_size]
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_size: int,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
        return_residual: bool = False,
        normalize_topk: bool = True,
    ):
        super().__init__()

        if top_k > num_experts:
            raise ValueError(f"`top_k` should be <= `num_experts`, got top_k={top_k}, num_experts={num_experts}.")

        self.num_experts = num_experts
        self.top_k = top_k
        self.return_residual = return_residual
        self.normalize_topk = normalize_topk

        self.gate = nn.Linear(hidden_size, num_experts)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, ffn_size),
                nn.GELU(approximate="tanh"),
                nn.Linear(ffn_size, hidden_size),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x:
            Tensor with shape [num_tokens, hidden_size].
        """

        residual = x

        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)

        topk_weights, topk_indices = torch.topk(
            gate_probs,
            k=self.top_k,
            dim=-1,
        )

        if self.normalize_topk:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        output = torch.zeros_like(x)

        for expert_id, expert in enumerate(self.experts):
            expert_position_mask = topk_indices.eq(expert_id)
            token_mask = expert_position_mask.any(dim=-1)

            if not token_mask.any():
                continue

            expert_input = x[token_mask]

            expert_weight = (
                topk_weights[token_mask] * expert_position_mask[token_mask].float()
            ).sum(dim=-1, keepdim=True)

            expert_output = expert(expert_input)
            output[token_mask] += expert_output * expert_weight

        if self.return_residual:
            return output, residual

        return output


def create_mixer_cls(
    config: BertConfig,
    cross_attn: bool = False,
    return_residual: bool = False,
):
    """
    Build FlashAttention MHA class for a transformer block.
    """

    rotary_kwargs = {}

    if getattr(config, "position_embedding_type", "absolute") == "rotary":
        rotary_kwargs["rotary_emb_dim"] = getattr(config, "rotary_emb_dim", config.hidden_size)
        rotary_kwargs["rotary_emb_base"] = getattr(config, "rotary_emb_base", 10000.0)
        rotary_kwargs["rotary_emb_scale_base"] = getattr(config, "rotary_emb_scale_base", None)
        rotary_kwargs["rotary_emb_interleaved"] = getattr(config, "rotary_emb_interleaved", False)

    return partial(
        MHA,
        num_heads=config.num_attention_heads,
        cross_attn=cross_attn,
        dropout=config.attention_probs_dropout_prob,
        causal=False,
        fused_bias_fc=getattr(config, "fused_bias_fc", False),
        use_flash_attn=getattr(config, "use_flash_attn", False),
        return_residual=return_residual,
        **rotary_kwargs,
    )


def create_mlp_cls(
    config: BertConfig,
    layer_idx: Optional[int] = None,
    return_residual: bool = False,
):
    """
    Build either a standard FFN or SparseMoE FFN.
    """

    inner_dim = config.intermediate_size

    if getattr(config, "use_moe", False):
        return partial(
            SparseMoEFeedForward,
            ffn_size=inner_dim,
            num_experts=getattr(config, "num_experts", 4),
            top_k=getattr(config, "top_k", 2),
            dropout=config.hidden_dropout_prob,
            return_residual=return_residual,
            normalize_topk=getattr(config, "normalize_topk", True),
        )

    fused_mlp = getattr(config, "fused_mlp", False)

    if fused_mlp:
        if FusedMLP is None:
            raise ImportError("FusedMLP is not available. Please check your flash-attn installation.")

        mlp_checkpoint_lvl = getattr(config, "mlp_checkpoint_lvl", 0)

        if isinstance(mlp_checkpoint_lvl, Sequence):
            if layer_idx is None:
                raise ValueError("`layer_idx` is required when `mlp_checkpoint_lvl` is a sequence.")
            mlp_checkpoint_lvl = mlp_checkpoint_lvl[layer_idx]

        return partial(
            FusedMLP,
            hidden_features=inner_dim,
            checkpoint_lvl=mlp_checkpoint_lvl,
            return_residual=return_residual,
        )

    approximate = (
        "tanh"
        if config.hidden_act in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"]
        else "none"
    )

    return partial(
        Mlp,
        hidden_features=inner_dim,
        activation=partial(F.gelu, approximate=approximate),
        return_residual=return_residual,
    )


def create_transformer_block(
    config: BertConfig,
    layer_idx: Optional[int] = None,
):
    """
    Create one FlashAttention-compatible transformer block.
    """

    last_layer_subset = getattr(config, "last_layer_subset", False)
    cross_attn = last_layer_subset and layer_idx == config.num_hidden_layers - 1

    return_residual = not cross_attn

    mixer_cls = create_mixer_cls(
        config=config,
        cross_attn=cross_attn,
        return_residual=return_residual,
    )

    mlp_cls = create_mlp_cls(
        config=config,
        layer_idx=layer_idx,
        return_residual=return_residual,
    )

    norm_cls = partial(nn.LayerNorm, eps=config.layer_norm_eps)

    return Block(
        config.hidden_size,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        prenorm=False,
        resid_dropout1=config.hidden_dropout_prob,
        resid_dropout2=config.hidden_dropout_prob,
        fused_dropout_add_ln=getattr(config, "fused_dropout_add_ln", False),
        return_residual=return_residual,
    )


class BertEncoderMoE(nn.Module):
    """
    FlashAttention-compatible BERT encoder with optional SparseMoE FFN.

    Notes
    -----
    `key_padding_mask` follows FlashAttention convention:
        True  = valid token
        False = padding token
    """

    def __init__(self, config: BertConfig):
        super().__init__()

        self.config = config
        self.use_flash_attn = getattr(config, "use_flash_attn", False)

        self.layers = nn.ModuleList([
            create_transformer_block(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        subset_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states:
            Tensor with shape [batch_size, seq_len, hidden_size].

        key_padding_mask:
            Boolean tensor with shape [batch_size, seq_len].
            True indicates valid tokens.

        subset_mask:
            Optional boolean tensor. If provided, only output selected tokens
            after the final transformer layer.
        """

        if key_padding_mask is None or not self.use_flash_attn:
            mixer_kwargs = (
                {"key_padding_mask": key_padding_mask}
                if key_padding_mask is not None
                else None
            )

            for layer in self.layers:
                hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)

            if subset_mask is not None:
                hidden_states = hidden_states[subset_mask]

            return hidden_states

        batch_size, seq_len = hidden_states.shape[:2]

        hidden_states, indices, cu_seqlens, max_seqlen_in_batch, _ = unpad_input(
            hidden_states,
            key_padding_mask,
        )

        mixer_kwargs = {
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen_in_batch,
        }

        if subset_mask is None:
            for layer in self.layers:
                hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)

            hidden_states = pad_input(
                hidden_states,
                indices,
                batch_size,
                seq_len,
            )

            return hidden_states

        for layer in self.layers[:-1]:
            hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)

        subset_idx = torch.nonzero(
            subset_mask[key_padding_mask],
            as_tuple=False,
        ).flatten()

        subset_seqlens = (subset_mask & key_padding_mask).sum(
            dim=-1,
            dtype=torch.int32,
        )

        subset_cu_seqlens = F.pad(
            torch.cumsum(subset_seqlens, dim=0, dtype=torch.int32),
            (1, 0),
        )

        hidden_states_subset, hidden_states = index_first_axis_residual(
            hidden_states,
            subset_idx,
        )

        mixer_kwargs = {
            "x_kv": hidden_states,
            "cu_seqlens": subset_cu_seqlens,
            "max_seqlen": max_seqlen_in_batch,
            "cu_seqlens_k": cu_seqlens,
            "max_seqlen_k": max_seqlen_in_batch,
        }

        hidden_states = self.layers[-1](
            hidden_states_subset,
            mixer_kwargs=mixer_kwargs,
        )

        return hidden_states


def build_bert_moe_config(
    vocab_size: int,
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    intermediate_size: Optional[int] = None,
    max_position_embeddings: int = 8192,
    use_flash_attn: bool = True,
    use_moe: bool = True,
    num_experts: int = 4,
    top_k: int = 2,
    hidden_dropout_prob: float = 0.1,
    attention_probs_dropout_prob: float = 0.1,
    layer_norm_eps: float = 1e-12,
    hidden_act: str = "gelu",
    pad_token_id: int = 0,
    normalize_topk: bool = True,
) -> BertConfig:
    """
    Build the BertConfig used by EpiZoo's MoE transformer.
    """

    config = BertConfig(
        vocab_size=vocab_size,
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size or 4 * hidden_size,
        max_position_embeddings=max_position_embeddings,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        layer_norm_eps=layer_norm_eps,
        hidden_act=hidden_act,
        pad_token_id=pad_token_id,
    )

    config.use_flash_attn = use_flash_attn
    config.use_moe = use_moe
    config.num_experts = num_experts
    config.top_k = top_k
    config.normalize_topk = normalize_topk

    return config