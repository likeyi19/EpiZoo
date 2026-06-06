# epizoo/models/lora.py

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple, Type

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA wrapper for nn.Linear.

    Original linear layer is frozen.
    Only lora_A and lora_B are trainable.
    """

    def __init__(
        self,
        linear: nn.Linear,
        r: int = 8,
        alpha: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()

        if not isinstance(linear, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(linear)}.")

        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(1, r)

        for param in self.linear.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(r, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        x = self.dropout(x)
        update = (x @ self.lora_A.T) @ self.lora_B.T
        return base + update * self.scaling


class LoRAEmbedding(nn.Module):
    """
    LoRA wrapper for nn.Embedding.

    Original embedding table is frozen.
    Only lora_A and lora_B are trainable.
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        if not isinstance(embedding, nn.Embedding):
            raise TypeError(f"Expected nn.Embedding, got {type(embedding)}.")

        self.embedding = embedding
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.embedding.weight.requires_grad = False

        num_embeddings, emb_dim = embedding.weight.shape

        self.lora_A = nn.Parameter(torch.zeros(num_embeddings, r))
        self.lora_B = nn.Parameter(torch.zeros(r, emb_dim))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        base = self.embedding(input_ids)
        update = (self.lora_A[input_ids] @ self.lora_B) * self.scaling
        return base + self.dropout(update)


def apply_lora_to_transformer(
    model: nn.Module,
    r: int = 8,
    alpha: int = 32,
    dropout: float = 0.0,
    target_keywords: Sequence[str] = ("Wqkv", "out_proj", "mlp"),
) -> nn.Module:
    """
    Apply LoRA to transformer Linear layers.

    This matches the original logic:
        - target modules whose names contain Wqkv / out_proj / mlp
        - only nn.Linear modules are wrapped
    """

    targets = _find_modules(
        model=model,
        module_types=(nn.Linear,),
        target_keywords=target_keywords,
    )

    for name, module in targets:
        _replace_module(
            model=model,
            module_name=name,
            new_module=LoRALinear(
                module,
                r=r,
                alpha=alpha,
                dropout=dropout,
            ),
        )
        print(f"[LoRA] Applied to transformer module: {name}")

    return model


def apply_lora_to_decoder(
    model: nn.Module,
    r: int = 8,
    alpha: int = 32,
    dropout: float = 0.0,
    target_keywords: Optional[Sequence[str]] = None,
) -> nn.Module:
    """
    Apply LoRA to decoder Linear layers.

    If `model` is a single nn.Linear, return a LoRALinear wrapper directly.
    If `model` is a module containing Linear layers, wrap all matched Linear layers.
    """

    if isinstance(model, nn.Linear):
        print("[LoRA] Applied to decoder Linear layer")
        return LoRALinear(
            model,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )

    targets = _find_modules(
        model=model,
        module_types=(nn.Linear,),
        target_keywords=target_keywords,
    )

    for name, module in targets:
        _replace_module(
            model=model,
            module_name=name,
            new_module=LoRALinear(
                module,
                r=r,
                alpha=alpha,
                dropout=dropout,
            ),
        )
        print(f"[LoRA] Applied to decoder module: {name}")

    return model


def apply_lora_to_embedding(
    model: nn.Module,
    r: int = 8,
    alpha: int = 32,
    dropout: float = 0.0,
    target_keywords: Sequence[str] = ("ccre_emb", "rank_emb"),
    skip_names: Sequence[str] = ("seq_emb",),
) -> nn.Module:
    """
    Apply LoRA to embedding layers.

    For the current EpiZoo naming:
        - apply to ccre_emb and rank_emb
        - skip seq_emb, which corresponds to precomputed SEAM lookup table

    This matches the old behavior:
        - apply to cCRE_embedding and rank_embedding
        - skip cCRE_embedding_1
    """

    targets = _find_modules(
        model=model,
        module_types=(nn.Embedding,),
        target_keywords=target_keywords,
        skip_names=skip_names,
    )

    for name, module in targets:
        _replace_module(
            model=model,
            module_name=name,
            new_module=LoRAEmbedding(
                module,
                r=r,
                alpha=alpha,
                dropout=dropout,
            ),
        )
        print(f"[LoRA] Applied to embedding module: {name}")

    return model


def freeze_module(module: nn.Module) -> nn.Module:
    """
    Freeze all parameters in a module.
    """

    for param in module.parameters():
        param.requires_grad = False

    return module


def count_parameters(model: nn.Module):
    """
    Count total, trainable, and frozen parameters.
    """

    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen = total - trainable

    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {frozen:,}")

    return total, trainable, frozen


def _find_modules(
    model: nn.Module,
    module_types: Tuple[Type[nn.Module], ...],
    target_keywords: Optional[Sequence[str]] = None,
    skip_names: Sequence[str] = (),
):
    """
    Find modules by type and optional name keywords.

    Targets are collected before replacement to avoid modifying the module tree
    while iterating through it.
    """

    targets = []

    for name, module in model.named_modules():
        if not name:
            continue

        if name in skip_names:
            continue

        if not isinstance(module, module_types):
            continue

        if target_keywords is not None and not any(key in name for key in target_keywords):
            continue

        targets.append((name, module))

    return targets


def _get_parent_module(
    model: nn.Module,
    module_name: str,
) -> nn.Module:
    """
    Get parent module from a dotted module name.
    """

    parent = model

    for name in module_name.split(".")[:-1]:
        parent = getattr(parent, name)

    return parent


def _replace_module(
    model: nn.Module,
    module_name: str,
    new_module: nn.Module,
) -> None:
    """
    Replace a child module by dotted module name.
    """

    parent = _get_parent_module(model, module_name)
    child_name = module_name.split(".")[-1]
    setattr(parent, child_name, new_module)