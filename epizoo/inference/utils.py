# epizoo/inference/utils.py

from __future__ import annotations

from typing import Optional

import torch


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get torch device.
    """

    if device is not None:
        return torch.device(device)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_input_ids(batch):
    """
    Extract input_ids from a dataloader batch.

    Supported formats:
        1. input_ids
        2. (input_ids, ...)
        3. {"input_ids": input_ids, ...}
    """

    if isinstance(batch, dict):
        return batch["input_ids"]

    if isinstance(batch, torch.Tensor):
        return batch

    if isinstance(batch, (tuple, list)):
        return batch[0]

    raise ValueError(
        "Unsupported batch format. Expected tensor, tuple/list, or dict."
    )