"""Shared dataset and DataLoader helper functions."""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


def detection_collate_fn(
    batch: Sequence[Tuple[Tensor, Dict[str, Tensor]]]
) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
    """Stack equal-sized images and keep targets as a list of dictionaries."""

    images, targets = zip(*batch)
    return torch.stack(list(images), dim=0), list(targets)


def seed_worker(worker_id: int) -> None:
    """Create deterministic random states for DataLoader workers."""

    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def move_targets_to_device(
    targets: Sequence[Dict[str, Tensor]],
    device: torch.device,
) -> List[Dict[str, Tensor]]:
    """Move all tensor values in detection targets to a device."""

    moved_targets: List[Dict[str, Tensor]] = []

    for target in targets:
        moved_target: Dict[str, Tensor] = {}

        for key, value in target.items():
            moved_target[key] = value.to(device)

        moved_targets.append(moved_target)

    return moved_targets
