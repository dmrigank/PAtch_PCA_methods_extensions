"""Reproducibility helpers."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, Torch, and CUDA RNGs when available."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def seed_worker(worker_id: int) -> None:
    """Seed a PyTorch dataloader worker from Torch's initial seed."""
    del worker_id
    try:
        import torch
    except ImportError:
        return
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def torch_generator(seed: int) -> Any:
    """Return a seeded Torch generator for dataloaders."""
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
