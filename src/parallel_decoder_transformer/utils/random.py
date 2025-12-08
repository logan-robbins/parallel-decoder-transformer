"""Randomness helpers for deterministic experiments."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: Optional[int], *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Parameters
    ----------
    seed:
        The seed to apply. When ``None`` the function is a no-op so callers
        can pass configuration values directly.
    deterministic:
        When ``True`` (default) additional PyTorch determinism flags are
        toggled to deliver reproducible kernels on CUDA-enabled hosts.
    """

    if seed is None:
        return

    value = int(seed)
    random.seed(value)
    os.environ["PYTHONHASHSEED"] = str(value)
    np.random.seed(value)
    torch.manual_seed(value)

    if torch.cuda.is_available():  # pragma: no cover - exercised on CUDA hosts
        torch.cuda.manual_seed_all(value)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover - backend not available
            pass
