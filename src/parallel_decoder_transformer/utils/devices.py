"""Device utilities with macOS MPS/CUDA support."""

from __future__ import annotations

import os
from typing import Literal

import torch


def resolve_device() -> Literal["cpu", "cuda", "mps"]:
    """Resolve the runtime device from the PDT_DEVICE environment variable.

    This function requires the PDT_DEVICE environment variable to be explicitly set
    to one of 'cpu', 'cuda', or 'mps'. It will raise a ValueError if the
    variable is not set or if the specified device is unavailable.
    """
    device = os.getenv("PDT_DEVICE")

    if not device:
        raise ValueError("PDT_DEVICE environment variable must be set to 'cpu', 'cuda', or 'mps'.")

    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("PDT_DEVICE is set to 'cuda', but CUDA is not available.")
        return "cuda"
    elif device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("PDT_DEVICE is set to 'mps', but MPS is not available.")
        return "mps"
    elif device == "cpu":
        return "cpu"
    else:
        raise ValueError(f"Unsupported device specified in PDT_DEVICE: {device}")
