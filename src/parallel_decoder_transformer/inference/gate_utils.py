"""Diagnostic and migration utilities for SNC per-head gating."""

from __future__ import annotations

import math
from typing import Dict

import torch

from .snc_cross_attn import SharedNotesCrossAttention


def gate_entropy(gate_values: torch.Tensor) -> float:
    """Compute entropy over per-head gate values treated as a distribution.

    The gate values (sigmoid outputs in [0, 1]) are normalized via softmax to
    form a probability distribution, then Shannon entropy is computed.  Uniform
    gate values yield maximum entropy ``log(H)``; a single dominant head yields
    entropy near zero.

    Args:
        gate_values: 1-D tensor of per-head gate values, shape ``(H,)``.

    Returns:
        Entropy in nats (natural log).
    """
    if gate_values.numel() == 0:
        return 0.0
    probs = torch.softmax(gate_values.float(), dim=0)
    log_probs = torch.log(probs + 1e-12)
    entropy = -(probs * log_probs).sum()
    return float(entropy.item())


def log_gate_stats(
    module: SharedNotesCrossAttention,
    prefix: str,
) -> Dict[str, float]:
    """Extract per-head gate statistics suitable for WandB logging.

    For each head ``h`` the mean sigmoid gate value is reported as
    ``{prefix}/gate_head_{h}_mean``.  An aggregate entropy metric is included
    as ``{prefix}/gate_entropy``.

    Args:
        module: A ``SharedNotesCrossAttention`` instance whose ``gate``
            parameter will be inspected.
        prefix: String prefix for the returned metric keys.

    Returns:
        Dictionary mapping metric names to float values.
    """
    gate_param = module.gate.detach().float()
    gate_sigmoid = torch.sigmoid(gate_param)
    stats: Dict[str, float] = {}
    for h in range(gate_sigmoid.numel()):
        stats[f"{prefix}/gate_head_{h}_mean"] = float(gate_sigmoid[h].item())
    stats[f"{prefix}/gate_entropy"] = gate_entropy(gate_sigmoid)
    return stats


def migrate_scalar_gate_checkpoint(
    state_dict: dict,
    key: str,
    num_heads: int,
) -> dict:
    """Broadcast a scalar gate parameter to per-head shape for checkpoint migration.

    When upgrading from ``gate_mode="scalar"`` (gate shape ``(1,)``) to
    ``gate_mode="per_head"`` or ``"per_head_dynamic"`` (gate shape ``(H,)``),
    this utility expands the stored parameter by repeating the scalar value
    across all heads.

    Args:
        state_dict: Model state dictionary (modified in-place and returned).
        key: The key in *state_dict* that holds the scalar gate tensor.
        num_heads: Target number of attention heads.

    Returns:
        The (mutated) state dictionary with the gate tensor expanded.

    Raises:
        KeyError: If *key* is not present in *state_dict*.
        ValueError: If the tensor at *key* does not have shape ``(1,)``.
    """
    if key not in state_dict:
        raise KeyError(f"Key '{key}' not found in state_dict")
    tensor = state_dict[key]
    if tensor.shape != (1,):
        raise ValueError(
            f"Expected scalar gate with shape (1,) at key '{key}', got {tuple(tensor.shape)}"
        )
    state_dict[key] = tensor.expand(num_heads).clone()
    return state_dict


__all__ = [
    "gate_entropy",
    "log_gate_stats",
    "migrate_scalar_gate_checkpoint",
]
