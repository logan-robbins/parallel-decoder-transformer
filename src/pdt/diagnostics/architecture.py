"""Layer-resolved telemetry for trainable PDT architectural extensions."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


__all__ = ["architecture_telemetry"]


@torch.no_grad()
def architecture_telemetry(model: Any) -> dict[str, object]:
    """Report actual head/module sizes, norms, and learned gate openings."""

    sidecar_result: dict[str, object] = {}
    sidecar = getattr(model, "sidecar", None)
    if sidecar is not None:
        for name in (
            "planner_head",
            "plan_memory_proj",
            "semantic_heads",
            "speculation_head",
        ):
            module = getattr(sidecar, name, None)
            if isinstance(module, nn.Module):
                sidecar_result[name] = _module_statistics(module)

    layer_result: list[dict[str, object]] = []
    for ordinal, layer in enumerate(getattr(model, "instrumented_layers", ())):
        row: dict[str, object] = {
            "layer_index": int(getattr(layer, "pdt_layer_idx", ordinal)),
        }
        snc = getattr(layer, "snc", None)
        plan_adapter = getattr(layer, "plan_adapter", None)
        if isinstance(snc, nn.Module):
            row["snc"] = _module_statistics(snc)
            row["snc_inner_gate_probability"] = _gate_probability(
                getattr(snc, "gate", None), "SNC inner gate"
            )
        if isinstance(plan_adapter, nn.Module):
            row["plan_adapter"] = _module_statistics(plan_adapter)
        notes_gate = getattr(layer, "notes_gate", None)
        if notes_gate is not None:
            row["snc_outer_gate_probability"] = _gate_probability(
                notes_gate, "SNC outer gate"
            )
        adapter_gate = getattr(layer, "adapter_gate", None)
        if adapter_gate is not None:
            row["adapter_outer_gate_probability"] = _gate_probability(
                adapter_gate, "adapter outer gate"
            )
        layer_result.append(row)

    return {
        "sidecar_modules": sidecar_result,
        "instrumented_layers": layer_result,
    }


def _module_statistics(module: nn.Module) -> dict[str, int | float]:
    parameters = tuple(module.parameters())
    scalars = sum(parameter.numel() for parameter in parameters)
    trainable_scalars = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    squared_l2 = sum(
        float(parameter.detach().float().square().sum().item()) for parameter in parameters
    )
    parameter_l2 = math.sqrt(squared_l2)
    if not math.isfinite(parameter_l2):
        raise ValueError(f"Non-finite parameter norm in {type(module).__name__}.")
    return {
        "parameter_scalars": scalars,
        "trainable_scalars": trainable_scalars,
        "parameter_l2": parameter_l2,
    }


def _gate_probability(value: object, label: str) -> float:
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError(f"{label} must be a scalar tensor.")
    probability = float(torch.sigmoid(value.detach().float()).item())
    if not math.isfinite(probability):
        raise ValueError(f"{label} produced a non-finite probability.")
    return probability
