"""Layer-resolved telemetry for trainable PDT architectural extensions."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sized
from typing import Any, Protocol, cast

import torch
from torch import nn

from pdt.trunk.physical_decoder import PhysicalDecoderLayerBank


__all__ = ["architecture_telemetry"]


class _PhysicalDecoderTelemetry(Protocol):
    fork_layer: int
    num_decoders: int
    layers: Sized

    def base_parameters(self) -> Iterator[nn.Parameter]: ...

    def extension_parameters(self) -> Iterator[nn.Parameter]: ...


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

    physical_decoder = getattr(model, "physical_decoder", None)
    physical_result: dict[str, object] = {}
    if isinstance(physical_decoder, nn.Module):
        decoder = cast(_PhysicalDecoderTelemetry, physical_decoder)
        physical_result = {
            "fork_layer": int(decoder.fork_layer),
            "num_decoders": int(decoder.num_decoders),
            "branch_layers": len(decoder.layers),
            "base_parameter_bank": _parameters_statistics(
                tuple(decoder.base_parameters())
            ),
            "extension_parameters": _parameters_statistics(
                tuple(decoder.extension_parameters())
            ),
        }

    layer_result: list[dict[str, object]] = []
    for ordinal, raw_layer in enumerate(getattr(model, "instrumented_layers", ())):
        layer = cast(PhysicalDecoderLayerBank, raw_layer)
        row: dict[str, object] = {
            "layer_index": int(getattr(layer, "pdt_layer_idx", ordinal)),
        }
        snc = getattr(layer, "snc", None)
        plan_attention = getattr(layer, "plan_attention", None)
        if isinstance(snc, nn.Module):
            row["snc"] = _module_statistics(snc)
            row["snc_inner_gate_probability"] = _gate_probability(
                getattr(snc, "gate", None), "SNC inner gate"
            )
        if isinstance(plan_attention, nn.Module):
            row["plan_attention"] = _module_statistics(plan_attention)
        notes_gate = getattr(layer, "notes_gate", None)
        if notes_gate is not None:
            row["snc_outer_gate_probability"] = _gate_probability(
                notes_gate, "SNC outer gate"
            )
        plan_gate = getattr(layer, "plan_gate", None)
        if plan_gate is not None:
            row["plan_outer_gate_probability"] = _gate_probability(
                plan_gate, "plan outer gate"
            )
        layer_result.append(row)

    return {
        "sidecar_modules": sidecar_result,
        "physical_decoder": physical_result,
        "instrumented_layers": layer_result,
    }


def _module_statistics(module: nn.Module) -> dict[str, int | float]:
    return _parameters_statistics(tuple(module.parameters()))


def _parameters_statistics(
    parameters: tuple[nn.Parameter, ...],
) -> dict[str, int | float]:
    scalars = sum(parameter.numel() for parameter in parameters)
    trainable_scalars = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    squared_l2 = sum(
        float(parameter.detach().float().square().sum().item()) for parameter in parameters
    )
    parameter_l2 = math.sqrt(squared_l2)
    if not math.isfinite(parameter_l2):
        raise ValueError("Non-finite parameter norm in architecture telemetry.")
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
