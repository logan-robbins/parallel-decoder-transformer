"""Actual-module architecture telemetry contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pdt.diagnostics.architecture import architecture_telemetry


class _PhysicalDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fork_layer = 7
        self.num_decoders = 3
        self.base = nn.Parameter(torch.randn(3, 2, 2))
        self.extension = nn.Parameter(torch.randn(2, 2))
        self.layers = [object()]

    def base_parameters(self):
        yield self.base

    def extension_parameters(self):
        yield self.extension


def test_architecture_telemetry_reports_heads_layers_and_gate_openings() -> None:
    snc = nn.Linear(3, 2)
    snc.gate = nn.Parameter(torch.tensor(0.0))
    layer = SimpleNamespace(
        pdt_layer_idx=7,
        snc=snc,
        plan_attention=nn.Linear(2, 2),
        notes_gate=nn.Parameter(torch.tensor(0.0)),
        plan_gate=nn.Parameter(torch.tensor(-4.0)),
    )
    model = SimpleNamespace(
        sidecar=SimpleNamespace(
            planner_head=nn.Linear(4, 3),
            plan_memory_proj=nn.Linear(3, 2),
            semantic_heads=nn.Linear(2, 3),
            speculation_head=nn.Linear(2, 2),
        ),
        physical_decoder=_PhysicalDecoder(),
        instrumented_layers=[layer],
    )

    result = architecture_telemetry(model)
    assert set(result["sidecar_modules"]) == {
        "planner_head",
        "plan_memory_proj",
        "semantic_heads",
        "speculation_head",
    }
    row = result["instrumented_layers"][0]
    assert row["layer_index"] == 7
    assert row["snc_inner_gate_probability"] == pytest.approx(0.5)
    assert row["snc_outer_gate_probability"] == pytest.approx(0.5)
    assert row["plan_outer_gate_probability"] == pytest.approx(
        float(torch.sigmoid(torch.tensor(-4.0)))
    )
    assert row["snc"]["parameter_scalars"] > 0
    assert row["plan_attention"]["parameter_scalars"] > 0
    assert result["physical_decoder"]["num_decoders"] == 3
    assert result["physical_decoder"]["base_parameter_bank"]["parameter_scalars"] == 12
