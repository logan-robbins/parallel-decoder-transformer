"""Actual-module architecture telemetry contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pdt.diagnostics.architecture import architecture_telemetry


def test_architecture_telemetry_reports_heads_layers_and_gate_openings() -> None:
    snc = nn.Linear(3, 2)
    snc.gate = nn.Parameter(torch.tensor(0.0))
    layer = SimpleNamespace(
        pdt_layer_idx=7,
        snc=snc,
        stream_adapter=nn.Linear(2, 2),
        notes_gate=nn.Parameter(torch.tensor(0.0)),
        adapter_gate=nn.Parameter(torch.tensor(-4.0)),
    )
    model = SimpleNamespace(
        sidecar=SimpleNamespace(
            planner_head=nn.Linear(4, 3),
            plan_notes_proj=nn.Linear(3, 2),
            speculation_head=nn.Linear(2, 2),
            stream_classifier=nn.Linear(2, 3),
        ),
        instrumented_layers=[layer],
    )

    result = architecture_telemetry(model)
    assert set(result["sidecar_modules"]) == {
        "planner_head",
        "plan_notes_proj",
        "speculation_head",
        "stream_classifier",
    }
    row = result["instrumented_layers"][0]
    assert row["layer_index"] == 7
    assert row["snc_inner_gate_probability"] == pytest.approx(0.5)
    assert row["snc_outer_gate_probability"] == pytest.approx(0.5)
    assert row["adapter_outer_gate_probability"] == pytest.approx(
        float(torch.sigmoid(torch.tensor(-4.0)))
    )
    assert row["snc"]["parameter_scalars"] > 0
