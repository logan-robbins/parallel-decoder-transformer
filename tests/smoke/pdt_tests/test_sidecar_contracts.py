"""Canonical sidecar composition and speculation-writer contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from pdt.config.loader import load_config
from pdt.config.schemas import (
    LossWeights,
    PlannerHeadConfig,
    PlanNotesProjectionConfig,
    SNCConfig,
    SidecarConfig,
    SpeculationHeadConfig,
    StreamAdapterConfig,
    StreamClassifierConfig,
)
from pdt.model import Sidecar
from pdt.sidecar.adapters import StreamAdapterLayer
from pdt.sidecar.heads.speculation import SpeculationHead
from pdt.sidecar.snc import SharedNotesCrossAttention


def _tiny_sidecar_config() -> SidecarConfig:
    return SidecarConfig(
        hidden_size=8,
        notes_dim=4,
        plan_vocab_size=16,
        num_streams=2,
        snc=SNCConfig(hidden_size=8, notes_dim=4, num_heads=2),
        adapters=StreamAdapterConfig(
            hidden_size=8,
            bottleneck_size=4,
            streams=("stream_0", "stream_1"),
        ),
        planner_head=PlannerHeadConfig(hidden_size=8, vocab_size=16, num_slots=2),
        plan_notes_proj=PlanNotesProjectionConfig(hidden_size=8, notes_dim=4),
        speculation_head=SpeculationHeadConfig(hidden_size=8, notes_dim=4),
        stream_classifier=StreamClassifierConfig(hidden_size=8, num_streams=2),
    )


def test_canonical_sidecar_excludes_untrained_commit_heads() -> None:
    sidecar = Sidecar(_tiny_sidecar_config())

    assert not hasattr(sidecar, "coverage_head")
    assert not hasattr(sidecar, "agreement_head")
    assert not any(
        key.startswith(("coverage_head.", "agreement_head.")) for key in sidecar.state_dict()
    )


def test_speculation_writer_is_exactly_one_unscaled_projection() -> None:
    head = SpeculationHead(SpeculationHeadConfig(hidden_size=3, notes_dim=2, dropout=0.0))
    hidden = torch.tensor([[1.0, 2.0, 3.0]])

    actual = head(hidden)
    expected = F.linear(hidden, head.projector.weight, head.projector.bias)

    torch.testing.assert_close(actual, expected)
    assert not hasattr(head.config, "teacher_scale")


def test_removed_sidecar_options_fail_as_unknown_constructor_arguments() -> None:
    with pytest.raises(TypeError, match="spectral_norm"):
        SNCConfig(spectral_norm=False)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="teacher_scale"):
        SpeculationHeadConfig(teacher_scale=1.0)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="coverage_head"):
        SidecarConfig(coverage_head=object())  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="coverage"):
        LossWeights(coverage=0.0)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="readiness"):
        LossWeights(readiness=0.0)  # type: ignore[call-arg]


def test_canonical_yaml_materializes_without_removed_runtime_or_sidecar_fields() -> None:
    root = Path(__file__).resolve().parents[3]
    config = load_config(root / "configs" / "pdt_qwen3_4b.yaml")

    assert not hasattr(config.runtime, "agreement_threshold")
    assert not hasattr(config.runtime, "commit_horizon")
    assert not hasattr(config.sidecar, "coverage_head")
    assert not hasattr(config.sidecar, "agreement_head")


def test_canonical_trainable_parameter_count_is_exact() -> None:
    root = Path(__file__).resolve().parents[3]
    config = load_config(root / "configs" / "pdt_qwen3_4b.yaml")

    with torch.device("meta"):
        sidecar = Sidecar(config.sidecar)
        snc = SharedNotesCrossAttention(
            config.sidecar.snc,
            gating_init=config.instrumentation.snc_gate_init,
        )
        adapters = StreamAdapterLayer(config.sidecar.adapters)

    sidecar_parameters = sum(parameter.numel() for parameter in sidecar.parameters())
    snc_parameters = sum(parameter.numel() for parameter in snc.parameters())
    adapter_parameters = sum(parameter.numel() for parameter in adapters.parameters())
    per_layer_parameters = snc_parameters + adapter_parameters + 2
    total_parameters = sidecar_parameters + (
        len(config.instrumentation.target_layers) * per_layer_parameters
    )

    assert sidecar_parameters == 133_704_707
    assert snc_parameters == 14_428_161
    assert adapter_parameters == 7_873_536
    assert per_layer_parameters == 22_301_699
    assert total_parameters == 401_325_095


def test_fp32_sidecar_heads_accept_bfloat16_trunk_hidden_states() -> None:
    sidecar = Sidecar(_tiny_sidecar_config()).float().eval()
    hidden = torch.randn(1, 3, 8, dtype=torch.bfloat16)
    mask = torch.ones((1, 3), dtype=torch.long)

    planner = sidecar.planner_head(hidden, attention_mask=mask)
    ownership = torch.tensor([[[True, False], [False, True]]])
    anchors = sidecar.plan_notes_proj(planner.quantized, ownership)
    write = sidecar.speculation_head(hidden[:, -1])
    classifier = sidecar.stream_classifier(hidden)

    assert planner.quantized.dtype == torch.float32
    assert anchors.dtype == torch.float32
    assert write.dtype == torch.float32
    assert classifier.dtype == torch.float32
    assert torch.isfinite(anchors).all()
    assert torch.isfinite(write).all()
    assert torch.isfinite(classifier).all()


def test_instrumented_phi_accepts_cross_dtype_hidden_and_notes() -> None:
    config = _tiny_sidecar_config()
    snc = SharedNotesCrossAttention(config.snc).to(dtype=torch.bfloat16).eval()
    adapters = StreamAdapterLayer(config.adapters).to(dtype=torch.bfloat16).eval()
    hidden = torch.randn(1, 2, 8, dtype=torch.bfloat16)
    fp32_notes = torch.randn(1, 4, 4, dtype=torch.float32)
    notes_mask = torch.ones((1, 4), dtype=torch.bool)

    snc_delta = snc(hidden, fp32_notes, notes_mask=notes_mask)
    adapter_delta = adapters(hidden, "stream_0")

    assert snc_delta.dtype == hidden.dtype
    assert adapter_delta.dtype == hidden.dtype
    assert snc_delta.shape == hidden.shape
    assert adapter_delta.shape == hidden.shape
