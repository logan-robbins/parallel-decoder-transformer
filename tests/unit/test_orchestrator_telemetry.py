"""Telemetry surface tests for the inference orchestrator."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from parallel_decoder_transformer.inference.config import InferenceConfig
from parallel_decoder_transformer.inference.orchestrator import MultiStreamOrchestrator
from parallel_decoder_transformer.utils.plan_catalog import PlanHashParams


def test_finalize_includes_instrumentation_and_timings() -> None:
    """Final manifest should expose instrumentation metadata and timing traces."""

    orchestrator = MultiStreamOrchestrator.__new__(MultiStreamOrchestrator)
    config = InferenceConfig(
        streams=("intro",),
        stride_B=1,
        commit_L=1,
        read_lag_delta=0,
        max_snapshots_K=1,
    )
    orchestrator.config = config
    orchestrator.device = "cpu"
    orchestrator._instrumented_layers = (2, 4)
    orchestrator._timings = {"bootstrap": 0.01}
    orchestrator.alpha = 1.0
    orchestrator._start_time = None
    orchestrator._plan_token_ids = None
    orchestrator._plan_mask = None
    orchestrator._plan_logits = None
    orchestrator._cadence_events = []
    orchestrator._step_timings = [
        {"step": 1, "stream": "intro", "stride_index": 0, "token_index": 0, "duration_s": 0.005}
    ]
    orchestrator._gate_trace = [{"step": 1, "stream": "intro", "value": 0.8}]
    orchestrator._step_count = 1
    orchestrator._gate_values = {"intro": 0.8}
    orchestrator._coverage_manifest = {"intro": []}
    orchestrator._coverage_history = {"intro": []}
    orchestrator._plan_embeddings = None
    orchestrator._plan_mask_bool = None
    orchestrator._plan_ids_list = None
    orchestrator._plan_mask_list = None
    orchestrator._plan_source = "none"
    orchestrator._plan_hash_params = PlanHashParams(vocab_size=1000, hash_buckets=1000, salt="test")
    orchestrator._rollback_events = []
    orchestrator._cadence_events = []
    orchestrator.states = {
        "intro": SimpleNamespace(
            generated_text="Hello world",
            generated_pieces=["Hello", "world"],
            generated_tokens=[42],
            latest_snapshot_version=0,
            rollback_buffer=[],
        )
    }

    manifest = orchestrator.finalize()

    assert manifest["instrumented_layers"] == [2, 4]
    assert manifest["timings"]["per_token"] == orchestrator._step_timings
    assert manifest["gate_trace"] == orchestrator._gate_trace


def test_stride_sync_records_timings(monkeypatch):
    orchestrator = MultiStreamOrchestrator.__new__(MultiStreamOrchestrator)
    orchestrator._sync_profile = True
    orchestrator._timings = {}
    orchestrator._sync_overhead_total = 0.0
    orchestrator._last_stride_start = 0.0
    orchestrator._synchronize_device = lambda: None
    times = iter([0.0, 0.05])
    monkeypatch.setattr(
        "parallel_decoder_transformer.inference.orchestrator.time.time", lambda: next(times)
    )

    orchestrator._on_stride_complete()

    assert "stride_sync_durations" in orchestrator._timings
    assert orchestrator._timings["stride_sync_durations"][0] == pytest.approx(0.05)
    assert orchestrator._sync_overhead_total == pytest.approx(0.05)
