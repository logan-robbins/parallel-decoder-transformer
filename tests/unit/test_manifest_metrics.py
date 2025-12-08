"""Tests for manifest metrics post-processing."""

from __future__ import annotations

import math

from parallel_decoder_transformer.evaluation.manifest_metrics import (
    aggregate_metrics,
    compute_coverage_summary,
    compute_gate_summary,
    compute_rollback_summary,
    summarize_manifest,
)


def test_compute_rollback_summary_basic() -> None:
    manifest = {
        "timings": {"stride_durations": [1.0, 1.5]},
        "streams": {
            "intro": {"token_ids": [1, 2, 3]},
            "wrap": {"token_ids": [4, 5]},
        },
        "rollbacks": [
            {"stream": "intro", "tokens_removed": [10, 11]},
            {"stream": "intro", "tokens_removed": [12]},
            {"stream": "wrap", "tokens_removed": [13, 14, 15]},
        ],
        "events": [
            {"agreement": 0.9, "rollback_performed": False},
            {"agreement": 0.2, "rollback_performed": True},
            {"agreement": 0.7, "rollback_performed": False},
        ],
    }
    summary = compute_rollback_summary(manifest)
    assert summary.total_events == 3
    assert summary.total_tokens_removed == 6
    assert summary.per_stream_events["intro"] == 2
    assert summary.per_stream_tokens["wrap"] == 3
    assert math.isclose(summary.per_stride_rate or 0.0, 1.5)
    assert summary.length_p50 == 2.0
    assert summary.length_p95 == 3.0
    assert summary.agreement_correlation is not None


def test_compute_gate_summary_thresholds() -> None:
    manifest = {
        "gate_trace": [
            {"stream": "intro", "value": 0.9},
            {"stream": "intro", "value": 0.8},
            {"stream": "intro", "value": 0.3},
            {"stream": "wrap", "value": 0.1},
            {"stream": "wrap", "value": 0.6},
            {"stream": "wrap", "value": 0.9},
        ]
    }
    summary = compute_gate_summary(
        manifest,
        gate_low_threshold=0.2,
        gate_high_threshold=0.7,
        entropy_bins=5,
    )
    assert set(summary.per_stream.keys()) == {"intro", "wrap"}
    intro_stats = summary.per_stream["intro"]
    assert intro_stats.count == 3
    assert intro_stats.high_fraction is not None and intro_stats.high_fraction > 0
    wrap_stats = summary.per_stream["wrap"]
    assert wrap_stats.oscillation_count == 2


def test_compute_coverage_summary_with_logits() -> None:
    manifest = {
        "config": {"coverage_threshold": 0.6},
        "plan": {
            "catalog": [
                {"index": 0, "stream": "intro", "text": "First milestone", "plan_item_id": 101},
                {"index": 1, "stream": "intro", "text": "Second milestone", "plan_item_id": 102},
                {"index": 2, "stream": "wrap", "text": "Closing summary", "plan_item_id": 103},
            ]
        },
        "streams": {
            "intro": {
                "text": "First milestone delivered but second milestone pending.",
                "coverage": {
                    "plan_items": [
                        {
                            "index": 0,
                            "stream": "intro",
                            "text": "First milestone",
                            "status": "covered",
                        },
                        {
                            "index": 1,
                            "stream": "intro",
                            "text": "Second milestone",
                            "status": "missing",
                        },
                    ]
                },
                "token_ids": [],
            },
            "wrap": {
                "text": "Closing summary reiterates the milestones.",
                "coverage": {
                    "plan_items": [
                        {
                            "index": 2,
                            "stream": "wrap",
                            "text": "Closing summary",
                            "status": "partial",
                        },
                    ]
                },
                "token_ids": [],
            },
        },
    }
    summary = compute_coverage_summary(manifest)
    assert summary.source == "logits"
    intro_stats = summary.per_stream["intro"]
    assert intro_stats.tp == 1.0
    assert intro_stats.fn == 1.0
    wrap_stats = summary.per_stream["wrap"]
    assert wrap_stats.tp == 1.0
    assert wrap_stats.fp == 0.0


def test_compute_coverage_summary_with_text_override() -> None:
    manifest = {
        "streams": {
            "intro": {
                "text": "Alpha project launch complete. Beta audit pending.",
                "token_ids": [],
            }
        }
    }
    override = {"intro": ["Alpha project launch", "Gamma stretch goal"]}
    summary = compute_coverage_summary(
        manifest,
        plan_text_override=override,
        coverage_partial_threshold=0.3,
    )
    assert summary.source == "text"
    intro_stats = summary.per_stream["intro"]
    assert intro_stats.tp == 1.0
    assert intro_stats.fn == 1.0


def test_summarize_and_aggregate_metrics() -> None:
    manifest = {
        "timings": {"stride_durations": [1.0]},
        "streams": {
            "intro": {
                "text": "First milestone completed.",
                "token_ids": [1, 2, 3],
                "coverage": {
                    "plan_items": [
                        {
                            "index": 0,
                            "stream": "intro",
                            "text": "First milestone",
                            "status": "covered",
                        },
                    ]
                },
            }
        },
        "plan": {
            "catalog": [
                {"index": 0, "stream": "intro", "text": "First milestone", "plan_item_id": 1},
            ]
        },
        "rollbacks": [],
        "gate_trace": [{"stream": "intro", "value": 0.5}],
    }
    summary = summarize_manifest(manifest, path="m1")
    aggregate = aggregate_metrics([summary])
    assert aggregate.count == 1
    assert aggregate.coverage.total_tp == 1.0
    assert aggregate.gate.per_stream["intro"].count == 1
