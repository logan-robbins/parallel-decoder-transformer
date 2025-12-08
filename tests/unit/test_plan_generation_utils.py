from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

import parallel_decoder_transformer

_DATASETS_PACKAGE = "parallel_decoder_transformer.datasets"
if _DATASETS_PACKAGE not in sys.modules:
    datasets_pkg = types.ModuleType(_DATASETS_PACKAGE)
    datasets_pkg.__path__ = [
        str(
            Path(__file__).resolve().parents[2]
            / "src"
            / "parallel_decoder_transformer"
            / "datasets"
        )
    ]
    sys.modules[_DATASETS_PACKAGE] = datasets_pkg
    setattr(parallel_decoder_transformer, "datasets", datasets_pkg)

plan_generation = importlib.import_module(f"{_DATASETS_PACKAGE}.plan_generation")
PlanValidationError = plan_generation.PlanValidationError
_attach_stream_ids = plan_generation._attach_stream_ids
_validate_plan_contracts = plan_generation._validate_plan_contracts


def test_attach_stream_ids_normalizes_headers() -> None:
    streams = [
        {"header": "Part 1", "summary": "s1", "entities": ["A"], "constraints": ["c1"]},
        {"header": "Part 2", "summary": "s2", "entities": ["B"], "constraints": ["c2"]},
    ]
    normalized = _attach_stream_ids(streams)
    assert normalized[0]["stream_id"] == "stream_1"
    assert normalized[1]["stream_id"] == "stream_2"
    assert normalized[0]["entities"] == ["A"]


def _sectional_payload() -> dict[str, object]:
    ranges = [("A", "H"), ("I", "P"), ("Q", "Z")]
    streams: list[dict[str, object]] = []
    for idx, (start, end) in enumerate(ranges, start=1):
        streams.append(
            {
                "stream_id": f"stream_{idx}",
                "header": f"Header {idx}",
                "summary": f"Summary {idx}",
                "entities": [],
                "constraints": [f"constraint-{idx}"],
                "section_contract": {
                    "type": "alphabet_range",
                    "start": start,
                    "end": end,
                },
                "notes_contract": ["note"],
            }
        )
    return {"sectional_independence": True, "streams": streams}


def test_validate_plan_contracts_rejects_cross_stream_dependencies() -> None:
    payload = _sectional_payload()
    payload_streams = payload["streams"]  # type: ignore[index]
    payload_streams[1]["summary"] = "Builds on stream 1 output"  # type: ignore[index]
    with pytest.raises(PlanValidationError):
        _validate_plan_contracts(payload)


def test_validate_plan_contracts_warns_when_lint_set(monkeypatch, caplog) -> None:
    payload = _sectional_payload()
    payload_streams = payload["streams"]  # type: ignore[index]
    payload_streams[2]["constraints"] = [
        "Depends on stream 1 to finish the answer"
    ]  # type: ignore[index]
    monkeypatch.setenv("PDT_PLAN_CROSS_STREAM_LINT", "warn")
    with caplog.at_level("WARNING"):
        _validate_plan_contracts(payload)
    assert "Stream 3" in caplog.text
    assert "constraint[0]" in caplog.text
