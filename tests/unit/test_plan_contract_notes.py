from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import parallel_decoder_transformer

_DATA_PACKAGE = "parallel_decoder_transformer.data"
_DATASETS_PACKAGE = "parallel_decoder_transformer.datasets"

if _DATA_PACKAGE not in sys.modules:
    data_pkg = types.ModuleType(_DATA_PACKAGE)
    data_pkg.__path__ = [
        str(Path(__file__).resolve().parents[2] / "src" / "parallel_decoder_transformer" / "data")
    ]
    sys.modules[_DATA_PACKAGE] = data_pkg
    setattr(parallel_decoder_transformer, "data", data_pkg)

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

plan_contract_notes = importlib.import_module(f"{_DATASETS_PACKAGE}.plan_contract_notes")
derive_initial_notes_from_plan = plan_contract_notes.derive_initial_notes_from_plan


def _base_plan() -> dict[str, object]:
    ranges = [("A", "H"), ("I", "P"), ("Q", "Z")]
    streams = []
    for idx, (start, end) in enumerate(ranges, start=1):
        streams.append(
            {
                "stream_id": f"stream_{idx}",
                "header": f"Header {idx}",
                "summary": f"Summary {idx}",
                "entities": [],
                "constraints": [],
                "notes_contract": [f"COVERAGE: plan_item_id=section_{idx}"],
                "section_contract": {
                    "type": "alphabet_range",
                    "start": start,
                    "end": end,
                },
            }
        )
    return {"streams": streams}


def test_answer_hint_from_notes_contract_adds_final_answer_fact() -> None:
    plan = _base_plan()
    plan_streams = plan["streams"]  # type: ignore[index]
    plan_streams[2]["notes_contract"] = [
        "Ensure the final answer is restated in this stream",
    ]  # type: ignore[index]
    notes = derive_initial_notes_from_plan(plan, input_text="Context with answer text")
    answer_stream = notes[2]
    assert any(entity.type == "final_answer" for entity in answer_stream.entities)
    assert any(fact.predicate == "answers" for fact in answer_stream.facts)


def test_answer_hint_from_coverage_plan_item_adds_fact() -> None:
    plan = _base_plan()
    plan_streams = plan["streams"]  # type: ignore[index]
    plan_streams[2]["notes_contract"] = [
        "COVERAGE: plan_item_id=final_answer, status=covered",
    ]  # type: ignore[index]
    notes = derive_initial_notes_from_plan(plan, input_text="Answer spans the conclusion")
    answer_stream = notes[2]
    assert any(entity.id.endswith("_answer") for entity in answer_stream.entities)
    assert any(fact.predicate == "answers" for fact in answer_stream.facts)
