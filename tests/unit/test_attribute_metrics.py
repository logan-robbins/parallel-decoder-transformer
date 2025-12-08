from __future__ import annotations

from parallel_decoder_transformer.data.extraction import (
    EntityCard,
    EvidenceSpan,
    FactStatement,
    StreamNotes,
)
from parallel_decoder_transformer.evaluation.attributes import compute_attribute_consistency


def _make_stream_notes() -> StreamNotes:
    return StreamNotes(
        stream_id="stream_intro",
        entities=[
            EntityCard(
                id="battery-alpha",
                name="Battery Alpha",
                aliases=["Alpha Battery"],
                canonical=True,
            )
        ],
        facts=[
            FactStatement(
                subj_id="battery-alpha",
                predicate="capacity",
                object="5 GW",
                evidence_span=EvidenceSpan(start=0, end=5, text="Battery capacity is 5 GW."),
            )
        ],
    )


def test_attribute_consistency_cross_stream_and_time() -> None:
    notes_by_stream = {
        "stream_intro": ["Battery Alpha - capacity: 5 GW"],
        "stream_wrap": ["Battery Alpha - capacity: 5 GW", "Battery Alpha - capacity: 6 GW"],
    }
    plan = {
        "stream_intro": ["Battery Alpha overview"],
        "stream_wrap": ["Battery Alpha wrap"],
    }
    structured = {"stream_intro": _make_stream_notes()}

    result = compute_attribute_consistency(
        notes_by_stream,
        plan_by_stream=plan,
        structured_notes=structured,
    )
    payload = result.to_payload()

    assert payload["cross_stream"]["total"] == 1
    assert payload["cross_stream"]["violations"] == 1
    assert payload["per_stream"]["stream_wrap"]["violations"] == 1
    assert payload["time"]["stream_wrap"]["violations"] == 1
    assert payload["source_counts"]["structured"] == 1
    assert payload["source_counts"]["text"] >= 1


def test_attribute_consistency_plan_filtering() -> None:
    notes_by_stream = {
        "stream_intro": ["Battery Alpha - capacity: 5 GW"],
        "stream_wrap": ["Closing remarks - tone: optimistic"],
    }
    plan = {
        "stream_intro": ["Battery Alpha coverage"],
        "stream_wrap": ["non overlapping topic"],
    }

    result = compute_attribute_consistency(notes_by_stream, plan_by_stream=plan)
    payload = result.to_payload()

    assert payload["total_tuples"] == 1
    assert payload["cross_stream"]["total"] == 0
    assert payload["per_stream"] == {}
