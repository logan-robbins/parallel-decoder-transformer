"""Unit tests for plan and notes metadata serialisation."""

from __future__ import annotations

from parallel_decoder_transformer.data.extraction import (
    CoverageSignal,
    CoverageStatus,
    EntityCard,
    EvidenceSpan,
    FactStatement,
    StreamNotes,
)
from parallel_decoder_transformer.datasets.example import (
    DatasetExample,
    ExampleNotes,
    PlanPayload,
    PlanSection,
    SectionDecomposition,
)
from parallel_decoder_transformer.datasets.processors import ArticleProcessor


def test_plan_items_and_notes_serialisation() -> None:
    processor = ArticleProcessor.__new__(ArticleProcessor)
    plan_sections = [
        PlanSection(
            stream="stream_1",
            header="Introduction",
            summary="Summary 1",
            entities=["entity_a"],
            constraints=["constraint_a"],
        ),
        PlanSection(
            stream="stream_2",
            header="Body",
            summary="Summary 2",
            entities=["entity_b"],
            constraints=[],
        ),
        PlanSection(
            stream="stream_3",
            header="Conclusion",
            summary="Summary 3",
            entities=[],
            constraints=["constraint_c"],
        ),
    ]
    plan_payload = PlanPayload(reasoning=["inspect article"], sections=plan_sections, raw={})

    true_notes = [
        StreamNotes(
            stream_id="stream_1",
            entities=[EntityCard(id="entity_a", name="EntityA")],
            facts=[
                FactStatement(
                    subj_id="entity_a",
                    predicate="is",
                    object="B",
                    certainty=1.0,
                    evidence_span=EvidenceSpan(start=0, end=5, text="A is B."),
                )
            ],
            coverage=[CoverageSignal(plan_item_id="Summary 1", status=CoverageStatus.COVERED)],
        ),
        StreamNotes(stream_id="stream_2"),
        StreamNotes(stream_id="stream_3"),
    ]
    speculative_notes = [StreamNotes(stream_id=f"stream_{idx + 1}") for idx in range(3)]
    notes = ExampleNotes(
        true_notes=true_notes, speculative_notes=speculative_notes, speculative_reasoning=[]
    )

    plan_items = processor._plan_items_by_stream(plan_payload)
    assert plan_items["stream_1"][0] == "Summary 1"

    notes_strings = processor._notes_strings_by_stream(notes, plan_payload)
    assert "stream_2" in notes_strings

    metadata = {
        "plan_items": plan_items,
        "notes_text": {stream: " ".join(strings) for stream, strings in notes_strings.items()},
    }

    example = DatasetExample(
        example_id="example-1",
        prompt="Write about the topic",
        plan=plan_payload,
        sections=SectionDecomposition(
            texts=["A", "B", "C"],
            reasoning=["r1", "r2", "r3"],
            shared_entities=[],
        ),
        notes=notes,
        metadata=metadata,
    )

    record = example.to_record()
    assert record["plan_items"]["stream_1"][0] == "Summary 1"
    assert "stream_1" in record["notes_text"]
