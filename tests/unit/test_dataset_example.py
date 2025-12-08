from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROTOTYPE_SRC = ROOT / "prototype" / "src"
if PROTOTYPE_SRC.exists():
    sys.path.insert(0, str(PROTOTYPE_SRC))

from parallel_decoder_transformer.data.extraction import EntityCard, StreamNotes
from parallel_decoder_transformer.datasets.example import (
    DatasetExample,
    ExampleNotes,
    PlanPayload,
    PlanSection,
    SectionDecomposition,
)


def test_dataset_example_roundtrip() -> None:
    plan = PlanPayload(
        reasoning=["Step 1", "Step 2"],
        sections=[
            PlanSection("stream_1", "Intro", "Summary A", ["Entity"], ["Constraint"]),
            PlanSection("stream_2", "Core", "Summary B", [], []),
            PlanSection("stream_3", "Wrap", "Summary C", [], []),
        ],
        raw={"plan": "raw"},
    )
    sections = SectionDecomposition(
        texts=["Intro text", "Core text", "Wrap text"],
        reasoning=["R1", "R2", "R3"],
        shared_entities=["Entity"],
    )
    true_stream_notes = [
        StreamNotes(
            stream_id=f"stream_{idx + 1}", entities=[EntityCard(id=f"ent-{idx}", name="Entity")]
        )
        for idx in range(3)
    ]
    spec_stream_notes = [
        StreamNotes(
            stream_id=f"stream_{idx + 1}", entities=[EntityCard(id=f"spec-{idx}", name="Entity")]
        )
        for idx in range(3)
    ]
    notes = ExampleNotes(
        true_notes=true_stream_notes,
        speculative_notes=spec_stream_notes,
        speculative_reasoning=["Guess"],
    )
    example = DatasetExample(
        example_id="test::1",
        prompt="Write about testing",
        plan=plan,
        sections=sections,
        notes=notes,
        metadata={"source": "unit"},
    )

    record = example.to_record()
    reconstructed = DatasetExample.from_record(record)

    assert reconstructed.example_id == example.example_id
    assert reconstructed.prompt == example.prompt
    assert reconstructed.plan.sections[0].header == "Intro"
    assert reconstructed.sections.shared_entities == ["Entity"]
    assert reconstructed.notes.true_notes[0].entities[0].name == "Entity"
