"""Core dataset example dataclasses for the PDT training corpus."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True)
class PlanSection:
    """A single section entry in a teacher-generated plan."""

    stream: str
    header: str
    summary: str
    entities: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)


@dataclass(slots=True)
class PlanPayload:
    """Top-level plan produced by the planning LLM."""

    reasoning: List[str]
    sections: List[PlanSection]
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SectionDecomposition:
    """Per-stream section text outputs from the teacher."""

    texts: List[str]
    reasoning: List[str]
    shared_entities: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ExampleNotes:
    """True and speculative notes paired with an example."""

    true_notes: List[Any]  # List[StreamNotes]
    speculative_notes: List[Any]  # List[StreamNotes]
    speculative_reasoning: List[str] = field(default_factory=list)


@dataclass
class DatasetExample:
    """A single training example in the PDT KD corpus."""

    example_id: str
    prompt: str
    plan: PlanPayload
    sections: SectionDecomposition
    notes: ExampleNotes
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        """Serialise the example to a flat dict suitable for Parquet/JSONL.

        All ``metadata`` keys are merged into the top-level record so that
        downstream collators can access them without an extra nesting level.
        """
        record: Dict[str, Any] = {
            "example_id": self.example_id,
            "prompt": self.prompt,
            "plan": {
                "reasoning": self.plan.reasoning,
                "sections": [
                    {
                        "stream": s.stream,
                        "header": s.header,
                        "summary": s.summary,
                        "entities": list(s.entities),
                        "constraints": list(s.constraints),
                    }
                    for s in self.plan.sections
                ],
                "raw": self.plan.raw,
            },
            "sections": {
                "texts": list(self.sections.texts),
                "reasoning": list(self.sections.reasoning),
                "shared_entities": list(self.sections.shared_entities),
            },
            "notes": {
                "true_notes": [n.as_dict() for n in self.notes.true_notes],
                "speculative_notes": [n.as_dict() for n in self.notes.speculative_notes],
                "speculative_reasoning": list(self.notes.speculative_reasoning),
            },
        }
        record.update(self.metadata)
        return record

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "DatasetExample":
        """Deserialise from a record produced by :meth:`to_record`."""
        from parallel_decoder_transformer.data.extraction import load_stream_notes

        plan_raw = record["plan"]
        plan = PlanPayload(
            reasoning=list(plan_raw["reasoning"]),
            sections=[
                PlanSection(
                    stream=s["stream"],
                    header=s["header"],
                    summary=s["summary"],
                    entities=list(s.get("entities", [])),
                    constraints=list(s.get("constraints", [])),
                )
                for s in plan_raw["sections"]
            ],
            raw=dict(plan_raw.get("raw", {})),
        )
        sections_raw = record["sections"]
        sections = SectionDecomposition(
            texts=list(sections_raw["texts"]),
            reasoning=list(sections_raw["reasoning"]),
            shared_entities=list(sections_raw.get("shared_entities", [])),
        )
        notes_raw = record["notes"]
        notes = ExampleNotes(
            true_notes=[load_stream_notes(n) for n in notes_raw["true_notes"]],
            speculative_notes=[load_stream_notes(n) for n in notes_raw["speculative_notes"]],
            speculative_reasoning=list(notes_raw.get("speculative_reasoning", [])),
        )
        core_keys = {"example_id", "prompt", "plan", "sections", "notes"}
        metadata = {k: v for k, v in record.items() if k not in core_keys}
        return cls(
            example_id=record["example_id"],
            prompt=record["prompt"],
            plan=plan,
            sections=sections,
            notes=notes,
            metadata=metadata,
        )
