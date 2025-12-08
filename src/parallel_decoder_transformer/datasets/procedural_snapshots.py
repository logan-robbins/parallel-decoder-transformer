"""Procedural snapshot generation from final notes based on evidence spans.

This module replaces LLM-generated versioned_notes with deterministic procedural logic.
Facts are assigned to snapshots based on their evidence_span position in the source text.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from parallel_decoder_transformer.data.extraction import StreamNotes

logger = logging.getLogger(__name__)


def generate_procedural_snapshots(
    final_notes: Sequence[StreamNotes],
    z_n: str,
    note_cadence_M: int,
    lag_delta: int,
) -> list[dict[str, Any]]:
    """
    Generate versioned_notes snapshots procedurally from final notes.

    Args:
        final_notes: Complete final notes from the LLM (ground truth)
        z_n: The generated text (used to map evidence spans to positions)
        note_cadence_M: Cadence for snapshot creation (number of characters per snapshot)
        lag_delta: Expected lag between streams

    Returns:
        List of snapshot dictionaries matching the versioned_notes schema

    Raises:
        ValueError: If any fact is missing a valid evidence_span
    """
    # Validate all facts have evidence spans
    _validate_evidence_spans(final_notes)

    # Calculate text length and number of snapshots needed
    text_length = len(z_n)
    if text_length == 0:
        logger.warning("Empty z_n text, creating single snapshot with all facts")
        return [_create_snapshot(0, final_notes, lag_delta, note_cadence_M)]

    # Determine chunk size based on text length and cadence
    # We want approximately note_cadence_M chunks of text
    chunk_size = max(1, text_length // max(1, note_cadence_M))
    num_snapshots = (text_length + chunk_size - 1) // chunk_size

    # Flatten all facts with their character positions
    positioned_facts = _extract_positioned_facts(final_notes)

    # Build cumulative snapshots
    snapshots: list[dict[str, Any]] = []
    for snapshot_id in range(num_snapshots):
        cursor = (snapshot_id + 1) * chunk_size  # End of current chunk
        visible_notes = _filter_notes_by_cursor(final_notes, positioned_facts, cursor)
        snapshot = _create_snapshot(snapshot_id, visible_notes, lag_delta, note_cadence_M)
        snapshots.append(snapshot)

    # Always ensure at least one snapshot exists (final state)
    if not snapshots:
        snapshots.append(_create_snapshot(0, final_notes, lag_delta, note_cadence_M))

    return snapshots


def _validate_evidence_spans(notes: Sequence[StreamNotes]) -> None:
    """Ensure all facts have valid evidence_span fields."""
    for note in notes:
        for fact in note.facts:
            if not hasattr(fact, "evidence_span") or fact.evidence_span is None:
                raise ValueError(
                    f"Fact missing evidence_span: {fact.subj_id} {fact.predicate} {fact.object}"
                )
            if not hasattr(fact.evidence_span, "start"):
                raise ValueError(
                    f"Fact evidence_span missing 'start': {fact.subj_id} {fact.predicate}"
                )
            if fact.evidence_span.start < 0:
                raise ValueError(f"Fact evidence_span has negative start position: {fact.subj_id}")


def _extract_positioned_facts(
    notes: Sequence[StreamNotes],
) -> dict[tuple[str, str, str, str], int]:
    """
    Extract all facts with their character positions.

    Returns:
        Dictionary mapping (stream_id, subj_id, predicate, object) -> start_position
    """
    positioned: dict[tuple[str, str, str, str], int] = {}
    for note in notes:
        for fact in note.facts:
            key = (note.stream_id, fact.subj_id, fact.predicate, str(fact.object))
            positioned[key] = fact.evidence_span.start
    return positioned


def _filter_notes_by_cursor(
    all_notes: Sequence[StreamNotes],
    positioned_facts: Mapping[tuple[str, str, str, str], int],
    cursor: int,
) -> list[StreamNotes]:
    """
    Filter notes to include only facts whose evidence appears before the cursor.

    This creates a cumulative snapshot where facts "appear" when their evidence is seen.
    """
    filtered: list[StreamNotes] = []

    for note in all_notes:
        # Filter facts based on evidence position
        visible_facts = []
        for fact in note.facts:
            key = (note.stream_id, fact.subj_id, fact.predicate, str(fact.object))
            fact_position = positioned_facts.get(key, 0)
            if fact_position < cursor:
                visible_facts.append(fact)

        # Filter entities based on whether any of their facts are visible
        visible_entity_ids = {fact.subj_id for fact in visible_facts}
        visible_entities = [entity for entity in note.entities if entity.id in visible_entity_ids]

        # Filter coverage based on visible facts
        # A coverage item is visible if we have facts supporting it
        visible_coverage = list(note.coverage) if visible_facts else []

        # Create filtered stream notes
        if visible_facts or visible_entities or visible_coverage:
            filtered.append(
                StreamNotes(
                    stream_id=note.stream_id,
                    entities=visible_entities,
                    facts=visible_facts,
                    coverage=visible_coverage,
                )
            )
        else:
            # Include empty stream placeholder to maintain stream count
            filtered.append(
                StreamNotes(
                    stream_id=note.stream_id,
                    entities=[],
                    facts=[],
                    coverage=[],
                )
            )

    return filtered


def _create_snapshot(
    snapshot_id: int,
    notes: Sequence[StreamNotes],
    lag_delta: int,
    note_cadence_M: int,
) -> dict[str, Any]:
    """Create a snapshot entry matching the versioned_notes schema."""
    ent_count = sum(len(note.entities) for note in notes)
    fact_count = sum(len(note.facts) for note in notes)

    return {
        "snapshot_id": snapshot_id,
        "source": "procedural_bus",
        "lag_delta": lag_delta,
        "note_cadence_M": note_cadence_M,
        "ent_count": ent_count,
        "fact_count": fact_count,
        "notes": [note.as_dict() for note in notes],
    }


__all__ = ["generate_procedural_snapshots"]
