"""Integration tests for procedural snapshot generation."""

import pytest

from parallel_decoder_transformer.data.extraction import (
    CoverageSignal,
    CoverageStatus,
    EntityCard,
    EvidenceSpan,
    FactStatement,
    StreamNotes,
)
from parallel_decoder_transformer.datasets.procedural_snapshots import (
    generate_procedural_snapshots,
)


def test_procedural_snapshots_basic():
    """Test basic procedural snapshot generation with ordered facts."""
    # Create a simple text with clear evidence positions
    z_n = "Alice was born in 1990. Bob founded the company in 2010. Charlie became CEO in 2020."

    # Create facts with evidence spans at different positions
    facts = [
        FactStatement(
            subj_id="alice",
            predicate="born_in",
            object="1990",
            evidence_span=EvidenceSpan(start=0, end=25, text="Alice was born in 1990."),
            certainty=0.9,
        ),
        FactStatement(
            subj_id="bob",
            predicate="founded",
            object="company",
            evidence_span=EvidenceSpan(start=26, end=60, text="Bob founded the company in 2010."),
            certainty=0.9,
        ),
        FactStatement(
            subj_id="charlie",
            predicate="became_ceo",
            object="2020",
            evidence_span=EvidenceSpan(start=61, end=89, text="Charlie became CEO in 2020."),
            certainty=0.9,
        ),
    ]

    entities = [
        EntityCard(id="alice", name="Alice", aliases=[], type="person", canonical=True),
        EntityCard(id="bob", name="Bob", aliases=[], type="person", canonical=True),
        EntityCard(id="charlie", name="Charlie", aliases=[], type="person", canonical=True),
    ]

    coverage = [
        CoverageSignal(plan_item_id="stream_1_section", status=CoverageStatus.COVERED),
    ]

    notes = [
        StreamNotes(
            stream_id="stream_1",
            entities=entities,
            facts=facts,
            coverage=coverage,
        )
    ]

    # Generate snapshots with cadence of 3 (should create ~3 snapshots)
    snapshots = generate_procedural_snapshots(
        final_notes=notes,
        z_n=z_n,
        note_cadence_M=3,
        lag_delta=1,
    )

    # Verify we got multiple snapshots
    assert len(snapshots) > 1, "Should generate multiple snapshots"

    # Verify snapshot structure
    for snapshot in snapshots:
        assert "snapshot_id" in snapshot
        assert "source" in snapshot
        assert snapshot["source"] == "procedural_bus"
        assert "lag_delta" in snapshot
        assert "note_cadence_M" in snapshot
        assert "ent_count" in snapshot
        assert "fact_count" in snapshot
        assert "notes" in snapshot

    # Verify cumulative behavior: each snapshot should have >= facts than previous
    for i in range(1, len(snapshots)):
        prev_facts = snapshots[i - 1]["fact_count"]
        curr_facts = snapshots[i]["fact_count"]
        assert curr_facts >= prev_facts, f"Snapshot {i} should have >= facts than snapshot {i-1}"

    # Final snapshot should have all facts
    final_snapshot = snapshots[-1]
    assert final_snapshot["fact_count"] == 3, "Final snapshot should contain all facts"
    assert final_snapshot["ent_count"] == 3, "Final snapshot should contain all entities"


def test_procedural_snapshots_empty_text():
    """Test handling of empty text."""
    facts = [
        FactStatement(
            subj_id="test",
            predicate="test",
            object="test",
            evidence_span=EvidenceSpan(start=0, end=4, text="test"),
            certainty=0.9,
        ),
    ]

    entities = [
        EntityCard(id="test", name="Test", aliases=[], type="entity", canonical=True),
    ]

    notes = [
        StreamNotes(
            stream_id="stream_1",
            entities=entities,
            facts=facts,
            coverage=[],
        )
    ]

    # Empty text should still create one snapshot with all facts
    snapshots = generate_procedural_snapshots(
        final_notes=notes,
        z_n="",
        note_cadence_M=3,
        lag_delta=1,
    )

    assert len(snapshots) == 1, "Should create single snapshot for empty text"
    assert snapshots[0]["fact_count"] == 1


def test_procedural_snapshots_missing_evidence_span():
    """Test that missing evidence spans raise an error."""
    # Create a fact without evidence_span
    facts = [
        FactStatement(
            subj_id="test",
            predicate="test",
            object="test",
            evidence_span=None,  # Missing!
            certainty=0.9,
        ),
    ]

    notes = [
        StreamNotes(
            stream_id="stream_1",
            entities=[],
            facts=facts,
            coverage=[],
        )
    ]

    # Should raise ValueError for missing evidence_span
    with pytest.raises(ValueError, match="missing evidence_span"):
        generate_procedural_snapshots(
            final_notes=notes,
            z_n="Some text",
            note_cadence_M=3,
            lag_delta=1,
        )


def test_procedural_snapshots_multiple_streams():
    """Test snapshot generation with multiple independent streams."""
    z_n = "Stream 1 content here. Stream 2 different content. Stream 3 final content."

    stream_1_facts = [
        FactStatement(
            subj_id="s1_ent",
            predicate="describes",
            object="content",
            evidence_span=EvidenceSpan(start=0, end=22, text="Stream 1 content here."),
            certainty=0.9,
        ),
    ]

    stream_2_facts = [
        FactStatement(
            subj_id="s2_ent",
            predicate="describes",
            object="different",
            evidence_span=EvidenceSpan(start=23, end=50, text="Stream 2 different content."),
            certainty=0.9,
        ),
    ]

    stream_3_facts = [
        FactStatement(
            subj_id="s3_ent",
            predicate="describes",
            object="final",
            evidence_span=EvidenceSpan(start=51, end=75, text="Stream 3 final content."),
            certainty=0.9,
        ),
    ]

    notes = [
        StreamNotes(
            stream_id="stream_1",
            entities=[
                EntityCard(id="s1_ent", name="S1", aliases=[], type="section", canonical=True)
            ],
            facts=stream_1_facts,
            coverage=[],
        ),
        StreamNotes(
            stream_id="stream_2",
            entities=[
                EntityCard(id="s2_ent", name="S2", aliases=[], type="section", canonical=True)
            ],
            facts=stream_2_facts,
            coverage=[],
        ),
        StreamNotes(
            stream_id="stream_3",
            entities=[
                EntityCard(id="s3_ent", name="S3", aliases=[], type="section", canonical=True)
            ],
            facts=stream_3_facts,
            coverage=[],
        ),
    ]

    snapshots = generate_procedural_snapshots(
        final_notes=notes,
        z_n=z_n,
        note_cadence_M=3,
        lag_delta=1,
    )

    # Verify all snapshots maintain 3 streams
    for snapshot in snapshots:
        assert len(snapshot["notes"]) == 3, "Each snapshot should contain 3 streams"

    # Final snapshot should have facts from all streams
    final_snapshot = snapshots[-1]
    assert final_snapshot["fact_count"] == 3, "Final snapshot should have all facts"


def test_procedural_snapshots_sectional_independence():
    """Test that snapshots respect sectional independence."""
    # Text with three distinct sections that don't reference each other
    z_n = (
        "Section A discusses topic A with details about A. "
        "Section B covers topic B independently. "
        "Section C concludes with topic C information."
    )

    facts_a = [
        FactStatement(
            subj_id="topic_a",
            predicate="discusses",
            object="details",
            evidence_span=EvidenceSpan(
                start=0, end=50, text="Section A discusses topic A with details about A."
            ),
            certainty=0.9,
        ),
    ]

    facts_b = [
        FactStatement(
            subj_id="topic_b",
            predicate="covers",
            object="independently",
            evidence_span=EvidenceSpan(
                start=51, end=92, text="Section B covers topic B independently."
            ),
            certainty=0.9,
        ),
    ]

    facts_c = [
        FactStatement(
            subj_id="topic_c",
            predicate="concludes",
            object="information",
            evidence_span=EvidenceSpan(
                start=93, end=140, text="Section C concludes with topic C information."
            ),
            certainty=0.9,
        ),
    ]

    notes = [
        StreamNotes(
            stream_id="stream_a",
            entities=[
                EntityCard(id="topic_a", name="Topic A", aliases=[], type="topic", canonical=True)
            ],
            facts=facts_a,
            coverage=[],
        ),
        StreamNotes(
            stream_id="stream_b",
            entities=[
                EntityCard(id="topic_b", name="Topic B", aliases=[], type="topic", canonical=True)
            ],
            facts=facts_b,
            coverage=[],
        ),
        StreamNotes(
            stream_id="stream_c",
            entities=[
                EntityCard(id="topic_c", name="Topic C", aliases=[], type="topic", canonical=True)
            ],
            facts=facts_c,
            coverage=[],
        ),
    ]

    snapshots = generate_procedural_snapshots(
        final_notes=notes,
        z_n=z_n,
        note_cadence_M=3,
        lag_delta=1,
    )

    # Early snapshots should have fewer facts than later ones (cumulative)
    assert snapshots[0]["fact_count"] < snapshots[-1]["fact_count"]

    # Each stream should maintain its identity across snapshots
    for snapshot in snapshots:
        stream_ids = {stream["stream_id"] for stream in snapshot["notes"]}
        assert stream_ids == {"stream_a", "stream_b", "stream_c"}
