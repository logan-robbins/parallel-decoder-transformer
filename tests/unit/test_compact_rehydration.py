"""Unit tests for compact array rehydration."""

import pytest

from parallel_decoder_transformer.datasets.config import (
    GenerationConfig,
    LLMConfig,
    SpeculativeNotesNoiseConfig,
)
from parallel_decoder_transformer.datasets.notes_generation import (
    NotesGenerationConfig,
    NotesGenerator,
)


def _create_stub_generator(tmp_path, monkeypatch):
    """Create a stub generator for testing."""
    import parallel_decoder_transformer.utils.llm_client_factory as llm_factory
    import parallel_decoder_transformer.datasets.async_llm as async_llm_module
    from types import SimpleNamespace

    dummy_client = SimpleNamespace(model="test-model", api_key="test-key")

    monkeypatch.setattr(llm_factory, "create_llm_client", lambda cfg: dummy_client)
    monkeypatch.setattr(
        async_llm_module,
        "AsyncStructuredLLMClient",
        lambda client: SimpleNamespace(submit_batch=lambda *args, **kwargs: []),
    )

    llm_cfg = LLMConfig()
    true_cfg = GenerationConfig()
    spec_cfg = GenerationConfig()
    noise_cfg = SpeculativeNotesNoiseConfig()
    notes_cfg = NotesGenerationConfig(output_root=tmp_path / "notes", resume_existing=False)

    return NotesGenerator(
        llm_config=llm_cfg,
        true_cfg=true_cfg,
        speculative_cfg=spec_cfg,
        noise_cfg=noise_cfg,
        notes_cfg=notes_cfg,
    )


def test_rehydrate_compact_notes_basic(tmp_path, monkeypatch):
    """Test basic compact to verbose conversion."""
    generator = _create_stub_generator(tmp_path, monkeypatch)

    compact_notes = [
        {
            "stream_id": "stream_1",
            "ENT": [
                ["alice", "Alice Smith", "person", True, ["Alice", "A. Smith"]],
                ["bob", "Bob Jones", "person", True, []],
            ],
            "FACT": [
                ["alice", "born_in", "1990", 0.9, [0, 25, "Alice was born in 1990."]],
                ["bob", "founded", "company", 0.85, [26, 50, "Bob founded the company."]],
            ],
            "COVERAGE": [
                ["stream_1_section", "covered"],
                ["stream_1_plan_1", "partial"],
            ],
        }
    ]

    rehydrated = generator._rehydrate_compact_notes(compact_notes)

    assert len(rehydrated) == 1
    stream = rehydrated[0]

    # Check stream_id
    assert stream["stream_id"] == "stream_1"

    # Check entities
    assert len(stream["ENT"]) == 2
    alice = stream["ENT"][0]
    assert alice["id"] == "alice"
    assert alice["name"] == "Alice Smith"
    assert alice["type"] == "person"
    assert alice["canonical"] is True
    assert alice["aliases"] == ["Alice", "A. Smith"]

    bob = stream["ENT"][1]
    assert bob["id"] == "bob"
    assert bob["aliases"] == []

    # Check facts
    assert len(stream["FACT"]) == 2
    fact1 = stream["FACT"][0]
    assert fact1["subj_id"] == "alice"
    assert fact1["predicate"] == "born_in"
    assert fact1["object"] == "1990"
    assert fact1["certainty"] == 0.9
    assert fact1["evidence_span"]["start"] == 0
    assert fact1["evidence_span"]["end"] == 25
    assert fact1["evidence_span"]["text"] == "Alice was born in 1990."

    # Check coverage
    assert len(stream["COVERAGE"]) == 2
    cov1 = stream["COVERAGE"][0]
    assert cov1["plan_item_id"] == "stream_1_section"
    assert cov1["status"] == "covered"


def test_rehydrate_compact_notes_multiple_streams(tmp_path, monkeypatch):
    """Test rehydration with multiple streams."""
    generator = _create_stub_generator(tmp_path, monkeypatch)

    compact_notes = [
        {
            "stream_id": "stream_1",
            "ENT": [["e1", "Entity 1", "type1", True, []]],
            "FACT": [["e1", "relates", "something", 0.8, [0, 10, "text here"]]],
            "COVERAGE": [["item1", "covered"]],
        },
        {
            "stream_id": "stream_2",
            "ENT": [["e2", "Entity 2", "type2", False, ["alias"]]],
            "FACT": [["e2", "describes", "other", 0.7, [10, 20, "more text"]]],
            "COVERAGE": [["item2", "partial"]],
        },
    ]

    rehydrated = generator._rehydrate_compact_notes(compact_notes)

    assert len(rehydrated) == 2
    assert rehydrated[0]["stream_id"] == "stream_1"
    assert rehydrated[1]["stream_id"] == "stream_2"
    assert rehydrated[1]["ENT"][0]["canonical"] is False
    assert rehydrated[1]["ENT"][0]["aliases"] == ["alias"]


def test_rehydrate_compact_notes_missing_stream_id(tmp_path, monkeypatch):
    """Test that missing stream_id raises error."""
    generator = _create_stub_generator(tmp_path, monkeypatch)

    compact_notes = [
        {
            # Missing stream_id
            "ENT": [],
            "FACT": [],
            "COVERAGE": [],
        }
    ]

    with pytest.raises(ValueError, match="missing stream_id"):
        generator._rehydrate_compact_notes(compact_notes)


def test_rehydrate_compact_notes_malformed_entity(tmp_path, monkeypatch):
    """Test that malformed entity array raises error."""
    generator = _create_stub_generator(tmp_path, monkeypatch)

    # Wrong number of elements
    compact_notes = [
        {
            "stream_id": "stream_1",
            "ENT": [
                ["id", "name", "type"],  # Missing canonical and aliases
            ],
            "FACT": [],
            "COVERAGE": [],
        }
    ]

    with pytest.raises(ValueError, match="must have 5 elements"):
        generator._rehydrate_compact_notes(compact_notes)


def test_rehydrate_compact_notes_malformed_fact(tmp_path, monkeypatch):
    """Test that malformed fact array raises error."""
    generator = _create_stub_generator(tmp_path, monkeypatch)

    # Wrong number of elements in fact
    compact_notes = [
        {
            "stream_id": "stream_1",
            "ENT": [],
            "FACT": [
                ["subj", "pred", "obj"],  # Missing certainty and evidence
            ],
            "COVERAGE": [],
        }
    ]

    with pytest.raises(ValueError, match="must have 5 elements"):
        generator._rehydrate_compact_notes(compact_notes)


def test_rehydrate_compact_notes_malformed_evidence_span(tmp_path, monkeypatch):
    """Test that malformed evidence_span raises error."""
    generator = _create_stub_generator(tmp_path, monkeypatch)

    # Wrong number of elements in evidence_span
    compact_notes = [
        {
            "stream_id": "stream_1",
            "ENT": [],
            "FACT": [
                ["subj", "pred", "obj", 0.9, [0, 10]],  # Missing text
            ],
            "COVERAGE": [],
        }
    ]

    with pytest.raises(ValueError, match="evidence_span must have 3 elements"):
        generator._rehydrate_compact_notes(compact_notes)


def test_rehydrate_compact_notes_malformed_coverage(tmp_path, monkeypatch):
    """Test that malformed coverage array raises error."""
    generator = _create_stub_generator(tmp_path, monkeypatch)

    # Wrong number of elements
    compact_notes = [
        {
            "stream_id": "stream_1",
            "ENT": [],
            "FACT": [],
            "COVERAGE": [
                ["plan_item_id"],  # Missing status
            ],
        }
    ]

    with pytest.raises(ValueError, match="must have 2 elements"):
        generator._rehydrate_compact_notes(compact_notes)


def test_rehydrate_compact_notes_empty_arrays(tmp_path, monkeypatch):
    """Test that empty arrays are handled correctly."""
    generator = _create_stub_generator(tmp_path, monkeypatch)

    compact_notes = [
        {
            "stream_id": "stream_1",
            "ENT": [],
            "FACT": [],
            "COVERAGE": [],
        }
    ]

    rehydrated = generator._rehydrate_compact_notes(compact_notes)

    assert len(rehydrated) == 1
    assert rehydrated[0]["ENT"] == []
    assert rehydrated[0]["FACT"] == []
    assert rehydrated[0]["COVERAGE"] == []


def test_rehydrate_compact_notes_type_conversions(tmp_path, monkeypatch):
    """Test that type conversions are applied correctly."""
    generator = _create_stub_generator(tmp_path, monkeypatch)

    # Use numeric/boolean types that might come from JSON
    compact_notes = [
        {
            "stream_id": "stream_1",
            "ENT": [
                [123, 456, "type", 1, []],  # IDs as numbers, canonical as 1
            ],
            "FACT": [
                [
                    "subj",
                    "pred",
                    789,
                    "0.95",
                    [0, 10, "text"],
                ],  # Object as number, certainty as string
            ],
            "COVERAGE": [],
        }
    ]

    rehydrated = generator._rehydrate_compact_notes(compact_notes)

    # All should be converted to proper types
    assert rehydrated[0]["ENT"][0]["id"] == "123"
    assert rehydrated[0]["ENT"][0]["name"] == "456"
    assert rehydrated[0]["ENT"][0]["canonical"] is True

    assert rehydrated[0]["FACT"][0]["object"] == "789"
    assert rehydrated[0]["FACT"][0]["certainty"] == 0.95
    assert isinstance(rehydrated[0]["FACT"][0]["certainty"], float)
