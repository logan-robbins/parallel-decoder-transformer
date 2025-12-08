"""Unit tests for parallel per-stream note generation."""

import pytest

from parallel_decoder_transformer.data.extraction import (
    EvidenceSpan,
    FactStatement,
    StreamNotes,
)
from parallel_decoder_transformer.datasets.notes_generation import (
    NotesGenerator,
    NotesGenerationConfig,
)
from parallel_decoder_transformer.datasets.config import (
    GenerationConfig,
    SpeculativeNotesNoiseConfig,
)


class TestExtractStreamSlice:
    """Test _extract_stream_slice() method."""

    @pytest.fixture
    def generator(self):
        """Create minimal NotesGenerator for testing."""
        # Create mock config
        llm_config = type("Config", (), {})()
        true_cfg = GenerationConfig(
            max_new_tokens=1000,
            temperature=0.7,
            top_p=0.95,
            stop_sequences=[],
            seed=42,
        )
        spec_cfg = GenerationConfig(
            max_new_tokens=1000,
            temperature=0.9,
            top_p=0.95,
            stop_sequences=[],
            seed=43,
        )
        noise_cfg = SpeculativeNotesNoiseConfig(
            paraphrase_ratio=0.15,
            drop_ratio=0.1,
            hallucination_ratio=0.05,
        )
        notes_cfg = NotesGenerationConfig(
            batch_size=10,
            concurrency=5,
            variants_per_sample=3,
            output_root="outputs/test",
            resume_existing=False,
            max_workers=4,
        )

        # Mock LLM client
        llm_config.backend = "test"
        llm_config.openai = type("OpenAI", (), {"reasoning_effort": "low"})()

        # We can't fully instantiate without a real client, so we'll test methods directly
        return None  # Will test via isolated method calls

    def test_extract_valid_slice(self):
        """Test extracting a valid text slice."""
        input_text = "Maienfeld is a municipality in the district of Landquart in the Swiss canton of Graubünden."
        section_contract = {
            "type": "source_slice",
            "start_idx": 0,
            "end_idx": 50,
        }

        # Create generator instance (simplified)
        gen = type("Gen", (), {})()
        gen._extract_stream_slice = NotesGenerator._extract_stream_slice.__get__(gen, type(gen))

        sliced_text, offset = gen._extract_stream_slice(input_text, section_contract)

        assert sliced_text == "Maienfeld is a municipality in the district of Lan"
        assert offset == 0

    def test_extract_middle_slice(self):
        """Test extracting a slice from the middle of text."""
        input_text = "Maienfeld is a municipality in the district of Landquart in the Swiss canton of Graubünden."
        section_contract = {
            "type": "source_slice",
            "start_idx": 20,
            "end_idx": 60,
        }

        gen = type("Gen", (), {})()
        gen._extract_stream_slice = NotesGenerator._extract_stream_slice.__get__(gen, type(gen))

        sliced_text, offset = gen._extract_stream_slice(input_text, section_contract)

        assert sliced_text == "ipality in the district of Landquart in "
        assert offset == 20

    def test_extract_invalid_contract_type(self):
        """Test that unsupported contract types raise ValueError."""
        input_text = "Test text"
        section_contract = {
            "type": "invalid_type",
            "start_idx": 0,
            "end_idx": 5,
        }

        gen = type("Gen", (), {})()
        gen._extract_stream_slice = NotesGenerator._extract_stream_slice.__get__(gen, type(gen))

        with pytest.raises(ValueError, match="Unsupported section_contract type"):
            gen._extract_stream_slice(input_text, section_contract)

    def test_extract_invalid_indices(self):
        """Test that invalid indices raise ValueError."""
        input_text = "Test text with exactly 29 chars"

        gen = type("Gen", (), {})()
        gen._extract_stream_slice = NotesGenerator._extract_stream_slice.__get__(gen, type(gen))

        # Negative start_idx
        with pytest.raises(ValueError, match="start_idx must be non-negative"):
            gen._extract_stream_slice(
                input_text,
                {
                    "type": "source_slice",
                    "start_idx": -1,
                    "end_idx": 10,
                },
            )

        # end_idx > length
        with pytest.raises(ValueError, match="exceeds input_text length"):
            gen._extract_stream_slice(
                input_text,
                {
                    "type": "source_slice",
                    "start_idx": 0,
                    "end_idx": 100,
                },
            )

        # start_idx >= end_idx
        with pytest.raises(ValueError, match="must be < end_idx"):
            gen._extract_stream_slice(
                input_text,
                {
                    "type": "source_slice",
                    "start_idx": 10,
                    "end_idx": 10,
                },
            )

    def test_extract_boundary_cases(self):
        """Test boundary cases for slice extraction."""
        input_text = "0123456789"

        gen = type("Gen", (), {})()
        gen._extract_stream_slice = NotesGenerator._extract_stream_slice.__get__(gen, type(gen))

        # Full text
        sliced, offset = gen._extract_stream_slice(
            input_text,
            {
                "type": "source_slice",
                "start_idx": 0,
                "end_idx": 10,
            },
        )
        assert sliced == "0123456789"
        assert offset == 0

        # Single character
        sliced, offset = gen._extract_stream_slice(
            input_text,
            {
                "type": "source_slice",
                "start_idx": 5,
                "end_idx": 6,
            },
        )
        assert sliced == "5"
        assert offset == 5

    def test_extract_rejects_empty_input_text(self):
        """Ensure empty plan text raises ValueError."""
        input_text = "   "
        gen = type("Gen", (), {})()
        gen._extract_stream_slice = NotesGenerator._extract_stream_slice.__get__(gen, type(gen))

        with pytest.raises(ValueError, match="input_text is empty"):
            gen._extract_stream_slice(
                input_text,
                {
                    "type": "source_slice",
                    "start_idx": 0,
                    "end_idx": 1,
                },
            )

    def test_extract_rejects_empty_slice(self):
        """Ensure whitespace-only slices raise ValueError."""
        input_text = "Title     Body"
        gen = type("Gen", (), {})()
        gen._extract_stream_slice = NotesGenerator._extract_stream_slice.__get__(gen, type(gen))

        with pytest.raises(ValueError, match="Extracted slice"):
            gen._extract_stream_slice(
                input_text,
                {
                    "type": "source_slice",
                    "start_idx": 5,
                    "end_idx": 10,
                },
            )


class TestRemapEvidenceIndices:
    """Test _remap_evidence_indices() method."""

    def test_remap_single_fact(self):
        """Test remapping evidence indices for a single fact."""
        evidence_span = EvidenceSpan(start=10, end=20, text="test quote")
        fact = FactStatement(
            subj_id="E1",
            predicate="located_in",
            object="Graubünden",
            evidence_span=evidence_span,
            certainty=1.0,
        )
        notes = StreamNotes(
            stream_id="stream_1",
            entities=[],
            facts=[fact],
            coverage=[],
        )

        gen = type("Gen", (), {})()
        gen._remap_evidence_indices = NotesGenerator._remap_evidence_indices.__get__(gen, type(gen))

        remapped = gen._remap_evidence_indices(notes, offset=100)

        assert remapped.facts[0].evidence_span.start == 110
        assert remapped.facts[0].evidence_span.end == 120
        assert remapped.facts[0].evidence_span.text == "test quote"

    def test_remap_multiple_facts(self):
        """Test remapping evidence indices for multiple facts."""
        facts = [
            FactStatement(
                subj_id=f"E{i}",
                predicate="test",
                object="obj",
                evidence_span=EvidenceSpan(start=i * 10, end=i * 10 + 5, text=f"quote{i}"),
                certainty=1.0,
            )
            for i in range(3)
        ]
        notes = StreamNotes(
            stream_id="stream_2",
            entities=[],
            facts=facts,
            coverage=[],
        )

        gen = type("Gen", (), {})()
        gen._remap_evidence_indices = NotesGenerator._remap_evidence_indices.__get__(gen, type(gen))

        remapped = gen._remap_evidence_indices(notes, offset=500)

        assert remapped.facts[0].evidence_span.start == 500
        assert remapped.facts[1].evidence_span.start == 510
        assert remapped.facts[2].evidence_span.start == 520

    def test_remap_zero_offset(self):
        """Test that zero offset leaves indices unchanged."""
        evidence_span = EvidenceSpan(start=42, end=100, text="test")
        fact = FactStatement(
            subj_id="E1",
            predicate="test",
            object="obj",
            evidence_span=evidence_span,
            certainty=1.0,
        )
        notes = StreamNotes(
            stream_id="stream_1",
            entities=[],
            facts=[fact],
            coverage=[],
        )

        gen = type("Gen", (), {})()
        gen._remap_evidence_indices = NotesGenerator._remap_evidence_indices.__get__(gen, type(gen))

        remapped = gen._remap_evidence_indices(notes, offset=0)

        assert remapped.facts[0].evidence_span.start == 42
        assert remapped.facts[0].evidence_span.end == 100


class TestMergeStreamResponses:
    """Test _merge_stream_responses() method."""

    def test_merge_three_streams(self):
        """Test merging responses from 3 streams."""
        stream_responses = [
            {
                "response": type(
                    "Response",
                    (),
                    {
                        "parsed_json": {
                            "stream_id": "stream_1",
                            "ENT": [["E1", "Maienfeld", "municipality", True, []]],
                            "FACT": [["E1", "located_in", "Graubünden", 1.0, [10, 20, "quote1"]]],
                            "COVERAGE": [["item_1", "covered"]],
                        }
                    },
                )(),
                "stream_id": "stream_1",
                "stream_idx": 0,
                "slice_offset": 0,
            },
            {
                "response": type(
                    "Response",
                    (),
                    {
                        "parsed_json": {
                            "stream_id": "stream_2",
                            "ENT": [["E2", "Alps", "location", True, []]],
                            "FACT": [["E2", "surrounds", "town", 1.0, [5, 15, "quote2"]]],
                            "COVERAGE": [["item_2", "covered"]],
                        }
                    },
                )(),
                "stream_id": "stream_2",
                "stream_idx": 1,
                "slice_offset": 1000,
            },
            {
                "response": type(
                    "Response",
                    (),
                    {
                        "parsed_json": {
                            "stream_id": "stream_3",
                            "ENT": [["E3", "Canton", "region", True, []]],
                            "FACT": [["E3", "contains", "district", 1.0, [8, 18, "quote3"]]],
                            "COVERAGE": [["item_3", "covered"]],
                        }
                    },
                )(),
                "stream_id": "stream_3",
                "stream_idx": 2,
                "slice_offset": 2000,
            },
        ]

        plan = type(
            "Plan",
            (),
            {
                "sample_id": "test_plan_001",
                "payload": {},
            },
        )()

        gen = type("Gen", (), {})()
        gen._merge_stream_responses = NotesGenerator._merge_stream_responses.__get__(gen, type(gen))

        merged = gen._merge_stream_responses(stream_responses, plan)

        assert len(merged) == 3
        assert merged[0]["stream_id"] == "stream_1"
        assert merged[1]["stream_id"] == "stream_2"
        assert merged[2]["stream_id"] == "stream_3"

        # Check evidence remapping for stream 2 (offset=1000)
        fact_stream_2 = merged[1]["FACT"][0]
        assert fact_stream_2["evidence_span"]["start"] == 1005  # start + offset
        assert fact_stream_2["evidence_span"]["end"] == 1015  # end + offset

        # Check evidence remapping for stream 3 (offset=2000)
        fact_stream_3 = merged[2]["FACT"][0]
        assert fact_stream_3["evidence_span"]["start"] == 2008
        assert fact_stream_3["evidence_span"]["end"] == 2018


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
