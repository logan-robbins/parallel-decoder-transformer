from __future__ import annotations

from typing import Dict, Tuple

import torch

from parallel_decoder_transformer.inference import (
    DecodeConfig,
    DynamicNotesBus,
    DynamicNotesBusConfig,
    GateAnnealingConfig,
    InferenceConfig,
    NotesWindowBuilder,
)
from parallel_decoder_transformer.inference.state import StreamState
from parallel_decoder_transformer.inference.window import TopologyMask

from ._sectional_test_utils import embed_plan_notes, load_plan_contract_notes_module

_plan_contract_notes = load_plan_contract_notes_module()
derive_initial_notes_from_plan = _plan_contract_notes.derive_initial_notes_from_plan
build_versioned_note_snapshots = _plan_contract_notes.build_versioned_note_snapshots


def _qa_plan() -> Dict[str, object]:
    """Plan that assigns the final answer section to the third stream."""

    return {
        "sectional_independence": True,
        "note_cadence_M": 2,
        "expected_dnb_lag_delta": 0,
        "streams": [
            {
                "stream_id": "stream_context",
                "header": "Context setup",
                "summary": "Restate the question and surface entities.",
                "entities": ["ENT(id=question,name=QA Question,type=query)"],
                "constraints": [
                    "FACT(subj_id=question,predicate=asks,object='What is the capital of France?')",
                ],
                "section_contract": {"type": "qa_section", "segment": "question"},
                "notes_contract": [
                    "COVERAGE(plan_item_id=qa_question,status=covered)",
                    "ENT(id=context_city,name=France,type=country)",
                ],
            },
            {
                "stream_id": "stream_evidence",
                "header": "Evidence gathering",
                "summary": "Retrieve supporting facts.",
                "entities": ["ENT(id=source,name=Encyclopedia,type=document)"],
                "constraints": [
                    "FACT(subj_id=source,predicate=states,object='Paris is the capital of France')",
                ],
                "section_contract": {"type": "qa_section", "segment": "evidence"},
                "notes_contract": [
                    "COVERAGE(plan_item_id=qa_evidence,status=covered)",
                ],
            },
            {
                "stream_id": "stream_answer",
                "header": "Direct answer",
                "summary": "Provide the final answer explicitly.",
                "entities": ["ENT(id=answer_city,name=Paris,type=city)"],
                "constraints": [],
                "section_contract": {"type": "qa_section", "segment": "final_answer"},
                "notes_contract": [
                    "Ensure the final answer is spelled out in this stream.",
                    "FACT: subj_id=answer_city,predicate=answers,object='Paris is the capital of France'",
                    "COVERAGE: plan_item_id=final_answer,status=covered",
                ],
            },
        ],
    }


def _make_state(stream: str) -> StreamState:
    tokens = torch.tensor([[3, 4]], dtype=torch.long)
    mask = torch.ones_like(tokens)
    return StreamState(
        stream=stream,
        input_ids=tokens.clone(),
        attention_mask=mask.clone(),
        commit_stride=2,
        commit_horizon=4,
    )


def _seed_buses(
    streams: Tuple[str, ...],
    vectors: Dict[str, torch.Tensor],
    *,
    notes_dim: int,
) -> Dict[str, DynamicNotesBus]:
    buses: Dict[str, DynamicNotesBus] = {}
    for stream in streams:
        bus = DynamicNotesBus(DynamicNotesBusConfig(snapshot_dim=notes_dim, max_snapshots=4, lag=0))
        vector = vectors[stream].view(1, 1, -1)
        bus.push(vector, stride=0)
        bus.push(torch.zeros_like(vector), stride=0)
        buses[stream] = bus
    return buses


def test_answer_stream_retains_plan_snapshot_and_semantics() -> None:
    plan = _qa_plan()
    context = "France's capital city is Paris, according to the briefing."
    derived_notes = derive_initial_notes_from_plan(plan, input_text=context)
    streams = ("context", "evidence", "answer")
    notes_dim = 8
    vectors = embed_plan_notes(derived_notes, notes_dim)

    answer_note = next(note for note in derived_notes if note.stream_id == "stream_answer")
    assert any(entity.type == "final_answer" for entity in answer_note.entities)
    assert any(fact.predicate == "answers" and "Paris" in fact.object for fact in answer_note.facts)

    snapshots = build_versioned_note_snapshots(
        plan_seed=derived_notes,
        final_notes=derived_notes,
        lag_delta=0,
        note_cadence=2,
    )
    assert snapshots[0]["source"] == "plan_contract"
    assert snapshots[1]["source"] == "teacher_true"

    buses = _seed_buses(streams, vectors, notes_dim=notes_dim)
    states = {stream: _make_state(stream) for stream in streams}
    config = InferenceConfig(
        streams=streams,
        stride_B=2,
        commit_L=4,
        read_lag_delta=0,
        max_snapshots_K=8,
        gate_g=1.0,
        emission_cadence_M_by_stream={stream: 1 for stream in streams},
        decode=DecodeConfig(max_new_tokens=1, do_sample=False),
        gate_annealing=GateAnnealingConfig(enabled=False),
        sectional_self_tokens=3,
    )
    builder = NotesWindowBuilder.from_config(
        config,
        notes_dim,
        topology_mask=TopologyMask(streams, allow_self=True),
    )

    answer_state = states["answer"]
    baseline_window = builder.build(answer_state, buses)
    assert baseline_window.producers == ("answer",)
    baseline_vector = baseline_window.notes.clone()

    # Remove other stream buses entirely to simulate zeroing out their notes.
    buses_without_others = {"answer": buses["answer"]}
    isolated_window = builder.build(answer_state, buses_without_others)
    torch.testing.assert_close(isolated_window.notes, baseline_vector)

    # After the self-only window expires, additional snapshots (e.g., from teacher updates)
    # can be appended, but the original plan snapshot must remain available.
    buses["answer"].push(torch.full((1, 1, notes_dim), 0.25), stride=1)
    for _ in range(config.sectional_self_tokens):
        answer_state.append_token(5, past_key_values=None)
    expanded_window = builder.build(answer_state, buses)
    assert len(expanded_window.producers) >= 1
    answer_indices = [
        idx for idx, producer in enumerate(expanded_window.producers) if producer == "answer"
    ]
    assert answer_indices, "Answer stream snapshot missing after additional updates."
    torch.testing.assert_close(
        expanded_window.notes[:, answer_indices[0] : answer_indices[0] + 1, :],
        baseline_vector,
    )
