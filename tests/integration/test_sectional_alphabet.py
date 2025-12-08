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


def _alphabet_plan() -> Dict[str, object]:
    """Three-way sectional plan covering the alphabet in contiguous ranges."""

    return {
        "sectional_independence": True,
        "note_cadence_M": 4,
        "expected_dnb_lag_delta": 0,
        "streams": [
            {
                "stream_id": "stream_intro",
                "header": "Letters A–H",
                "summary": "Emit the first 8 letters.",
                "entities": [
                    "ENT(id=a_h_span,name=A through H,type=alphabet_range,canonical=True)",
                ],
                "constraints": [
                    "FACT(subj_id=a_h_span,predicate=covers,object='A-H segment')",
                ],
                "section_contract": {"type": "alphabet_range", "start": "A", "end": "H"},
                "notes_contract": [
                    "ENT(id=a_h_summary,name=Open range,type=contract)",
                    "FACT(subj_id=a_h_summary,predicate=describes,object='A to H')",
                    "COVERAGE(plan_item_id=alphabet_open,status=covered)",
                ],
            },
            {
                "stream_id": "stream_core",
                "header": "Letters I–P",
                "summary": "Cover the middle letters.",
                "entities": [
                    "ENT(id=i_p_span,name=I through P,type=alphabet_range,canonical=True)",
                ],
                "constraints": [
                    "FACT(subj_id=i_p_span,predicate=covers,object='I-P segment')",
                ],
                "section_contract": {"type": "alphabet_range", "start": "I", "end": "P"},
                "notes_contract": [
                    "FACT(subj_id=i_p_span,predicate=details,object='I to P core body')",
                    "COVERAGE(plan_item_id=alphabet_middle,status=covered)",
                ],
            },
            {
                "stream_id": "stream_wrap",
                "header": "Letters Q–Z",
                "summary": "Finish the alphabet and close.",
                "entities": [
                    "ENT(id=q_z_span,name=Q through Z,type=alphabet_range,canonical=True)",
                ],
                "constraints": [
                    "FACT(subj_id=q_z_span,predicate=covers,object='Q-Z segment')",
                ],
                "section_contract": {"type": "alphabet_range", "start": "Q", "end": "Z"},
                "notes_contract": [
                    "FACT(subj_id=q_z_span,predicate=details,object='Wrap up Q to Z')",
                    "COVERAGE(plan_item_id=alphabet_wrap,status=covered)",
                ],
            },
        ],
    }


def _make_state(stream: str) -> StreamState:
    dummy_prompt = torch.tensor([[1, 2]], dtype=torch.long)
    mask = torch.ones_like(dummy_prompt)
    return StreamState(
        stream=stream,
        input_ids=dummy_prompt.clone(),
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


def test_plan_snapshot_visible_and_stable_for_wrap_stream() -> None:
    plan = _alphabet_plan()
    derived_notes = derive_initial_notes_from_plan(plan, input_text="Alphabet task")
    assert len(derived_notes) == 3
    for note in derived_notes:
        assert note.entities, f"{note.stream_id} missing entities"
        assert note.facts, f"{note.stream_id} missing facts"
        assert note.coverage, f"{note.stream_id} missing coverage annotations"

    streams = ("intro", "core", "wrap")
    notes_dim = 8
    vectors = embed_plan_notes(derived_notes, notes_dim)
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
        sectional_self_tokens=4,
    )
    builder = NotesWindowBuilder.from_config(
        config,
        notes_dim,
        topology_mask=TopologyMask(streams, allow_self=True),
    )

    wrap_state = states["wrap"]
    window = builder.build(wrap_state, buses)
    assert window.notes.shape == (1, 1, notes_dim)
    assert window.producers == ("wrap",)
    wrap_plan = window.notes.clone()

    # Mutate producer buses for intro/core; wrap window should remain identical while it
    # is in the self-only regime (generated_count < sectional_self_tokens).
    torch.manual_seed(0)
    for producer in ("intro", "core"):
        noise = torch.randn(1, 1, notes_dim)
        buses[producer].push(noise, stride=1)

    mutated_window = builder.build(wrap_state, buses)
    torch.testing.assert_close(mutated_window.notes, wrap_plan)
    assert mutated_window.producers == ("wrap",)

    # Once the warmup tokens have elapsed, other producers become visible, but the
    # self snapshot must still be included so the plan contract is never lost.
    for _ in range(config.sectional_self_tokens):
        wrap_state.append_token(0, past_key_values=None)
    expanded_window = builder.build(wrap_state, buses)
    assert set(expanded_window.producers) >= {"intro", "core", "wrap"}
    wrap_index = expanded_window.producers.index("wrap")
    torch.testing.assert_close(
        expanded_window.notes[:, wrap_index : wrap_index + 1, :],
        wrap_plan,
    )
