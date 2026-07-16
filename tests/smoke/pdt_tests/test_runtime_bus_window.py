"""Canonical addressed notes-bus and fixed-window contracts."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from pdt.config.schemas import NotesBusConfig, SNCConfig
from pdt.runtime.counterfactuals import (
    apply_anchor_swap,
    apply_bus_mutation,
    apply_source_swap,
)
from pdt.runtime.dnb_bus import DynamicNotesBus, Snapshot
from pdt.runtime.orchestrator import MultiStreamOrchestrator, _tokenize_user_prompt
from pdt.runtime.window import read_notes_lww
from pdt.sidecar.snc import SharedNotesCrossAttention


PRODUCERS = ("stream_0", "stream_1")
CODES = (0, 1, 2, 3)


class _TestCodec:
    width = 3
    num_codebooks = 4
    codes_per_codebook = 256

    @staticmethod
    def decode(indices: torch.Tensor) -> torch.Tensor:
        base = indices.to(dtype=torch.float32).sum(dim=-1, keepdim=True)
        return torch.cat((base, base + 1.0, base + 2.0), dim=-1)


class _ChatTokenizerStub:
    chat_template = "stub-template"
    pad_token_id = 0

    def __init__(self) -> None:
        self.call: dict[str, object] = {}
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.call = {"messages": messages, **kwargs}
        self.calls.append(self.call)
        return {
            "input_ids": torch.tensor([[4, 5, 6]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

    @staticmethod
    def decode(token_ids):
        return "".join(f"<{token_id}>" for token_id in token_ids)


def _bus(*, lag: int = 1) -> DynamicNotesBus:
    return DynamicNotesBus(
        NotesBusConfig(snapshot_dim=3, lag=lag, dtype="float32"),
        producers=PRODUCERS,
        device=torch.device("cpu"),
        codec=_TestCodec(),
    )


def test_runtime_uses_instruct_chat_template_with_generation_prompt() -> None:
    tokenizer = _ChatTokenizerStub()
    input_ids, attention_mask = _tokenize_user_prompt(
        tokenizer, "Explain the result.", device=torch.device("cpu")
    )

    assert input_ids.tolist() == [[4, 5, 6]]
    assert attention_mask.tolist() == [[1, 1, 1]]
    assert tokenizer.call == {
        "messages": [{"role": "user", "content": "Explain the result."}],
        "add_generation_prompt": True,
        "enable_thinking": False,
        "tokenize": True,
        "return_tensors": "pt",
        "return_dict": True,
    }


def test_structured_runtime_builds_exactly_k_addressed_private_prompts() -> None:
    tokenizer = _ChatTokenizerStub()
    orchestrator = object.__new__(MultiStreamOrchestrator)
    orchestrator.tokenizer = tokenizer
    orchestrator.device = torch.device("cpu")
    orchestrator.streams = ("stream_0", "stream_1")
    orchestrator.config = SimpleNamespace(runtime=SimpleNamespace(block_size=32))

    def capture(_self, **kwargs):
        return kwargs

    orchestrator._generate_from_tokenized_prompts = MethodType(capture, orchestrator)
    captured = orchestrator.generate_structured(
        "shared task",
        {"stream_0": "private alpha", "stream_1": "private beta"},
        {"stream_0": [[]], "stream_1": [[]]},
        max_new_tokens=32,
    )

    assert captured["max_new_tokens"] == 32
    assert list(captured["stream_prompts"]) == ["stream_0", "stream_1"]
    user_messages = [call["messages"][0]["content"] for call in tokenizer.calls]
    assert user_messages == [
        "shared task",
        "shared task\n\nPrivate observation:\n[stream_0]\nprivate alpha",
        "shared task\n\nPrivate observation:\n[stream_1]\nprivate beta",
    ]
    with pytest.raises(ValueError, match="exactly runtime.streams"):
        orchestrator.generate_structured(
            "shared task",
            {"stream_0": "private alpha"},
            {"stream_0": [[]], "stream_1": [[]]},
            max_new_tokens=32,
        )
    with pytest.raises(ValueError, match="whole number of tau-token blocks"):
        orchestrator.generate_structured(
            "shared task",
            {"stream_0": "private alpha", "stream_1": "private beta"},
            {"stream_0": [[]], "stream_1": [[]]},
            max_new_tokens=31,
        )
    with pytest.raises(ValueError, match="block_transition_ids must contain exactly"):
        orchestrator.generate_structured(
            "shared task",
            {"stream_0": "private alpha", "stream_1": "private beta"},
            {"stream_0": [[]]},
            max_new_tokens=32,
        )
    with pytest.raises(ValueError, match="exactly 2 rows"):
        orchestrator.generate_structured(
            "shared task",
            {"stream_0": "private alpha", "stream_1": "private beta"},
            {"stream_0": [[]], "stream_1": [[]]},
            max_new_tokens=64,
        )
    with pytest.raises(ValueError, match="row 0 must be empty"):
        orchestrator.generate_structured(
            "shared task",
            {"stream_0": "private alpha", "stream_1": "private beta"},
            {"stream_0": [[99], [20]], "stream_1": [[], [30]]},
            max_new_tokens=64,
        )
    with pytest.raises(ValueError, match="row 1 must be non-empty"):
        orchestrator.generate_structured(
            "shared task",
            {"stream_0": "private alpha", "stream_1": "private beta"},
            {"stream_0": [[], []], "stream_1": [[], [30]]},
            max_new_tokens=64,
        )

    tokenizer.calls.clear()
    natural = orchestrator.generate("shared-only task", max_new_tokens=16)
    assert natural["max_new_tokens"] == 16
    assert len(tokenizer.calls) == 1
    first_ids = natural["stream_prompts"]["stream_0"][0]
    second_ids = natural["stream_prompts"]["stream_1"][0]
    assert torch.equal(first_ids, second_ids)


def test_cached_scheduler_never_refeeds_prompt_and_publishes_tau_token_hidden() -> None:
    class FakeCache:
        def __init__(self, length: int) -> None:
            self.length = length

        def get_seq_length(self) -> int:
            return self.length

    class FakeLayer:
        def __init__(self) -> None:
            self.context = None

        def set_runtime_context(self, context) -> None:
            self.context = context

    class FakeTrunk:
        def __init__(self, layer: FakeLayer) -> None:
            self.layer = layer
            self.inputs: list[list[int]] = []
            self.call_shapes: list[tuple[int, int]] = []
            self.singleton_masks: list[tuple[list[int], list[list[bool]]]] = []
            self.cached_call_masks: list[tuple[list[list[int]], list[list[bool]]]] = []

        def forward(self, *, input_ids, past_key_values=None, **_kwargs):
            self.inputs.extend(input_ids.tolist())
            self.call_shapes.append(tuple(input_ids.shape))
            if input_ids.size(1) == 1:
                assert self.layer.context is not None
                self.singleton_masks.append(
                    (
                        input_ids[:, 0].tolist(),
                        self.layer.context.notes_mask.tolist(),
                    )
                )
            if past_key_values is not None:
                assert self.layer.context is not None
                self.cached_call_masks.append(
                    (input_ids.tolist(), self.layer.context.notes_mask.tolist())
                )
            hidden = input_ids.to(torch.float32).unsqueeze(-1)
            logits = torch.full((*input_ids.shape, 64), -100.0)
            next_ids = (input_ids + 1).clamp_max(63).unsqueeze(-1)
            logits.scatter_(-1, next_ids, 100.0)
            prior = 0 if past_key_values is None else past_key_values.get_seq_length()
            return SimpleNamespace(
                hidden_states=(hidden,),
                logits=logits,
                past_key_values=FakeCache(prior + input_ids.size(1)),
            )

    class FakePlanner:
        def __call__(self, hidden, *, attention_mask):
            del attention_mask
            return SimpleNamespace(
                indices=torch.zeros((1, 1), dtype=torch.long),
                quantized=hidden.new_zeros((1, 1, 1)),
                logits=hidden.new_zeros((1, 1, 2)),
            )

    class FakePlanProjection:
        def __call__(self, quantized, ownership):
            del quantized
            return ownership.new_zeros((*ownership.shape[:2], 1), dtype=torch.float32)

    class RecordingSpeculation:
        width = 1
        num_codebooks = 4
        codes_per_codebook = 256

        def __init__(self) -> None:
            self.hidden_values: list[float] = []

        def project(self, hidden):
            self.hidden_values.extend(hidden[..., 0].flatten().tolist())
            return hidden

        @staticmethod
        def quantize(projected):
            zero = projected.sum() * 0.0
            return SimpleNamespace(
                quantized=projected,
                indices=torch.arange(4, dtype=torch.long).expand(*projected.shape[:-1], 4),
                assignment_logits=torch.zeros((*projected.shape[:-1], 4, 4)),
                commitment_loss=zero,
                codebook_loss=zero,
                capacity_bits=32,
            )

        @staticmethod
        def decode(indices):
            return indices[:, :1].to(dtype=torch.float32)

    class FakeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(()))
            layer = FakeLayer()
            trunk = FakeTrunk(layer)
            speculation = RecordingSpeculation()
            self.trunk_adapter = SimpleNamespace(forward=trunk.forward, fake=trunk)
            self.sidecar = SimpleNamespace(
                planner_head=FakePlanner(),
                plan_notes_proj=FakePlanProjection(),
                speculation_head=speculation,
            )
            self.instrumented_layers = [layer]

    model = FakeModel()
    config = SimpleNamespace(
        instrumentation=SimpleNamespace(coordination_source="bus"),
        runtime=SimpleNamespace(
            streams=("stream_0", "stream_1"),
            block_size=2,
            notes_bus=NotesBusConfig(snapshot_dim=1, lag=1, dtype="float32"),
        ),
        sidecar=SimpleNamespace(
            num_streams=2,
            notes_dim=1,
            planner_head=SimpleNamespace(num_slots=1),
        ),
    )
    orchestrator = MultiStreamOrchestrator(model, _ChatTokenizerStub(), config)
    result = orchestrator.generate_structured(
        "shared",
        {"stream_0": "alpha", "stream_1": "beta"},
        {"stream_0": [[], [20, 21]], "stream_1": [[], [30, 31]]},
        max_new_tokens=4,
    )

    assert result.tokens_by_stream == {
        "stream_0": [7, 8, 22, 23],
        "stream_1": [7, 8, 32, 33],
    }
    assert not hasattr(result, "agreement_history")
    assert not hasattr(result, "rollback_events")
    assert result.dynamic_codes_by_stream == {
        "stream_0": [CODES, CODES],
        "stream_1": [CODES, CODES],
    }
    # One planner call, one K-row packed prefill, four K-row decode calls, and
    # one K-row transition call. The final prompt token 6 is never re-fed.
    assert model.trunk_adapter.fake.call_shapes == [
        (1, 3),
        (2, 3),
        (2, 1),
        (2, 1),
        (2, 2),
        (2, 1),
        (2, 1),
    ]
    assert model.trunk_adapter.fake.inputs == [
        [4, 5, 6],
        [4, 5, 6],
        [4, 5, 6],
        [7],
        [7],
        [8],
        [8],
        [20, 21],
        [30, 31],
        [22],
        [32],
        [23],
        [33],
    ]
    assert model.sidecar.speculation_head.hidden_values == [8.0, 8.0, 23.0, 33.0]
    assert (
        [[20, 21], [30, 31]],
        [[True, True, True, True], [True, True, True, True]],
    ) in model.trunk_adapter.fake.cached_call_masks
    assert model.trunk_adapter.fake.singleton_masks == [
        ([7, 7], [[True, True, False, False], [True, True, False, False]]),
        ([8, 8], [[True, True, False, False], [True, True, False, False]]),
        ([22, 32], [[True, True, True, True], [True, True, True, True]]),
        ([23, 33], [[True, True, True, True], [True, True, True, True]]),
    ]


def test_bus_lag_and_lww_window_have_fixed_addressed_slots() -> None:
    bus = _bus()
    bus.seed_anchor("stream_0", torch.tensor([10.0, 0.0, 0.0]))
    bus.seed_anchor("stream_1", torch.tensor([20.0, 0.0, 0.0]))
    bus.publish(
        "stream_0",
        published_block=0,
        stride=4,
        code_indices=(11, 0, 0, 0),
    )
    bus.publish(
        "stream_0",
        published_block=1,
        stride=8,
        code_indices=(12, 0, 0, 0),
    )

    block_zero = read_notes_lww(
        bus.delivered_updates(consumer_block=0),
        producers=PRODUCERS,
        consumer_block=0,
        notes_dim=3,
    )
    block_one = read_notes_lww(
        bus.delivered_updates(consumer_block=1),
        producers=PRODUCERS,
        consumer_block=1,
        notes_dim=3,
    )

    assert block_zero.notes.shape == (1, 4, 3)
    assert block_zero.producers == PRODUCERS + PRODUCERS
    assert block_zero.mask.tolist() == [[True, True, False, False]]
    assert block_zero.versions.tolist() == [0, 0, -1, -1]
    assert block_one.mask.tolist() == [[True, True, True, False]]
    assert block_one.versions.tolist() == [0, 0, 1, -1]
    assert block_one.published_blocks.tolist() == [-1, -1, 0, -1]
    assert block_one.lags.tolist() == [0, 0, 1, 0]
    assert block_one.notes[0, :, 0].tolist() == [10.0, 20.0, 11.0, 0.0]


def test_lww_read_is_order_independent_and_idempotent() -> None:
    bus = _bus(lag=0)
    anchor_0 = bus.seed_anchor("stream_0", torch.ones(3))
    anchor_1 = bus.seed_anchor("stream_1", torch.full((3,), 2.0))
    old = bus.publish(
        "stream_0",
        published_block=0,
        stride=4,
        code_indices=(3, 0, 0, 0),
    )
    latest = bus.publish(
        "stream_0",
        published_block=1,
        stride=8,
        code_indices=(4, 0, 0, 0),
    )

    ordered = read_notes_lww(
        (anchor_0, anchor_1, old, latest),
        producers=PRODUCERS,
        consumer_block=1,
        notes_dim=3,
    )
    reordered = read_notes_lww(
        (latest, anchor_1, old, latest, anchor_0),
        producers=PRODUCERS,
        consumer_block=1,
        notes_dim=3,
    )

    assert torch.equal(ordered.notes, reordered.notes)
    assert torch.equal(ordered.mask, reordered.mask)
    assert torch.equal(ordered.versions, reordered.versions)
    assert ordered.versions.tolist() == [0, 0, 2, -1]


def test_equal_version_conflict_and_future_delivery_fail_fast() -> None:
    current = Snapshot(
        producer="stream_0",
        version=1,
        published_block=0,
        stride=4,
        kind="dynamic",
        notes=torch.ones(3),
        code_indices=CODES,
        metadata={},
    )
    conflicting = Snapshot(
        producer="stream_0",
        version=1,
        published_block=0,
        stride=5,
        kind="dynamic",
        notes=torch.ones(3),
        code_indices=CODES,
        metadata={},
    )
    future = Snapshot(
        producer="stream_0",
        version=2,
        published_block=2,
        stride=8,
        kind="dynamic",
        notes=torch.ones(3),
        code_indices=CODES,
        metadata={},
    )

    with pytest.raises(ValueError, match="Conflicting updates"):
        read_notes_lww(
            (current, conflicting),
            producers=PRODUCERS,
            consumer_block=0,
            notes_dim=3,
        )
    with pytest.raises(ValueError, match="future block"):
        read_notes_lww(
            (future,),
            producers=PRODUCERS,
            consumer_block=1,
            notes_dim=3,
        )


def test_anchor_swap_uses_addressed_bus_api() -> None:
    bus = _bus()
    bus.seed_anchor("stream_0", torch.ones(3))
    bus.seed_anchor("stream_1", torch.full((3,), 2.0))

    apply_anchor_swap(
        bus,
        torch.tensor([[7.0, 7.0, 7.0], [8.0, 8.0, 8.0]]),
        PRODUCERS,
    )
    window = read_notes_lww(
        bus.delivered_updates(consumer_block=0),
        producers=PRODUCERS,
        consumer_block=0,
        notes_dim=3,
    )

    assert window.notes[0, :2].tolist() == [[7.0, 7.0, 7.0], [8.0, 8.0, 8.0]]


def test_source_swap_changes_only_receiver_sibling_dynamic_payloads() -> None:
    notes = torch.tensor([[[10.0], [20.0], [30.0], [11.0], [21.0], [31.0]]])
    mask = torch.ones((1, 6), dtype=torch.bool)
    donor = torch.tensor([[[40.0], [50.0], [60.0], [41.0], [51.0], [61.0]]])
    swapped, swapped_mask = apply_source_swap(
        notes,
        mask,
        anchor_mask=torch.tensor([True, True, True, False, False, False]),
        producer_indices=torch.tensor([0, 1, 2, 0, 1, 2]),
        consumer_index=0,
        donor_notes=donor,
        donor_mask=mask,
    )

    assert swapped[0, :, 0].tolist() == [10.0, 20.0, 30.0, 11.0, 51.0, 61.0]
    assert torch.equal(swapped_mask, mask)

    with pytest.raises(ValueError, match="registered donor intervention"):
        apply_source_swap(
            notes,
            mask,
            anchor_mask=torch.tensor([True, True, True, False, False, False]),
            producer_indices=torch.tensor([0, 1, 2, 0, 1, 2]),
            consumer_index=0,
        )


def test_bus_mutation_is_targeted_deterministic_and_non_aliasing() -> None:
    original = torch.tensor([[255, 7, 8, 9]], dtype=torch.long)
    mutated = apply_bus_mutation(
        original,
        codes_per_codebook=256,
        code_offset=1,
    )

    assert original.tolist() == [[255, 7, 8, 9]]
    assert mutated.tolist() == [[0, 7, 8, 9]]
    assert mutated.data_ptr() != original.data_ptr()
    with pytest.raises(ValueError, match="code_offset"):
        apply_bus_mutation(original, codes_per_codebook=256, code_offset=0)


def test_addressed_snc_distinguishes_payload_swap_but_not_slot_reordering() -> None:
    torch.manual_seed(7)
    snc = SharedNotesCrossAttention(
        SNCConfig(hidden_size=8, notes_dim=4, num_heads=2),
        num_producers=2,
    ).eval()
    torch.nn.init.normal_(snc.o_proj.weight)
    torch.nn.init.normal_(snc.o_proj.bias)
    hidden = torch.randn(1, 2, 8)
    notes = torch.randn(1, 3, 4)
    mask = torch.ones((1, 3), dtype=torch.bool)
    producer_ids = torch.tensor([[0, 1, 0]])
    kind_ids = torch.tensor([[0, 0, 1]])
    lags = torch.tensor([[0, 0, 1]])

    original = snc(
        hidden,
        notes,
        notes_mask=mask,
        producer_ids=producer_ids,
        kind_ids=kind_ids,
        lags=lags,
        force_gate=True,
    )
    payload_swapped = snc(
        hidden,
        notes[:, [2, 0, 1]],
        notes_mask=mask,
        producer_ids=producer_ids,
        kind_ids=kind_ids,
        lags=lags,
        force_gate=True,
    )
    reordered = snc(
        hidden,
        notes[:, [2, 0, 1]],
        notes_mask=mask[:, [2, 0, 1]],
        producer_ids=producer_ids[:, [2, 0, 1]],
        kind_ids=kind_ids[:, [2, 0, 1]],
        lags=lags[:, [2, 0, 1]],
        force_gate=True,
    )

    assert not torch.allclose(original, payload_swapped, atol=1e-6, rtol=1e-6)
    assert torch.allclose(original, reordered, atol=1e-6, rtol=1e-6)
