"""Integration tests for the multi-stream inference orchestrator."""

from __future__ import annotations

import types
from typing import Any, Dict, Iterable, Tuple

import torch
from torch import nn

from parallel_decoder_transformer.inference import (
    DecodeConfig,
    GateAnnealingConfig,
    InferenceConfig,
    MultiStreamOrchestrator,
)


class FakeTokenizer:
    def __call__(self, text: str, *, return_tensors: str = "pt", add_special_tokens: bool = True):
        input_ids = torch.tensor([[1, 2]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(65 + (int(token_id) % 26)) for token_id in token_ids)


class FakeTrunkModel(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        with torch.no_grad():
            values = torch.arange(vocab_size * hidden_size, dtype=torch.float32).view(
                vocab_size, hidden_size
            )
            self.embed.weight.copy_(values / hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        nn.init.zeros_(self.lm_head.weight)
        for i in range(min(hidden_size, vocab_size)):
            self.lm_head.weight.data[i, i] = 1.0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool = True,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ):
        hidden = self.proj(self.embed(input_ids))
        seq_len = hidden.size(1)
        if past_key_values is None:
            zero = torch.zeros(hidden.size(0), seq_len, hidden.size(-1), device=hidden.device)
            past_key_values = ((zero.clone(), zero.clone()),)
        return types.SimpleNamespace(hidden_states=[hidden], past_key_values=past_key_values)


class FakeTrunkAdapter:
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        self.model = FakeTrunkModel(hidden_size, vocab_size)

    def load_model(self) -> None:  # pragma: no cover - compatibility hook
        return None


class FakeStreamAdapters(nn.Module):
    def __init__(self, streams: Tuple[str, ...], hidden_size: int) -> None:
        super().__init__()
        self.stream_to_index = {stream: idx for idx, stream in enumerate(streams)}
        self.bias = nn.Parameter(torch.zeros(len(streams), hidden_size))

    def forward(self, stream: str, hidden_states: torch.Tensor) -> torch.Tensor:
        idx = self.stream_to_index[stream.lower()]
        bias = self.bias[idx].view(1, 1, -1)
        return hidden_states + bias


class FakeCrossAttention(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        notes: torch.Tensor,
        *,
        notes_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if notes.size(1) == 0:
            return hidden_states
        summary = notes.mean(dim=1, keepdim=True)
        return hidden_states + summary


class FakeSpeculationHead(nn.Module):
    def forward(self, adapted: torch.Tensor) -> torch.Tensor:
        return adapted.clone()


class FakeNotesHead(nn.Module):
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return history.clone()


class FakeAgreementHead(nn.Module):
    def __init__(self, scores: Iterable[float]) -> None:
        super().__init__()
        self.register_buffer("_scores", torch.tensor(list(scores), dtype=torch.float32))
        self._index = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._scores.numel() == 0:
            value = 1.0
        else:
            value = float(self._scores[self._index % self._scores.numel()].item())
        self._index += 1
        return hidden_states.new_full((hidden_states.size(0), 1, 1), value)


class FakePlannerHead(nn.Module):
    def __init__(self, hidden_size: int, plan_vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, plan_vocab_size)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


class FakePlanEmbedding(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, plan_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = plan_ids.shape
        return torch.zeros(
            (batch, seq_len, self.hidden_size), dtype=torch.float32, device=plan_ids.device
        )


class FakeCoverageHead(nn.Module):
    def forward(
        self,
        attended: torch.Tensor,
        plan_embeddings: torch.Tensor,
        plan_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len = plan_embeddings.size(0), plan_embeddings.size(1)
        return attended.new_zeros(batch, seq_len)


class FakeModel(nn.Module):
    def __init__(self, streams: Tuple[str, ...]) -> None:
        super().__init__()
        hidden_size = 4
        vocab_size = 32
        self.streams = streams
        self.config = types.SimpleNamespace(
            notes_dim=hidden_size,
            notes_head=types.SimpleNamespace(notes_dim=hidden_size),
            notes_bus=types.SimpleNamespace(dtype="float32"),
        )
        self.trunk_adapter = FakeTrunkAdapter(hidden_size, vocab_size)
        self.stream_adapters = FakeStreamAdapters(streams, hidden_size)
        self.cross_attention = FakeCrossAttention()
        self.speculation_head = FakeSpeculationHead()
        self.notes_head = FakeNotesHead()
        self.agreement_head = FakeAgreementHead([0.0, 0.8, 0.8, 0.8, 0.8])
        self.planner_head = FakePlannerHead(hidden_size, vocab_size)
        self.plan_embedding = FakePlanEmbedding(hidden_size=hidden_size)
        self.coverage_head = FakeCoverageHead()


def _sample_plan_contract() -> Dict[str, Any]:
    return {
        "sectional_independence": True,
        "note_cadence_M": 4,
        "expected_dnb_lag_delta": 0,
        "streams": [
            {
                "stream_id": "stream_intro",
                "header": "Intro",
                "summary": "Summarize the topic setup.",
                "entities": ["ENT(id=intro_ent,name=Topic Intro,type=section)"],
                "constraints": ["FACT(subj_id=intro_ent,predicate=explains,object='setup')"],
                "section_contract": {"type": "alphabet_range", "start": "A", "end": "H"},
                "notes_contract": [
                    "COVERAGE(plan_item_id=intro_plan,status=covered)",
                    "FACT(subj_id=intro_ent,predicate=answers,object='setup answer')",
                ],
            },
            {
                "stream_id": "stream_core",
                "header": "Core",
                "summary": "Deliver main evidence.",
                "entities": ["ENT(id=core_ent,name=Main Evidence,type=section)"],
                "constraints": [
                    "FACT(subj_id=core_ent,predicate=describes,object='evidence block')"
                ],
                "section_contract": {"type": "alphabet_range", "start": "I", "end": "P"},
                "notes_contract": [
                    "COVERAGE(plan_item_id=core_plan,status=covered)",
                    "FACT(subj_id=core_ent,predicate=answers,object='core answer')",
                ],
            },
            {
                "stream_id": "stream_wrap",
                "header": "Wrap",
                "summary": "Conclude the argument.",
                "entities": ["ENT(id=wrap_ent,name=Wrap,type=section)"],
                "constraints": ["FACT(subj_id=wrap_ent,predicate=states,object='conclusion')"],
                "section_contract": {"type": "alphabet_range", "start": "Q", "end": "Z"},
                "notes_contract": [
                    "COVERAGE(plan_item_id=wrap_plan,status=covered)",
                    "FACT(subj_id=wrap_ent,predicate=answers,object='final answer')",
                ],
            },
        ],
    }


def _assert_stride_pattern(events, stream: str, stride: int) -> None:
    stream_events = [item for item in events if item.stream == stream]
    stride_indices = [event.stride_index for event in stream_events]
    assert stride_indices == sorted(stride_indices)
    counts: Dict[int, int] = {}
    for event in stream_events:
        counts[event.stride_index] = counts.get(event.stride_index, 0) + 1
        assert counts[event.stride_index] <= stride


def test_orchestrator_emits_notes_and_rolls_back_on_low_agreement() -> None:
    torch.manual_seed(0)
    streams = ("intro", "core", "wrap")
    config = InferenceConfig(
        streams=streams,
        stride_B=2,
        commit_L=4,
        read_lag_delta=0,
        max_snapshots_K=4,
        gate_g=1.0,
        agreement_threshold_tau=0.5,
        emission_cadence_M_by_stream={stream: 1 for stream in streams},
        decode=DecodeConfig(
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
            max_new_tokens=3,
            do_sample=False,
            seed=123,
        ),
        gate_annealing=GateAnnealingConfig(enabled=False),
        rng_seed=123,
    )

    tokenizer = FakeTokenizer()
    model = FakeModel(streams)
    orchestrator = MultiStreamOrchestrator(model, tokenizer, config)
    orchestrator.start("test prompt")

    events = []
    while True:
        outcome = orchestrator.step()
        if outcome is None:
            break
        events.append(outcome)

    manifest = orchestrator.finalize()

    assert len(events) == manifest["steps"]
    assert any(event.rollback_performed for event in events)
    assert manifest["rollbacks"], "Expected rollback events recorded in manifest."

    for stream in streams:
        _assert_stride_pattern(events, stream, config.stride_B)

    bus_versions: Dict[str, int] = {
        stream: orchestrator.bus_by_stream[stream].latest_version() for stream in streams
    }
    assert bus_versions["intro"] >= 3

    for stream, state in orchestrator.states.items():
        for producer, version in state.last_seen_version.items():
            assert version <= bus_versions[producer]

    rollback_streams = {record["stream"] for record in manifest["rollbacks"]}
    assert "intro" in rollback_streams


def test_plan_contract_seeds_initial_snapshot() -> None:
    torch.manual_seed(0)
    streams = ("intro", "core", "wrap")
    config = InferenceConfig(
        streams=streams,
        stride_B=2,
        commit_L=4,
        read_lag_delta=0,
        max_snapshots_K=4,
        gate_g=1.0,
        agreement_threshold_tau=0.5,
        emission_cadence_M_by_stream={stream: 1 for stream in streams},
        decode=DecodeConfig(
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
            max_new_tokens=2,
            do_sample=False,
            seed=123,
        ),
        gate_annealing=GateAnnealingConfig(enabled=False),
        rng_seed=123,
    )
    tokenizer = FakeTokenizer()
    model = FakeModel(streams)
    orchestrator = MultiStreamOrchestrator(model, tokenizer, config)
    plan_payload = _sample_plan_contract()
    expected_vectors = orchestrator._plan_seed_vectors(plan_payload, "plan prompt")
    assert expected_vectors is not None
    orchestrator.start("plan prompt", plan_contract=plan_payload)
    for stream in streams:
        snapshots = orchestrator.bus_by_stream[stream].snapshot(lag=0, limit=4)
        assert snapshots, f"expected plan snapshot for {stream}"
        expected = expected_vectors[stream].to(
            device=snapshots[0].notes.device, dtype=snapshots[0].notes.dtype
        )
        assert torch.allclose(
            snapshots[0].notes.view(-1),
            expected.view(-1),
            atol=1e-5,
        ), f"plan snapshot mismatch for {stream}"
