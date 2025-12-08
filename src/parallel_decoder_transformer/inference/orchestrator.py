"""Inference orchestrator wiring Dynamic Notes Bus and SNC for multi-stream decoding."""

from __future__ import annotations

import logging
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn

try:  # pragma: no cover - optional dependency for memory reporting
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore

try:  # pragma: no cover - optional dependency for typing only
    from transformers.tokenization_utils import PreTrainedTokenizerBase
except ImportError:  # pragma: no cover - allows running without transformers during tests
    PreTrainedTokenizerBase = object  # type: ignore[misc,assignment]

from .config import DecodeConfig, InferenceConfig
from .dnb_bus import DynamicNotesBus, DynamicNotesBusConfig
from .scheduler import TriangularScheduler
from .state import PastKeyValues, StreamState
from .window import NotesWindow, NotesWindowBuilder, TopologyMask
from .replay import ReplayArtifactWriter
from ..datasets.plan_contract_notes import derive_initial_notes_from_plan
from ..data.extraction import NOTES_SCHEMA_VERSION
from ..data.teacher_provider import _HashingEmbedder, _stringify_stream_notes
from ..integration.instrumentation import InstrumentedTrunkAdapter
from ..utils.plan_catalog import (
    PlanHashParams,
    hash_plan_text,
    normalise_plan_map,
    resolve_plan_hash_params,
)

LOGGER = logging.getLogger("parallel decoder transformer.orchestrator")


class _OrchestratorNotesProvider:
    """Bridges orchestrator state to the instrumented trunk adapter."""

    def __init__(self, orchestrator: "MultiStreamOrchestrator", notes_dim: int) -> None:
        self._orchestrator = orchestrator
        self._empty_notes = torch.zeros(
            1,
            0,
            notes_dim,
            device=orchestrator.device,
            dtype=orchestrator.dtype,
        )

    def notes_for(
        self, stream: Union[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(stream, torch.Tensor):
            indices = stream.tolist()
            if not indices:
                return self._empty_notes, None
            stream_name = self._orchestrator.config.streams[indices[0]]
        else:
            stream_name = stream
        state = self._orchestrator.states.get(stream_name, None)
        if state is None or state.current_notes is None:
            return self._empty_notes, None
        mask = state.current_notes_mask
        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=state.current_notes.device)
        return state.current_notes, mask


@dataclass(frozen=True, slots=True)
class AgreementResult:
    """Outcome of evaluating an agreement gate."""

    score: float
    triggered: bool


class AgreementGate:
    """Applies an agreement threshold on attended hidden states."""

    def __init__(self, threshold: float) -> None:
        if threshold <= 0.0 or threshold >= 1.0:
            raise ValueError("AgreementGate threshold must lie inside (0, 1).")
        self.threshold = threshold

    def evaluate(self, agreement_tensor: torch.Tensor) -> AgreementResult:
        if agreement_tensor.numel() == 0:
            raise ValueError("Agreement head returned an empty tensor.")
        score = float(agreement_tensor.detach().mean().item())
        return AgreementResult(score=score, triggered=score < self.threshold)


@dataclass(slots=True)
class StepOutcome:
    """Telemetry describing a single decode step."""

    stream: str
    token_id: int
    token_text: str
    stride_index: int
    stride_completed: bool
    stream_completed: bool
    agreement: float
    notes_emitted: bool
    rollback_performed: bool
    cadence_mode: Optional[str] = None
    cadence_probability: Optional[float] = None
    cadence_multiplier: Optional[float] = None
    cadence_forced: bool = False
    coverage_logits: Optional[List[float]] = None
    top2_margin: Optional[float] = None
    counterfactuals: Optional[List[str]] = None


class MultiStreamOrchestrator:
    """Drives synchronous multi-stream decoding with Dynamic Notes Bus + SNC."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        config: InferenceConfig,
        *,
        topology_mask: Optional[TopologyMask] = None,
        decode_config: Optional[DecodeConfig] = None,
        logit_blend_alpha: Optional[float] = None,
        log_margins: bool = False,
        sync_profile: bool = False,
        replay_writer: Optional[ReplayArtifactWriter] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.decode_config = decode_config or config.decode
        self._validate_decode_config(self.decode_config)
        alpha_source = config.logit_blend_alpha if logit_blend_alpha is None else logit_blend_alpha
        self.alpha = float(max(0.0, min(1.0, alpha_source)))
        self._log_margins = bool(log_margins)
        self._sync_profile = bool(sync_profile)
        self._replay_writer = replay_writer
        self._replay_plan_recorded = False

        self._plan_hash_params: PlanHashParams = resolve_plan_hash_params(
            getattr(model, "config", model)
        )
        self._plan_vocab_size = int(self._plan_hash_params.vocab_size)
        self._plan_hash_buckets = int(self._plan_hash_params.hash_buckets)
        self._plan_hash_salt = self._plan_hash_params.salt

        self.device = self._resolve_device()
        self.dtype = self._resolve_dtype()

        notes_dim = self._resolve_notes_dim()
        bus_dtype = self._resolve_bus_dtype()

        self.window_builder = NotesWindowBuilder.from_config(
            config,
            notes_dim,
            topology_mask=topology_mask,
            device=self.device,
            dtype=self.dtype,
        )
        self.scheduler = TriangularScheduler(
            config.streams,
            stride=config.stride_B,
        )
        self.agreement_gate = AgreementGate(config.agreement_threshold_tau)

        self.bus_by_stream: Dict[str, DynamicNotesBus] = {
            stream: DynamicNotesBus(
                DynamicNotesBusConfig(
                    snapshot_dim=notes_dim,
                    max_snapshots=config.max_snapshots_K,
                    lag=config.read_lag_delta,
                    dtype=bus_dtype,
                    device=str(self.device),
                )
            )
            for stream in config.streams
        }

        self.states: Dict[str, StreamState] = {}
        self._base_hidden: Dict[str, torch.Tensor] = {}
        self._attended_history: Dict[str, List[torch.Tensor]] = {}
        self._rng = self._build_generator(config.rng_seed)

        self._active = False
        self._step_count = 0
        self._completed_streams: set[str] = set()
        self._rollback_events: List[Dict[str, Any]] = []
        self._timings: Dict[str, Any] = {}
        self._start_time: Optional[float] = None
        self._last_stride_start: Optional[float] = None
        self._plan_token_ids: Optional[torch.Tensor] = None
        self._plan_mask: Optional[torch.Tensor] = None
        self._plan_logits: Optional[torch.Tensor] = None
        self._plan_source: str = "none"
        self._gate_values: Dict[str, float] = {stream: config.gate_g for stream in config.streams}
        self._gate_cooldown: Dict[str, int] = {stream: 0 for stream in config.streams}
        self._cadence_events: List[Dict[str, Any]] = []
        self._coverage_history: Dict[str, List[List[float]]] = {
            stream: [] for stream in config.streams
        }
        self._coverage_manifest: Dict[str, List[Dict[str, Any]]] = {
            stream: [] for stream in config.streams
        }
        self._plan_embeddings: Optional[torch.Tensor] = None
        self._plan_mask_bool: Optional[torch.Tensor] = None
        self._plan_ids_list: Optional[List[int]] = None
        self._plan_mask_list: Optional[List[int]] = None
        self._plan_catalog_entries: Optional[List[Dict[str, Any]]] = None
        self._plan_catalog_index: Dict[int, Dict[str, Any]] = {}
        self._counterfactual_generator = torch.Generator(device="cpu")
        if self.config.rng_seed is not None:
            self._counterfactual_generator.manual_seed(int(self.config.rng_seed))
        else:
            self._counterfactual_generator.manual_seed(int(time.time()))
        self._frozen_windows: Dict[str, NotesWindow] = {}
        self._memory_trace: List[Dict[str, float]] = []
        self._notes_provider: Optional[_OrchestratorNotesProvider] = None
        self._configure_instrumentation(notes_dim)
        self._run_instrumentation_health_check(notes_dim)
        self._instrumented_layers: Tuple[int, ...] = self._resolve_instrumented_layers()
        self._step_timings: List[Dict[str, Any]] = []
        self._gate_trace: List[Dict[str, Any]] = []
        self._sync_overhead_total: float = 0.0
        flicker_window = max(2, int(self.config.safeguards.flicker.window))
        self._gate_histories: Dict[str, deque[float]] = {
            stream: deque(maxlen=flicker_window) for stream in config.streams
        }
        self._plan_histories: Dict[str, deque[Optional[int]]] = {
            stream: deque(maxlen=flicker_window) for stream in config.streams
        }
        self._lipschitz_events: List[Dict[str, Any]] = []
        self._flicker_events: List[Dict[str, Any]] = []
        self._flicker_counters: Dict[str, int] = {"gate": 0, "plan": 0}
        self._lipschitz_probe_budget: int = 0

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def start(
        self,
        prompt: str,
        planner_notes: Optional[Any] = None,
        *,
        prefix_by_stream: Optional[Dict[str, str]] = None,
        seed_text_by_stream: Optional[Dict[str, str]] = None,
        seed_notes_by_stream: Optional[Dict[str, Sequence[float]]] = None,
        plan_text_by_stream: Optional[Mapping[str, Sequence[str]]] = None,
        plan_contract: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Initialise decoding state from a prompt."""

        if not prompt:
            raise ValueError("Prompt must be a non-empty string.")
        self._reset_runtime_state()
        self._start_time = time.time()
        self._last_stride_start = self._start_time
        if hasattr(self.model, "eval"):
            self.model.eval()
        self._plan_catalog_entries = None
        self._plan_catalog_index = {}
        derived_planner_notes = planner_notes
        if plan_text_by_stream:
            derived = self._planner_notes_from_text(plan_text_by_stream)
            if derived is not None:
                derived_planner_notes = derived

        # Prepare a default tokenization once when no per-stream prefix is provided.
        default_encoded = None
        if not prefix_by_stream:
            default_encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

        # Use no_grad to avoid autograd tracking while keeping tensors mutable for dowparallel decoder transformer heads.
        plan_seed_vectors: Optional[Dict[str, torch.Tensor]] = None
        if plan_contract is not None:
            plan_seed_vectors = self._plan_seed_vectors(plan_contract, prompt)

        with torch.no_grad():
            for index, stream in enumerate(self.config.streams):
                if prefix_by_stream and stream in prefix_by_stream:
                    stream_prompt = f"{prefix_by_stream[stream]}{prompt}"
                    encoded = self.tokenizer(
                        stream_prompt, return_tensors="pt", add_special_tokens=True
                    )
                else:
                    encoded = default_encoded or self.tokenizer(
                        prompt, return_tensors="pt", add_special_tokens=True
                    )
                input_ids = encoded["input_ids"].to(self.device).clone()
                attention_mask = encoded.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, device=self.device)
                else:
                    attention_mask = attention_mask.to(self.device).clone()

                state = StreamState(
                    stream=stream,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    commit_stride=self.config.stride_B,
                    commit_horizon=self.config.commit_L,
                )
                self._emit_model_hook("on_bootstrap_stream", stream=stream, index=index)
                if plan_seed_vectors and stream in plan_seed_vectors:
                    plan_vector = plan_seed_vectors[stream].clone()
                    snapshot = self.bus_by_stream[stream].push(plan_vector, stride=0)
                    state.mark_snapshot_version(snapshot.version)
                    state.reset_snapshot_counter()
                outputs = self._run_trunk(
                    stream=stream,
                    notes=None,
                    notes_mask=None,
                    input_ids=state.input_ids,
                    attention_mask=state.attention_mask,
                    past_key_values=None,
                )
                if index == 0:
                    if derived_planner_notes is not None:
                        payload = self._normalise_planner_payload(
                            derived_planner_notes, attention_mask
                        )
                        self._plan_token_ids = payload["plan_token_ids"]
                        self._plan_mask = self._derive_plan_mask(payload["plan_mask"])
                        self._plan_logits = payload.get("plan_logits")
                        self._plan_source = payload.get("source", "external")
                    else:
                        planner_logits = self.model.planner_head(outputs.hidden_states[-1])
                        self._plan_logits = planner_logits.detach()
                        self._plan_token_ids = torch.argmax(planner_logits, dim=-1).long()
                        self._plan_mask = self._derive_plan_mask(attention_mask.clone())
                        self._plan_source = "model"
                state.past_key_values = outputs.past_key_values
                base_hidden = outputs.hidden_states[-1][:, -1:, :].to(device=self.device)
                self._base_hidden[stream] = base_hidden
                adapted = self._apply_stream_adapter(stream, base_hidden)
                speculative = self._bootstrap_speculation_notes(adapted)
                self._record_bootstrap_for_replay(stream, speculative)
                snapshot = self.bus_by_stream[stream].push(speculative.detach(), stride=0)
                state.mark_snapshot_version(snapshot.version)
                self.states[stream] = state
                self._attended_history[stream] = []

        self._inject_initial_seeds(
            seed_text_by_stream or {},
            seed_notes_by_stream or {},
        )

        for stream, state in self.states.items():
            window = self.window_builder.build(state, self.bus_by_stream)
            state.update_notes_window(window.notes, window.mask)
            self._mark_versions_consumed(state, window.producers, window.versions)

        if self._plan_token_ids is not None and self._plan_mask is not None:
            plan_ids = self._plan_token_ids.to(device=self.device, dtype=torch.long).clone()
            self._plan_embeddings = self.model.plan_embedding(plan_ids).detach()
            self._plan_mask_bool = self._plan_mask.to(dtype=torch.bool, device=self.device)
            self._plan_ids_list = [int(value) for value in plan_ids.view(-1).tolist()]
            self._plan_mask_list = [int(value) for value in self._plan_mask.view(-1).tolist()]
            if self._plan_catalog_entries is None:
                self._set_plan_catalog_entries(self._build_catalog_from_plan_ids())
        else:
            self._plan_embeddings = None
            self._plan_mask_bool = None
            self._plan_ids_list = None
            self._plan_mask_list = None
            self._plan_catalog_entries = None
            self._plan_catalog_index = {}

        self._record_plan_for_replay()
        self._active = True
        self._timings["bootstrap"] = time.time() - self._start_time
        self._step_count = 0

    def step(self) -> Optional[StepOutcome]:
        """Advance the orchestrator by one token emission."""

        if not self._active:
            return None
        tick_start = time.time()

        tick = self.scheduler.tick()
        stream = tick.stream
        state = self.states[stream]
        gate_after = float(self._gate_values.get(stream, self.config.gate_g))
        self._emit_model_hook(
            "on_step_begin",
            stream=stream,
            stride_index=tick.stride_index,
            token_index=tick.token_index,
            step=self._step_count + 1,
        )

        if self._stream_completed(state):
            self._completed_streams.add(stream)
            outcome = self.scheduler.advance()
            if outcome.stride_completed:
                self._on_stride_complete()
            if len(self._completed_streams) == len(self.states):
                self._active = False
            return None

        window = self.window_builder.build(state, self.bus_by_stream)
        window, counterfactual_tags = self._apply_counterfactuals(stream, window)
        state.update_notes_window(window.notes, window.mask)
        self._mark_versions_consumed(state, window.producers, window.versions)

        base_hidden = self._base_hidden[stream]
        adapted = self._apply_stream_adapter(stream, base_hidden)
        attended = self._apply_cross_attention(
            stream, base_hidden, adapted, window.notes, window.mask
        )
        self._maybe_probe_lipschitz(
            stream, base_hidden, adapted, window.notes, window.mask, attended
        )

        attended_logits = self._lm_head(attended)
        base_logits = self._lm_head(base_hidden) if self.alpha < 0.999 else attended_logits
        logits = self._blend_logits(attended_logits, base_logits)

        agreement_tensor = self.model.agreement_head(attended)
        agreement = float(agreement_tensor.detach().squeeze().item())

        token_id = self._sample_token(logits, state)
        token_text = self._decode_token(token_id)
        top2_margin = self._compute_top2_margin(attended_logits) if self._log_margins else None

        prev_kv = state.past_key_values
        state.append_token(token_id, past_key_values=None, token_text=token_text)

        history_bucket = self._attended_history.setdefault(stream, [])
        history_bucket.append(attended.detach())
        self._track_coverage(stream, attended)
        coverage_list = self._coverage_current(stream)
        cadence = self.config.cadence_for(stream)
        notes_emitted = False
        rollback_performed = False
        note_summary_tensor: Optional[torch.Tensor] = None
        emit, cadence_meta = self._should_emit_notes(stream, state, cadence, agreement)
        if cadence_meta is not None:
            cadence_meta.update(
                {
                    "stream": stream,
                    "stride_index": tick.stride_index,
                    "token_index": tick.token_index,
                }
            )
            self._cadence_events.append(cadence_meta)
        if emit:
            notes_emitted = True
            history = self._stack_attended_history(stream)
            note_summary = self.model.notes_head(history).mean(dim=1, keepdim=True)
            note_summary_tensor = note_summary.detach()
            stride = max(1, state.tokens_since_snapshot)
            snapshot = self.bus_by_stream[stream].push(note_summary.detach(), stride=stride)
            state.mark_snapshot_version(snapshot.version)
            state.reset_snapshot_counter()
            self._attended_history[stream] = []

            result = self.agreement_gate.evaluate(agreement_tensor)
            agreement = result.score
            if result.triggered:
                rollback_performed = self._perform_rollback(stream, state)
            coverage_list = self._finalise_coverage(stream, tick.stride_index, tick.token_index)
            self._update_gate_on_emission(stream, result, agreement)
        else:
            self._update_gate_on_stable_step(stream, agreement)
            coverage_list = self._coverage_current(stream)
        gate_after = float(self._gate_values.get(stream, self.config.gate_g))

        post_window = self.window_builder.build(state, self.bus_by_stream)
        post_window, _ = self._apply_counterfactuals(stream, post_window)
        state.update_notes_window(post_window.notes, post_window.mask)
        self._mark_versions_consumed(state, post_window.producers, post_window.versions)
        next_notes = state.current_notes
        next_mask = state.current_notes_mask

        if not rollback_performed:
            outputs = self._run_trunk(
                stream=stream,
                notes=next_notes,
                notes_mask=next_mask,
                input_ids=state.input_ids[:, -1:],
                attention_mask=state.attention_mask,
                past_key_values=prev_kv,
            )
            state.past_key_values = outputs.past_key_values
            self._base_hidden[stream] = outputs.hidden_states[-1][:, -1:, :].to(device=self.device)
        state.register_commit()

        stream_finished = self._stream_completed(state)
        if stream_finished:
            self._completed_streams.add(stream)
        else:
            self._completed_streams.discard(stream)

        outcome = self.scheduler.advance()
        if outcome.stride_completed:
            self._on_stride_complete()

        if len(self._completed_streams) == len(self.states):
            self._active = False

        self._step_count += 1
        duration = float(time.time() - tick_start)
        step_record = {
            "step": self._step_count,
            "stream": stream,
            "stride_index": tick.stride_index,
            "token_index": tick.token_index,
            "duration_s": duration,
        }
        if top2_margin is not None:
            step_record["top2_margin"] = top2_margin
        self._step_timings.append(step_record)
        self._gate_trace.append(
            {
                "step": self._step_count,
                "stream": stream,
                "value": gate_after,
            }
        )
        coverage_payload = list(coverage_list) if coverage_list is not None else None
        self._track_flicker(stream, gate_after, coverage_payload)
        self._emit_model_hook(
            "on_step_end",
            stream=stream,
            token_id=token_id,
            step=self._step_count,
        )
        memory_snapshot = self._memory_snapshot()
        if memory_snapshot is not None:
            memory_snapshot.update({"step": self._step_count, "stream": stream})
            self._memory_trace.append(memory_snapshot)

        coverage_for_replay = coverage_payload
        self._record_step_for_replay(
            stream=stream,
            token_id=token_id,
            agreement=agreement,
            coverage_logits=coverage_for_replay,
            note_vector=note_summary_tensor,
            emitted=notes_emitted,
            attended=attended,
            base_hidden=base_hidden,
        )

        return StepOutcome(
            stream=stream,
            token_id=token_id,
            token_text=token_text,
            stride_index=tick.stride_index,
            stride_completed=outcome.stride_completed,
            stream_completed=outcome.stream_completed,
            agreement=agreement,
            notes_emitted=notes_emitted,
            rollback_performed=rollback_performed,
            cadence_mode=cadence_meta["mode"] if cadence_meta is not None else None,
            cadence_probability=cadence_meta.get("final_probability") if cadence_meta else None,
            cadence_multiplier=cadence_meta.get("multiplier") if cadence_meta else None,
            cadence_forced=bool(cadence_meta.get("forced")) if cadence_meta else False,
            coverage_logits=coverage_payload,
            top2_margin=top2_margin,
            counterfactuals=counterfactual_tags,
        )

    def stream(self) -> Dict[str, Dict[str, Any]]:
        """Return current per-stream text and token statistics."""

        payload: Dict[str, Dict[str, Any]] = {}
        for stream, state in self.states.items():
            payload[stream] = {
                "text": state.generated_text,
                "token_count": state.generated_count,
                "latest_version": state.latest_snapshot_version,
                "gate": self._gate_values.get(stream, self.config.gate_g),
                "cadence_mode": self.config.cadence_policy.mode,
                "coverage": self._coverage_snapshot(stream, None),
            }
        return payload

    def finalize(self) -> Dict[str, Any]:
        """Return a manifest summarising the inference run."""

        end_time = time.time()
        total_duration = float(end_time - (self._start_time or end_time))
        self._timings["total"] = total_duration
        manifest = {
            "schema": {
                "notes": NOTES_SCHEMA_VERSION,
            },
            "timings": dict(self._timings),
            "instrumented_layers": list(self._instrumented_layers),
            "config": {
                "stride_B": self.config.stride_B,
                "commit_L": self.config.commit_L,
                "read_lag_delta": self.config.read_lag_delta,
                "max_snapshots_K": self.config.max_snapshots_K,
                "topology": self.config.topology,
                "gate_g": self.config.gate_g,
                "tau": self.config.agreement_threshold_tau,
                "M_by_stream": dict(self.config.emission_cadence_M_by_stream),
                "alpha": self.alpha,
                "gate_annealing": {
                    "enabled": self.config.gate_annealing.enabled,
                    "decay": self.config.gate_annealing.decay,
                    "min_value": self.config.gate_annealing.min_value,
                    "recovery": self.config.gate_annealing.recovery,
                    "stability_margin": self.config.gate_annealing.stability_margin,
                    "cooldown": self.config.gate_annealing.cooldown,
                },
                "cadence_policy": {
                    "mode": self.config.cadence_policy.mode,
                    "min_probability": self.config.cadence_policy.min_probability,
                    "max_interval": self.config.cadence_policy.max_interval,
                    "multiplier_min": self.config.cadence_policy.multiplier_min,
                    "multiplier_max": self.config.cadence_policy.multiplier_max,
                    "agreement_low": self.config.cadence_policy.agreement_low,
                    "agreement_high": self.config.cadence_policy.agreement_high,
                    "age_boost": self.config.cadence_policy.age_boost,
                },
                "rng_seed": self.config.rng_seed,
            },
            "streams": {
                stream: {
                    "text": state.generated_text,
                    "token_texts": list(state.generated_pieces),
                    "token_ids": list(state.generated_tokens),
                    "latest_version": state.latest_snapshot_version,
                    "rollback_buffer": list(state.rollback_buffer),
                    "gate": self._gate_values.get(stream, self.config.gate_g),
                    "coverage": self._coverage_snapshot(stream, None),
                }
                for stream, state in self.states.items()
            },
            "rollbacks": self._rollback_events,
            "steps": self._step_count,
        }
        manifest.update(self._plan_hash_params.as_dict())
        if self._step_timings:
            manifest["timings"]["per_token"] = list(self._step_timings)
        if getattr(self, "_sync_profile", False) and self._timings.get("stride_sync_durations"):
            manifest["timings"]["sync_overhead_s"] = self._sync_overhead_total
        if self._plan_token_ids is not None and self._plan_mask is not None:
            manifest["plan"] = {
                "source": self._plan_source,
                "token_ids": self._tensor_to_int_list(self._plan_token_ids),
                "mask": self._tensor_to_int_list(self._plan_mask),
            }
            if self._plan_logits is not None:
                manifest["plan"]["logits_shape"] = list(self._plan_logits.shape)
            if self._plan_catalog_entries:
                manifest["plan"]["catalog"] = [dict(entry) for entry in self._plan_catalog_entries]
        if self._cadence_events:
            manifest["cadence_events"] = list(self._cadence_events)
        if self._gate_trace:
            manifest["gate_trace"] = list(self._gate_trace)
        if getattr(self.config.counterfactuals, "enabled", False):
            manifest.setdefault("counterfactuals", self.config.counterfactuals.as_dict())
        memory_trace = getattr(self, "_memory_trace", None)
        if memory_trace:
            max_alloc = max(
                entry.get("allocated_bytes", entry.get("rss_bytes", 0.0)) for entry in memory_trace
            )
            manifest["memory"] = {
                "entries": list(memory_trace),
                "max_tracked_bytes": float(max_alloc),
                "max_estimate_bytes": float(
                    max(entry.get("estimate_bytes", 0.0) for entry in memory_trace)
                ),
            }
        safeguards_manifest = self._build_safeguards_manifest()
        if safeguards_manifest:
            manifest["safeguards"] = safeguards_manifest
        return manifest

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _emit_model_hook(self, hook_name: str, **kwargs: Any) -> None:
        handler = getattr(self.model, hook_name, None)
        if not callable(handler):
            return
        try:
            handler(**kwargs)
        except Exception:  # pragma: no cover - hook safety
            LOGGER.debug("model_hook_failed | hook=%s", hook_name, exc_info=True)

    def _inject_initial_seeds(
        self,
        seed_text_by_stream: Dict[str, str],
        seed_notes_by_stream: Dict[str, Sequence[float]],
    ) -> None:
        if not seed_text_by_stream and not seed_notes_by_stream:
            return
        notes_dim = self._resolve_notes_dim()
        with torch.no_grad():
            for stream, text in seed_text_by_stream.items():
                stream_norm = stream.lower()
                if stream_norm not in self.states:
                    raise ValueError(f"Seed text provided for unknown stream {stream!r}.")
                if not text.strip():
                    continue
                seed_tensor = self._seed_from_text(stream_norm, text)
                snapshot = self.bus_by_stream[stream_norm].push(seed_tensor, stride=0)
                self.states[stream_norm].mark_snapshot_version(snapshot.version)
                self.states[stream_norm].reset_snapshot_counter()
            for stream, vector in seed_notes_by_stream.items():
                stream_norm = stream.lower()
                if stream_norm not in self.states:
                    raise ValueError(f"Seed notes provided for unknown stream {stream!r}.")
                tensor = torch.as_tensor(vector, device=self.device, dtype=self.dtype)
                if tensor.numel() != notes_dim:
                    raise ValueError(
                        f"Seed notes for stream {stream!r} must have length {notes_dim}, received {tensor.numel()}."
                    )
                seed_tensor = tensor.view(1, 1, notes_dim).to(dtype=self.dtype)
                snapshot = self.bus_by_stream[stream_norm].push(seed_tensor, stride=0)
                self.states[stream_norm].mark_snapshot_version(snapshot.version)
                self.states[stream_norm].reset_snapshot_counter()

    def _plan_seed_vectors(
        self,
        plan_payload: Mapping[str, Any],
        prompt: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        try:
            plan_notes = derive_initial_notes_from_plan(plan_payload, input_text=prompt)
        except Exception:
            LOGGER.warning("plan_seed_failed | reason=derive_error", exc_info=True)
            return None
        if not plan_notes:
            LOGGER.warning("plan_seed_failed | reason=empty_plan_notes")
            return None
        notes_dim = self._resolve_notes_dim()
        embedder = _HashingEmbedder()
        serialized: Dict[str, torch.Tensor] = {}
        for note in plan_notes:
            try:
                strings = _stringify_stream_notes(note.as_dict())
            except Exception:
                strings = []
            vector = embedder.aggregate(strings, notes_dim)
            serialized[note.stream_id.lower()] = vector.to(device=self.device, dtype=self.dtype)
        assigned: Dict[str, torch.Tensor] = {}
        for idx, stream in enumerate(self.config.streams, start=1):
            key = stream
            vector = serialized.get(key)
            if vector is None:
                alias = f"stream_{idx}"
                vector = serialized.get(alias)
            if vector is None:
                named_alias = f"stream_{stream}"
                vector = serialized.get(named_alias)
            if vector is None and idx - 1 < len(plan_notes):
                fallback_key = plan_notes[idx - 1].stream_id.lower()
                vector = serialized.get(fallback_key)
            if vector is None:
                continue
            assigned[stream] = vector.clone().unsqueeze(0).unsqueeze(0)
        if not assigned:
            LOGGER.warning("plan_seed_failed | reason=no_matching_streams")
            return None
        return assigned

    def _apply_counterfactuals(
        self,
        stream: str,
        window: NotesWindow,
    ) -> Tuple[NotesWindow, Optional[List[str]]]:
        config = getattr(self.config, "counterfactuals", None)
        if config is None or not config.enabled or window.notes.size(1) == 0:
            return window, None
        interventions: List[str] = []
        notes = window.notes
        mask = window.mask
        versions = window.versions
        strides = window.strides
        producers = list(window.producers)
        mutated = False

        def ensure_mutable() -> None:
            nonlocal notes, mask, versions, strides, producers, mutated
            if mutated:
                return
            notes = notes.clone()
            mask = mask.clone()
            versions = versions.clone()
            strides = strides.clone()
            producers = list(producers)
            mutated = True

        if config.should_freeze(stream):
            cached = self._frozen_windows.get(stream)
            if cached is None:
                cached = self._clone_window(window)
                self._frozen_windows[stream] = cached
            notes = cached.notes.clone()
            mask = cached.mask.clone()
            versions = cached.versions.clone()
            strides = cached.strides.clone()
            producers = list(cached.producers)
            mutated = True
            interventions.append("freeze")
        else:
            self._frozen_windows.pop(stream, None)

        if config.should_ablate(stream) and notes.numel() > 0:
            ensure_mutable()
            notes.zero_()
            mask.zero_()
            interventions.append("ablate")

        extra = config.stale_extra_for(stream)
        if extra > 0 and notes.size(1) > 0:
            ensure_mutable()
            keep = max(1, notes.size(1) - extra)
            notes = notes[:, :keep, :]
            mask = mask[:, :keep]
            versions = versions[:keep]
            strides = strides[:keep]
            producers = producers[:keep]
            interventions.append(f"stale+{extra}")

        swap_lookup = config.swap_map()
        if swap_lookup and producers:
            handled: set[Tuple[str, str]] = set()
            for idx, producer in enumerate(producers):
                partner = swap_lookup.get(producer)
                if not partner:
                    continue
                try:
                    partner_idx = producers.index(partner)
                except ValueError:
                    continue
                key = tuple(sorted((producer, partner)))
                if key in handled:
                    continue
                ensure_mutable()
                notes[:, [idx, partner_idx], :] = notes[:, [partner_idx, idx], :]
                mask[:, [idx, partner_idx]] = mask[:, [partner_idx, idx]]
                versions[[idx, partner_idx]] = versions[[partner_idx, idx]]
                strides[[idx, partner_idx]] = strides[[partner_idx, idx]]
                producers[idx], producers[partner_idx] = producers[partner_idx], producers[idx]
                interventions.append(f"swap:{producer}<->{partner}")
                handled.add(key)

        if config.should_shuffle(stream) and notes.size(1) > 1:
            ensure_mutable()
            order = torch.randperm(
                notes.size(1),
                generator=self._counterfactual_generator,
                device=notes.device,
            )
            notes = notes[:, order, :]
            mask = mask[:, order]
            order_cpu = order.to(device=versions.device)
            versions = versions[order_cpu]
            strides = strides[order_cpu]
            order_list = order_cpu.tolist()
            producers = [producers[i] for i in order_list]
            interventions.append("shuffle")

        if not mutated:
            return window, interventions or None

        updated = NotesWindow(
            notes=notes,
            mask=mask,
            producers=tuple(producers),
            versions=versions,
            strides=strides,
        )
        return updated, interventions or None

    def _clone_window(self, window: NotesWindow) -> NotesWindow:
        return NotesWindow(
            notes=window.notes.clone(),
            mask=window.mask.clone(),
            producers=tuple(window.producers),
            versions=window.versions.clone(),
            strides=window.strides.clone(),
        )

    def _memory_snapshot(self) -> Optional[Dict[str, float]]:
        if not getattr(self.config, "memory_report", False):
            return None
        snapshot: Dict[str, float] = {}
        if self.device.type == "cuda" and torch.cuda.is_available():
            snapshot["allocated_bytes"] = float(torch.cuda.memory_allocated(self.device))
            snapshot["reserved_bytes"] = float(torch.cuda.memory_reserved(self.device))
        elif psutil is not None:
            process = psutil.Process(os.getpid())
            snapshot["rss_bytes"] = float(process.memory_info().rss)
        else:  # pragma: no cover - psutil optional
            return None
        snapshot["estimate_bytes"] = float(self._estimate_parallel_memory_bytes())
        return snapshot

    def _record_plan_for_replay(self) -> None:
        if self._replay_writer is None or self._replay_plan_recorded:
            return
        if self._plan_token_ids is None or self._plan_mask is None:
            return
        catalog = self._plan_catalog_entries or []
        plan_logits = (
            self._plan_logits.detach().to("cpu") if self._plan_logits is not None else None
        )
        self._replay_writer.record_plan(
            plan_token_ids=self._plan_token_ids.detach().to("cpu"),
            plan_mask=self._plan_mask.detach().to("cpu"),
            plan_logits=plan_logits,
            source=self._plan_source,
            catalog=catalog,
        )
        self._replay_plan_recorded = True

    def _record_bootstrap_for_replay(self, stream: str, vector: torch.Tensor) -> None:
        if self._replay_writer is None:
            return
        self._replay_writer.record_bootstrap(stream, vector.detach().to("cpu"))

    def _record_step_for_replay(
        self,
        *,
        stream: str,
        token_id: int,
        agreement: float,
        coverage_logits: Optional[Sequence[float]],
        note_vector: Optional[torch.Tensor],
        emitted: bool,
        attended: torch.Tensor,
        base_hidden: torch.Tensor,
    ) -> None:
        if self._replay_writer is None:
            return
        coverage_payload = list(coverage_logits) if coverage_logits is not None else None
        vector = note_vector.detach().to("cpu") if note_vector is not None else None
        self._replay_writer.record_step(
            stream,
            token_id=token_id,
            agreement=agreement,
            coverage_logits=coverage_payload,
            note_emitted=emitted,
            note_vector=vector,
            delta_norm=self._attended_delta_norm(attended, base_hidden),
        )

    def _attended_delta_norm(self, attended: torch.Tensor, base_hidden: torch.Tensor) -> float:
        with torch.no_grad():
            delta = attended - base_hidden
            return float(torch.linalg.norm(delta).item())

    def _estimate_parallel_memory_bytes(self) -> float:
        streams = len(self.config.streams)
        stride = max(1, self.config.stride_B)
        horizon = max(1, self.config.commit_L)
        snapshots = max(1, self.config.max_snapshots_K)
        hidden_size = self._resolve_model_hidden_size()
        notes_dim = self._resolve_notes_dim()
        bytes_per_hidden = hidden_size * 2.0  # assume bf16 activations
        bytes_per_note = notes_dim * 2.0
        kv_budget = streams * horizon * bytes_per_hidden
        stride_budget = streams * stride * bytes_per_hidden
        snapshot_budget = streams * snapshots * bytes_per_note
        return kv_budget + stride_budget + snapshot_budget

    def _seed_from_text(self, stream: str, text: str) -> torch.Tensor:
        with torch.no_grad():
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=self.device)
            else:
                attention_mask = attention_mask.to(self.device)
            outputs = self._run_trunk(
                stream=stream,
                notes=None,
                notes_mask=None,
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
            )
            hidden = outputs.hidden_states[-1]
            pooled = hidden.mean(dim=1, keepdim=True)
            notes_dim = self._resolve_notes_dim()
            if pooled.size(-1) > notes_dim:
                pooled = pooled[..., :notes_dim]
            elif pooled.size(-1) < notes_dim:
                pad = torch.zeros(
                    pooled.size(0),
                    pooled.size(1),
                    notes_dim - pooled.size(-1),
                    device=pooled.device,
                    dtype=pooled.dtype,
                )
                pooled = torch.cat([pooled, pad], dim=-1)
            norm = torch.linalg.norm(pooled, dim=-1, keepdim=True)
            norm = torch.clamp(norm, min=1e-6)
            pooled = pooled / norm
            return pooled.to(device=self.device, dtype=self.dtype)

    def _reset_runtime_state(self) -> None:
        self.states.clear()
        self._base_hidden.clear()
        self._attended_history.clear()
        self._completed_streams.clear()
        self._rollback_events.clear()
        self._timings.clear()
        for stream, bus in list(self.bus_by_stream.items()):
            self.bus_by_stream[stream] = DynamicNotesBus(bus.config)
        self.scheduler = TriangularScheduler(
            self.config.streams,
            stride=self.config.stride_B,
        )
        self._active = False
        self._plan_token_ids = None
        self._plan_mask = None
        self._plan_logits = None
        self._plan_source = "none"
        self._gate_values = {stream: self.config.gate_g for stream in self.config.streams}
        self._gate_cooldown = {stream: 0 for stream in self.config.streams}
        self._cadence_events = []
        self._coverage_history = {stream: [] for stream in self.config.streams}
        self._coverage_manifest = {stream: [] for stream in self.config.streams}
        self._plan_embeddings = None
        self._plan_mask_bool = None
        self._plan_ids_list = None
        self._plan_mask_list = None
        self._plan_catalog_entries = None
        self._plan_catalog_index = {}
        self._memory_trace.clear()
        self._replay_plan_recorded = False
        flicker_window = max(2, int(self.config.safeguards.flicker.window))
        self._gate_histories = {
            stream: deque(maxlen=flicker_window) for stream in self.config.streams
        }
        self._plan_histories = {
            stream: deque(maxlen=flicker_window) for stream in self.config.streams
        }
        self._lipschitz_events = []
        self._flicker_events = []
        self._flicker_counters = {"gate": 0, "plan": 0}
        self._lipschitz_probe_budget = 0

    def _stream_completed(self, state: StreamState) -> bool:
        max_tokens = self.decode_config.max_new_tokens
        if max_tokens <= 0:
            return False
        return state.generated_count >= max_tokens

    def _run_trunk(
        self,
        *,
        stream: Optional[str],
        notes: Optional[torch.Tensor],
        notes_mask: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[PastKeyValues],
    ):
        trunk_adapter = getattr(self.model, "trunk_adapter", None)
        # Accept any adapter with a model attribute (duck typing for replay)
        if not hasattr(trunk_adapter, "model"):
            raise RuntimeError("Expected InstrumentedTrunkAdapter for trunk execution.")

        trunk_model = trunk_adapter.model
        payload_notes = notes
        if payload_notes is not None:
            hidden_dtype = (
                self.model.trunk_hidden_dtype()
                if hasattr(self.model, "trunk_hidden_dtype")
                else payload_notes.dtype
            )
            payload_notes = payload_notes.to(device=self.device, dtype=hidden_dtype)
        mask_bool = (
            notes_mask.to(device=self.device, dtype=torch.bool) if notes_mask is not None else None
        )

        with trunk_adapter.activate_context(
            stream=stream,
            notes=payload_notes,
            notes_mask=mask_bool,
        ):
            return trunk_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )

    def _apply_stream_adapter(self, stream: str, hidden_states: torch.Tensor) -> torch.Tensor:
        # Adapters are applied within the instrumented trunk
        return hidden_states

    def _apply_cross_attention(
        self,
        stream: str,
        base_hidden: torch.Tensor,
        adapted: torch.Tensor,
        notes: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # SNC is applied within the instrumented trunk
        return adapted

    def _bootstrap_speculation_notes(self, adapted: torch.Tensor) -> torch.Tensor:
        trunk_adapter = self.model.trunk_adapter
        tap_notes: Optional[torch.Tensor] = None
        consume = getattr(trunk_adapter, "consume_speculation", None)
        if callable(consume):
            tap_notes = consume()
        if tap_notes is not None:
            return tap_notes[:, -1:, :].to(device=adapted.device, dtype=adapted.dtype)
        return self.model.speculation_head(adapted)

    def _lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        head = self.model.trunk_adapter.model.lm_head
        return head(hidden)

    def _blend_logits(self, attended: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        if self.alpha >= 0.999:
            return attended
        return self.alpha * attended + (1.0 - self.alpha) * base

    def _compute_top2_margin(self, logits: torch.Tensor) -> Optional[float]:
        if logits.ndim != 3 or logits.size(1) == 0 or logits.size(-1) == 0:
            return None
        row = logits[:, -1, :]
        if row.ndim != 2 or row.size(0) == 0:
            return None
        k = min(2, row.size(-1))
        values, _ = torch.topk(row, k=k, dim=-1)
        if values.numel() == 0:
            return None
        if k == 1:
            margin = values[:, 0] - values[:, 0]
        else:
            margin = values[:, 0] - values[:, 1]
        return float(margin.squeeze(0).detach().cpu().item())

    def _sample_token(self, logits: torch.Tensor, state: StreamState) -> int:
        scores = logits[:, -1, :]
        scores = scores.squeeze(0)
        if self.decode_config.temperature and self.decode_config.temperature != 1.0:
            scores = scores / self.decode_config.temperature
        scores = self._apply_repetition_penalty(scores, state.generated_tokens)
        if self.decode_config.do_sample:
            filtered = self._top_k_top_p_filter(scores)
            if torch.isinf(filtered).all():
                filtered = scores
            probs = torch.softmax(filtered, dim=-1)
            sample_kwargs: Dict[str, Any] = {}
            if self._rng is not None:
                sample_kwargs["generator"] = self._rng
            token_id = torch.multinomial(probs, num_samples=1, **sample_kwargs)
            return int(token_id.item())
        return int(torch.argmax(scores).item())

    def _apply_repetition_penalty(
        self, logits: torch.Tensor, history: Sequence[int]
    ) -> torch.Tensor:
        penalty = self.decode_config.repetition_penalty
        if penalty <= 1.0 or not history:
            return logits
        unique_tokens = set(history)
        adjusted = logits.clone()
        for token in unique_tokens:
            score = adjusted[token]
            adjusted[token] = torch.sign(score) * (torch.abs(score) / penalty)
        return adjusted

    def _top_k_top_p_filter(self, logits: torch.Tensor) -> torch.Tensor:
        top_k = self.decode_config.top_k
        top_p = self.decode_config.top_p
        filtered = logits.clone()
        if top_k > 0 and top_k < filtered.numel():
            threshold = torch.topk(filtered, top_k).values[-1]
            filtered[filtered < threshold] = float("-inf")
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
            cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            cutoff = cumulative > top_p
            cutoff[..., 0] = False
            filtered_indices = sorted_indices[cutoff]
            filtered[filtered_indices] = float("-inf")
        return filtered

    def _decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id], skip_special_tokens=False)

    def _configure_instrumentation(self, notes_dim: int) -> None:
        trunk_adapter = getattr(self.model, "trunk_adapter", None)
        # Accept any adapter with instrumentation_enabled and set_notes_provider (duck typing for replay)
        if not hasattr(trunk_adapter, "instrumentation_enabled") or not hasattr(
            trunk_adapter, "set_notes_provider"
        ):
            raise RuntimeError(
                "ParallelDecoderTransformer requires an InstrumentedTrunkAdapter. "
                "Set model.instrumentation.enabled=true in your config."
            )
        if not trunk_adapter.instrumentation_enabled:
            raise RuntimeError(
                "Trunk adapter instrumentation is not enabled. "
                "Set model.instrumentation.enabled=true in your config."
            )
        provider = _OrchestratorNotesProvider(self, notes_dim)
        trunk_adapter.set_notes_provider(provider)
        self._notes_provider = provider

    def _health_check_token_id(self) -> int:
        candidates = (
            getattr(self.tokenizer, "bos_token_id", None),
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.tokenizer, "pad_token_id", None),
        )
        for candidate in candidates:
            if candidate is None:
                continue
            if isinstance(candidate, (list, tuple)):
                if candidate:
                    return int(candidate[0])
                continue
            try:
                value = int(candidate)
            except (TypeError, ValueError):
                continue
            if value >= 0:
                return value
        return 0

    def _run_instrumentation_health_check(self, notes_dim: int) -> None:
        trunk_adapter = getattr(self.model, "trunk_adapter", None)
        # Accept any adapter with instrumentation_enabled (duck typing for replay)
        if not hasattr(trunk_adapter, "instrumentation_enabled"):
            raise RuntimeError(
                "Instrumentation health check requires an InstrumentedTrunkAdapter; none was attached."
            )
        if not trunk_adapter.instrumentation_enabled:
            raise RuntimeError("Trunk adapter instrumentation is not enabled.")
        if not self.config.streams:
            raise RuntimeError("Instrumentation requires at least one stream.")
        stream = self.config.streams[0]
        token_id = self._health_check_token_id()
        input_ids = torch.tensor([[token_id]], device=self.device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        try:
            with torch.no_grad():
                outputs = self._run_trunk(
                    stream=stream,
                    notes=None,
                    notes_mask=None,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=None,
                )
        except Exception as exc:
            raise RuntimeError("Instrumentation failed during health check forward pass.") from exc
        past_key_values = outputs.past_key_values
        if past_key_values is None or len(past_key_values) == 0:
            raise RuntimeError("Instrumentation failed health check: missing past_key_values.")
        for layer_index, pair in enumerate(past_key_values):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise RuntimeError(
                    f"Instrumentation failed health check: malformed KV entry at layer {layer_index}."
                )
            for tensor in pair:
                if not torch.is_tensor(tensor) or tensor.numel() == 0:
                    raise RuntimeError(
                        f"Instrumentation failed health check: empty KV tensor at layer {layer_index}."
                    )
        try:
            with torch.no_grad():
                next_token = input_ids[:, -1:]
                next_attention_mask = torch.ones_like(next_token)
                self._run_trunk(
                    stream=stream,
                    notes=None,
                    notes_mask=None,
                    input_ids=next_token,
                    attention_mask=next_attention_mask,
                    past_key_values=past_key_values,
                )
        except Exception as exc:
            raise RuntimeError("Instrumentation failed health check during cache reuse.") from exc
        LOGGER.debug(
            "instrumentation_health_check_passed | stream=%s | notes_dim=%d | layers=%s",
            stream,
            notes_dim,
            trunk_adapter.selected_layers,
        )

    def _resolve_instrumented_layers(self) -> Tuple[int, ...]:
        trunk_adapter = getattr(self.model, "trunk_adapter", None)
        if isinstance(trunk_adapter, InstrumentedTrunkAdapter):
            return trunk_adapter.selected_layers
        return tuple()

    def _stack_attended_history(self, stream: str) -> torch.Tensor:
        history = self._attended_history.get(stream, [])
        if not history:
            return self._base_hidden[stream]
        return torch.cat(history, dim=1)

    def _perform_rollback(self, stream: str, state: StreamState) -> bool:
        removed, restored_kv = state.rollback()
        if not removed:
            return False
        refresh = self._run_trunk(
            stream=stream,
            notes=state.current_notes,
            notes_mask=state.current_notes_mask,
            input_ids=state.input_ids,
            attention_mask=state.attention_mask,
            past_key_values=None,
        )
        state.past_key_values = (
            refresh.past_key_values if refresh.past_key_values is not None else restored_kv
        )
        self._base_hidden[stream] = refresh.hidden_states[-1][:, -1:, :].to(device=self.device)
        self._attended_history[stream] = []
        self._rollback_events.append(
            {
                "stream": stream,
                "tokens_removed": removed,
                "remaining_tokens": state.generated_tokens.copy(),
            }
        )
        corrected = self.model.notes_head(self._base_hidden[stream])
        snapshot = self.bus_by_stream[stream].push(corrected.detach(), stride=len(removed))
        state.mark_snapshot_version(snapshot.version)
        return True

    def _on_stride_complete(self) -> None:
        sync_duration = None
        if self._sync_profile:
            sync_start = time.time()
            self._synchronize_device()
            sync_end = time.time()
            sync_duration = sync_end - sync_start
            now = sync_end
        else:
            now = time.time()
        if self._last_stride_start is not None:
            duration = now - self._last_stride_start
            self._timings.setdefault("stride_durations", []).append(duration)
        self._last_stride_start = now
        if sync_duration is not None:
            self._sync_overhead_total += sync_duration
            self._timings.setdefault("stride_sync_durations", []).append(sync_duration)

    def _mark_versions_consumed(
        self,
        state: StreamState,
        producers: Sequence[str],
        versions: torch.Tensor,
    ) -> None:
        if len(producers) != len(versions):
            return
        for producer, version in zip(producers, versions.tolist()):
            state.update_last_seen_version(producer, int(version))

    def _synchronize_device(self) -> None:
        try:
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            elif (
                self.device.type == "mps"
                and hasattr(torch, "mps")
                and hasattr(torch.mps, "synchronize")
            ):
                torch.mps.synchronize()
        except Exception:
            LOGGER.debug("device_sync_failed", exc_info=True)

    def _track_coverage(self, stream: str, attended: torch.Tensor) -> None:
        if self._plan_embeddings is None or self._plan_mask_bool is None:
            return
        logits = self.model.coverage_head(attended, self._plan_embeddings, self._plan_mask_bool)
        logits_list = [
            float(value) for value in logits.squeeze(0).detach().to("cpu", torch.float32).tolist()
        ]
        self._coverage_history[stream].append(logits_list)

    def _coverage_current(self, stream: str) -> Optional[List[float]]:
        history = self._coverage_history.get(stream)
        if not history:
            return None
        return history[-1]

    def _finalise_coverage(
        self, stream: str, stride_index: int, token_index: int
    ) -> Optional[List[float]]:
        latest = self._coverage_current(stream)
        if latest is None:
            return None
        probabilities = self._sigmoid_probabilities(latest)
        items = self._render_plan_items(probabilities)
        self._coverage_manifest[stream].append(
            {
                "stride_index": stride_index,
                "token_index": token_index,
                "logits": list(latest),
                "probabilities": list(probabilities),
                "plan_items": items,
            }
        )
        return list(latest)

    def _coverage_snapshot(
        self, stream: str, override: Optional[List[float]]
    ) -> Optional[Dict[str, Any]]:
        if self._plan_embeddings is None:
            return None
        latest = override or self._coverage_current(stream)
        if latest is None:
            return None
        latest_copy = list(latest)
        probs = self._sigmoid_probabilities(latest_copy)
        payload: Dict[str, Any] = {
            "latest_logits": latest_copy,
            "latest_probabilities": list(probs),
            "threshold": float(self.config.coverage_threshold),
            "partial_band": float(self.config.coverage_partial_band),
        }
        if self._plan_ids_list is not None:
            payload["plan_ids"] = self._plan_ids_list
        if self._plan_mask_list is not None:
            payload["plan_mask"] = self._plan_mask_list
        plan_items = self._render_plan_items(probs)
        if plan_items:
            payload["plan_items"] = plan_items
        manifest = self._coverage_manifest.get(stream)
        if manifest:
            payload["emissions"] = [dict(record) for record in manifest]
        return payload

    def _sigmoid_probabilities(self, logits: Sequence[float]) -> List[float]:
        if not logits:
            return []
        tensor = torch.tensor(list(logits), dtype=torch.float32)
        return torch.sigmoid(tensor).tolist()

    def _render_plan_items(self, probabilities: Sequence[float]) -> List[Dict[str, Any]]:
        if not probabilities:
            return []
        mask = self._plan_mask_list
        entries: List[Dict[str, Any]] = []
        high = min(1.0, float(self.config.coverage_threshold + self.config.coverage_partial_band))
        low = max(0.0, float(self.config.coverage_threshold - self.config.coverage_partial_band))
        for index, probability in enumerate(probabilities):
            if mask is not None and (index >= len(mask) or not mask[index]):
                continue
            base = self._plan_catalog_index.get(index)
            payload: Dict[str, Any] = {
                "index": index,
                "probability": float(probability),
                "status": self._coverage_status(float(probability), low, high),
            }
            if base:
                for key in ("plan_item_id", "stream", "text"):
                    value = base.get(key)
                    if value is not None:
                        payload[key] = value
            elif self._plan_ids_list is not None and index < len(self._plan_ids_list):
                payload["plan_item_id"] = int(self._plan_ids_list[index])
            entries.append(payload)
        return entries

    def _coverage_status(self, probability: float, low: float, high: float) -> str:
        if probability >= high:
            return "covered"
        if probability <= low:
            return "missing"
        return "partial"

    def _maybe_probe_lipschitz(
        self,
        stream: str,
        base_hidden: torch.Tensor,
        adapted: torch.Tensor,
        notes: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        attended: torch.Tensor,
    ) -> None:
        cfg = self.config.safeguards.lipschitz
        if not cfg.enabled or notes is None or notes.numel() == 0:
            return
        if cfg.probe_interval <= 0:
            return
        step_index = self._step_count + 1
        if step_index % cfg.probe_interval != 0:
            self._lipschitz_probe_budget = 0
            return
        if cfg.max_stream_probes > 0 and self._lipschitz_probe_budget >= cfg.max_stream_probes:
            return
        with torch.no_grad():
            noise = torch.randn_like(notes)
            if mask is not None:
                scaled_mask = mask.to(dtype=noise.dtype, device=noise.device).unsqueeze(-1)
                noise = noise * scaled_mask
            norm = torch.linalg.norm(noise)
            if not torch.isfinite(norm) or float(norm.item()) <= 0.0:
                return
            scaled_noise = noise / norm.clamp_min(1e-6) * cfg.epsilon
            perturbed_notes = notes + scaled_noise
            attended_perturbed = self._apply_cross_attention(
                stream,
                base_hidden,
                adapted,
                perturbed_notes,
                mask,
            )
            ratio = self._lipschitz_ratio(attended_perturbed - attended, scaled_noise)
        self._lipschitz_probe_budget += 1
        violated = ratio > cfg.threshold
        if violated:
            LOGGER.warning(
                "lipschitz_violation | stream=%s | step=%d | estimate=%.4f | threshold=%.4f",
                stream,
                step_index,
                ratio,
                cfg.threshold,
            )
        event = {
            "step": step_index,
            "stream": stream,
            "epsilon": float(cfg.epsilon),
            "L_u_est": float(ratio),
            "threshold": float(cfg.threshold),
            "violated": violated,
        }
        self._lipschitz_events.append(event)
        if len(self._lipschitz_events) > cfg.max_history:
            self._lipschitz_events.pop(0)

    def _track_flicker(
        self,
        stream: str,
        gate_value: float,
        coverage_logits: Optional[Sequence[float]],
    ) -> None:
        cfg = self.config.safeguards.flicker
        if not cfg.enabled:
            return
        gate_history = self._gate_histories.setdefault(stream, deque(maxlen=max(2, cfg.window)))
        gate_history.append(float(gate_value))
        gate_std = None
        gate_violation = False
        if len(gate_history) >= 2:
            gate_std = self._stddev(gate_history)
            if gate_std > cfg.gate_std_threshold:
                gate_violation = True
                self._flicker_counters["gate"] += 1
        plan_history = self._plan_histories.setdefault(stream, deque(maxlen=max(2, cfg.window)))
        plan_history.append(self._top_plan_index(coverage_logits))
        plan_switches = None
        plan_violation = False
        if len(plan_history) >= 2:
            plan_switches = self._count_plan_switches(plan_history)
            if plan_switches >= cfg.plan_switch_threshold:
                plan_violation = True
                self._flicker_counters["plan"] += 1
        if not (gate_violation or plan_violation):
            return
        clamp_applied = False
        if cfg.clamp_on_violation:
            clamp_target = min(1.0, max(0.0, cfg.clamp_value))
            current_gate = self._gate_values.get(stream, self.config.gate_g)
            if current_gate > clamp_target:
                self._gate_values[stream] = clamp_target
                clamp_applied = True
        LOGGER.warning(
            "flicker_detected | stream=%s | gate_std=%s | plan_switches=%s | clamped=%s",
            stream,
            "n/a" if gate_std is None else f"{gate_std:.4f}",
            plan_switches,
            clamp_applied,
        )
        event = {
            "step": self._step_count + 1,
            "stream": stream,
            "gate_std": gate_std,
            "plan_switches": plan_switches,
            "gate_violation": gate_violation,
            "plan_violation": plan_violation,
            "clamped": clamp_applied,
        }
        max_events = max(8, cfg.window * len(self.config.streams))
        if len(self._flicker_events) >= max_events:
            self._flicker_events.pop(0)
        self._flicker_events.append(event)

    def _build_safeguards_manifest(self) -> Optional[Dict[str, Any]]:
        lipschitz_events = getattr(self, "_lipschitz_events", None) or []
        flicker_events = getattr(self, "_flicker_events", None) or []
        if not lipschitz_events and not flicker_events:
            return None
        payload: Dict[str, Any] = {}
        cfg = self.config.safeguards
        if lipschitz_events:
            payload["lipschitz"] = {
                "events": list(lipschitz_events),
                "epsilon": float(cfg.lipschitz.epsilon),
                "threshold": float(cfg.lipschitz.threshold),
                "probe_interval": int(cfg.lipschitz.probe_interval),
            }
        if flicker_events:
            payload["flicker"] = {
                "events": list(flicker_events),
                "gate_violations": int(getattr(self, "_flicker_counters", {}).get("gate", 0)),
                "plan_violations": int(getattr(self, "_flicker_counters", {}).get("plan", 0)),
                "window": int(cfg.flicker.window),
                "gate_std_threshold": float(cfg.flicker.gate_std_threshold),
                "plan_switch_threshold": int(cfg.flicker.plan_switch_threshold),
            }
        return payload or None

    @staticmethod
    def _lipschitz_ratio(output_delta: torch.Tensor, noise_delta: torch.Tensor) -> float:
        if output_delta.numel() == 0 or noise_delta.numel() == 0:
            return 0.0
        out_norm = torch.linalg.norm(output_delta).detach()
        in_norm = torch.linalg.norm(noise_delta).detach()
        denom = float(in_norm.cpu().item())
        if denom <= 0.0:
            return 0.0
        ratio = float((out_norm / in_norm).cpu().item())
        if not math.isfinite(ratio):
            return 0.0
        return ratio

    @staticmethod
    def _stddev(values: Sequence[float]) -> float:
        entries = [float(value) for value in values if math.isfinite(float(value))]
        if not entries:
            return 0.0
        mean = sum(entries) / len(entries)
        variance = sum((value - mean) ** 2 for value in entries) / len(entries)
        return math.sqrt(max(variance, 0.0))

    @staticmethod
    def _count_plan_switches(history: Sequence[Optional[int]]) -> int:
        switches = 0
        previous: Optional[int] = None
        for value in history:
            if value is None:
                continue
            if previous is None:
                previous = value
                continue
            if value != previous:
                switches += 1
            previous = value
        return switches

    def _top_plan_index(self, coverage_logits: Optional[Sequence[float]]) -> Optional[int]:
        if not coverage_logits:
            return None
        mask = self._plan_mask_list
        best_idx: Optional[int] = None
        best_value = float("-inf")
        for index, value in enumerate(coverage_logits):
            if mask is not None:
                if index >= len(mask) or not mask[index]:
                    continue
            float_value = float(value)
            if float_value > best_value:
                best_idx = index
                best_value = float_value
        return best_idx

    def _should_emit_notes(
        self,
        stream: str,
        state: StreamState,
        cadence: int,
        agreement_score: float,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        policy = self.config.cadence_policy
        if policy.mode == "deterministic":
            return state.cadence_reached(cadence), None

        tokens_since = state.tokens_since_snapshot
        if policy.max_interval > 0 and tokens_since >= policy.max_interval:
            metadata = {
                "mode": policy.mode,
                "forced": True,
                "base_probability": 1.0,
                "multiplier": 1.0,
                "final_probability": 1.0,
                "sample": None,
                "tokens_since": tokens_since,
            }
            return True, metadata

        base_prob = min(1.0, max(policy.min_probability, 1.0 / max(1, cadence)))
        multiplier = 1.0
        reason = "base"

        if policy.mode == "adaptive":
            clamped = max(0.0, min(1.0, agreement_score))
            if clamped <= policy.agreement_low:
                multiplier = policy.multiplier_max
                reason = "agreement_low"
            elif clamped >= policy.agreement_high:
                multiplier = policy.multiplier_min
                reason = "agreement_high"
            else:
                span = policy.agreement_high - policy.agreement_low
                ratio = (policy.agreement_high - clamped) / span
                multiplier = policy.multiplier_min + (
                    (policy.multiplier_max - policy.multiplier_min) * ratio
                )
                reason = "agreement_interp"
            if policy.age_boost > 0.0:
                age_ratio = max(0.0, tokens_since / max(1, cadence) - 1.0)
                if age_ratio > 0.0:
                    age_multiplier = 1.0 + policy.age_boost * age_ratio
                    multiplier *= age_multiplier
                    reason = "agreement_age"
            multiplier = max(policy.multiplier_min, min(policy.multiplier_max, multiplier))

        final_prob = min(1.0, max(policy.min_probability, base_prob * multiplier))

        if self._rng is not None:
            sample = torch.rand(1, generator=self._rng).item()
        else:
            sample = torch.rand(1).item()
        emit = sample < final_prob

        metadata = {
            "mode": policy.mode,
            "base_probability": base_prob,
            "multiplier": multiplier,
            "final_probability": final_prob,
            "sample": sample,
            "tokens_since": tokens_since,
            "forced": False,
            "emitted": emit,
        }
        if policy.mode == "adaptive":
            metadata["reason"] = reason
        return emit, metadata

    def _update_gate_on_emission(
        self,
        stream: str,
        agreement_result: AgreementResult,
        agreement_score: float,
    ) -> None:
        policy = self.config.gate_annealing
        if not policy.enabled:
            return
        current = self._gate_values.get(stream, self.config.gate_g)
        margin = self.config.agreement_threshold_tau + policy.stability_margin
        volatile = agreement_result.triggered or (agreement_score < margin)
        if volatile:
            updated = max(policy.min_value, current * policy.decay)
            self._gate_values[stream] = updated
            self._gate_cooldown[stream] = policy.cooldown
        else:
            self._recover_gate(stream)

    def _update_gate_on_stable_step(self, stream: str, agreement_score: float) -> None:
        policy = self.config.gate_annealing
        if not policy.enabled:
            return
        margin = self.config.agreement_threshold_tau + policy.stability_margin
        if agreement_score < margin:
            return
        self._recover_gate(stream)

    def _recover_gate(self, stream: str) -> None:
        policy = self.config.gate_annealing
        if not policy.enabled:
            return
        cooldown = self._gate_cooldown.get(stream, 0)
        if cooldown > 0:
            self._gate_cooldown[stream] = cooldown - 1
            return
        current = self._gate_values.get(stream, self.config.gate_g)
        if current >= self.config.gate_g:
            self._gate_values[stream] = self.config.gate_g
            return
        updated = min(self.config.gate_g, current + policy.recovery)
        self._gate_values[stream] = updated

    def _planner_notes_from_text(
        self,
        plan_text_by_stream: Mapping[str, Sequence[str]],
    ) -> Optional[Dict[str, torch.Tensor]]:
        normalized = normalise_plan_map(plan_text_by_stream, self.config.streams)
        flattened: List[str] = []
        stream_sequence: List[str] = []
        for stream in self.config.streams:
            entries = normalized.get(stream, [])
            for text in entries:
                flattened.append(text)
                stream_sequence.append(stream)
        if not flattened:
            return None
        plan_vocab = getattr(self.model.plan_embedding, "num_embeddings", None)
        if plan_vocab is None:
            raise ValueError("Model plan_embedding missing num_embeddings; cannot hash plan text.")
        bucket_count = self._plan_hash_buckets or int(plan_vocab)
        plan_ids = [
            hash_plan_text(text, bucket_count, salt=self._plan_hash_salt) for text in flattened
        ]
        tensor = torch.tensor(plan_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        mask = torch.ones_like(tensor)
        entries: List[Dict[str, Any]] = []
        for index, (stream, text, plan_id) in enumerate(zip(stream_sequence, flattened, plan_ids)):
            entries.append(
                {
                    "index": index,
                    "stream": stream,
                    "text": text,
                    "plan_item_id": int(plan_id),
                }
            )
        self._set_plan_catalog_entries(entries)
        return {
            "plan_token_ids": tensor,
            "plan_mask": mask,
        }

    def _set_plan_catalog_entries(self, entries: Optional[List[Dict[str, Any]]]) -> None:
        if not entries:
            self._plan_catalog_entries = None
            self._plan_catalog_index = {}
            return
        self._plan_catalog_entries = [dict(entry) for entry in entries]
        self._plan_catalog_index = {
            int(entry.get("index", idx)): dict(entry)
            for idx, entry in enumerate(self._plan_catalog_entries)
        }

    def _build_catalog_from_plan_ids(self) -> List[Dict[str, Any]]:
        if self._plan_ids_list is None:
            return []
        mask = self._plan_mask_list or [1] * len(self._plan_ids_list)
        entries: List[Dict[str, Any]] = []
        for index, plan_id in enumerate(self._plan_ids_list):
            mask_value = mask[index] if index < len(mask) else 1
            if not mask_value:
                continue
            entries.append(
                {
                    "index": index,
                    "plan_item_id": int(plan_id),
                }
            )
        return entries

    def _normalise_planner_payload(
        self,
        payload: Any,
        attention_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        plan_logits: Optional[torch.Tensor] = None
        source = "external"
        if isinstance(payload, dict):
            if "plan_token_ids" not in payload:
                raise ValueError("planner_notes dict must include 'plan_token_ids'.")
            plan_ids = self._coerce_plan_ids(payload["plan_token_ids"], attention_mask)
            plan_mask = self._coerce_plan_mask(
                payload.get("plan_mask", attention_mask.clone()),
                attention_mask,
            )
            raw_logits = payload.get("plan_logits")
            if raw_logits is not None:
                plan_logits = raw_logits.to(device=self.device)
        else:
            plan_ids = self._coerce_plan_ids(payload, attention_mask)
            plan_mask = attention_mask.clone()
        return {
            "plan_token_ids": plan_ids,
            "plan_mask": plan_mask,
            "plan_logits": plan_logits,
            "source": source,
        }

    def _coerce_plan_ids(self, value: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=self.device, dtype=torch.long)
        elif isinstance(value, (list, tuple)):
            tensor = torch.tensor(value, dtype=torch.long, device=self.device)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
        else:
            raise ValueError("Unsupported plan_token_ids payload type.")
        if tensor.dim() != 2 or tensor.size(0) != 1:
            raise ValueError("plan_token_ids must be shaped [1, seq_len].")
        target_len = attention_mask.size(1)
        if tensor.size(1) != target_len:
            raise ValueError(
                f"plan_token_ids length ({tensor.size(1)}) must match prompt length ({target_len})."
            )
        return tensor

    def _coerce_plan_mask(self, value: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=self.device, dtype=attention_mask.dtype)
        elif isinstance(value, (list, tuple)):
            tensor = torch.tensor(value, dtype=attention_mask.dtype, device=self.device)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
        else:
            raise ValueError("Unsupported plan_mask payload type.")
        if tensor.dim() != 2 or tensor.size(0) != 1:
            raise ValueError("plan_mask must be shaped [1, seq_len].")
        target_len = attention_mask.size(1)
        if tensor.size(1) != target_len:
            raise ValueError(
                f"plan_mask length ({tensor.size(1)}) must match prompt length ({target_len})."
            )
        return tensor

    def _tensor_to_int_list(self, tensor: torch.Tensor) -> List[List[int]]:
        payload = tensor.detach().to(device="cpu", dtype=torch.long)
        return [[int(value) for value in row] for row in payload.tolist()]

    def _derive_plan_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Derive a plan mask gated by the first `</plan>` sentinel if present."""

        if self._plan_token_ids is None:
            return attention_mask
        end_token_id = self._resolve_plan_end_token_id()
        if end_token_id is None:
            return attention_mask
        plan_row = self._plan_token_ids[0]
        try:
            end_index = plan_row.tolist().index(end_token_id)
        except ValueError:
            return attention_mask
        mask = attention_mask.clone()
        mask[:, end_index + 1 :] = 0
        return mask

    def _resolve_plan_end_token_id(self) -> Optional[int]:
        token = "</plan>"
        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if convert is None:
            return None
        try:
            token_id = convert(token)
        except Exception:
            return None
        if isinstance(token_id, int) and token_id >= 0:
            return token_id
        return None

    def _resolve_notes_dim(self) -> int:
        config = getattr(self.model, "config", None)
        if config is None:
            raise RuntimeError("Model configuration missing; notes_dim cannot be resolved.")
        notes_dim = getattr(config, "notes_dim", None)
        if notes_dim is None:
            head_cfg = getattr(config, "notes_head", None)
            if head_cfg is None or getattr(head_cfg, "notes_dim", None) is None:
                raise RuntimeError("notes_dim not available on model configuration.")
            notes_dim = head_cfg.notes_dim
        return int(notes_dim)

    def _resolve_model_hidden_size(self) -> int:
        config = getattr(self.model, "config", None)
        hidden = getattr(config, "hidden_size", None)
        if hidden is not None:
            return int(hidden)
        return self._resolve_notes_dim()

    def _resolve_bus_dtype(self) -> str:
        config = getattr(self.model, "config", None)
        if config is None or getattr(config, "notes_bus", None) is None:
            return "bfloat16"
        bus_cfg = config.notes_bus
        return str(getattr(bus_cfg, "dtype", "bfloat16"))

    def _resolve_device(self) -> torch.device:
        if hasattr(self.model.trunk_adapter.model, "device"):
            device = getattr(self.model.trunk_adapter.model, "device")
            if isinstance(device, torch.device):
                return device
        first_param = next(self.model.trunk_adapter.model.parameters())
        return first_param.device

    def _resolve_dtype(self) -> torch.dtype:
        first_param = next(self.model.trunk_adapter.model.parameters())
        return first_param.dtype

    def _build_generator(self, seed: Optional[int]) -> Optional[torch.Generator]:
        if seed is None:
            return None
        random.seed(seed)
        try:  # pragma: no branch - optional dependency
            import numpy as np

            np.random.seed(seed)
        except Exception:
            pass
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        torch.manual_seed(seed)
        return generator

    @staticmethod
    def _validate_decode_config(config: DecodeConfig) -> None:
        if config.top_k < 0:
            raise ValueError("DecodeConfig.top_k must be non-negative.")
        if config.top_p < 0 or config.top_p > 1:
            raise ValueError("DecodeConfig.top_p must lie within [0, 1].")


__all__ = [
    "AgreementGate",
    "AgreementResult",
    "MultiStreamOrchestrator",
    "StepOutcome",
]
