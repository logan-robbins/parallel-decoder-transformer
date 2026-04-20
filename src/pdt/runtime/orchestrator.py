"""Multi-stream inference orchestrator.

Responsibilities:

- Run the planner on the prompt, sample per-slot plan IDs, seed per-stream
  snapshot-0 on the Dynamic Notes Bus via ``plan_notes_proj``.
- Advance each stream one token at a time in a round-robin schedule.
- Assemble the visible notes window per stream via ``NotesWindowBuilder``.
- Thread ``LayerRuntimeContext`` into every instrumented trunk layer so
  SNC + per-stream adapter deltas execute correctly.
- At block boundaries (every ``\u03c4`` tokens): compute coverage + readiness
  per stream, decide commit vs rollback, publish new note snapshots.

This is the runtime-only path (inference + ablation). Training reuses a
small subset (prompt encode + planner forward) via
``pdt.training.trainer``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizerBase

from pdt.config.schemas import (
    NotesBusConfig,
    PDTConfig,
    RuntimeConfig,
)
from pdt.model import PDTModel
from pdt.runtime.counterfactuals import (
    CounterfactualConfig,
    apply_anchor_swap,
    apply_gate_ablation,
    apply_norm_scramble,
)
from pdt.runtime.dnb_bus import DynamicNotesBus
from pdt.runtime.state import StreamState
from pdt.runtime.window import NotesWindow, NotesWindowBuilder, TopologyMask
from pdt.trunk.instrumentation import LayerRuntimeContext


LOGGER = logging.getLogger("pdt.runtime.orchestrator")


__all__ = ["MultiStreamOrchestrator", "OrchestrationResult"]


@dataclass(slots=True)
class OrchestrationResult:
    text_by_stream: Dict[str, str]
    tokens_by_stream: Dict[str, List[int]]
    plan_slot_ids: torch.Tensor  # (1, S)
    planner_logits: torch.Tensor  # (1, S, V_p)
    snapshot0_anchors: torch.Tensor  # (1, K, d_notes)
    agreement_history: List[Dict[str, float]]
    rollback_events: List[Dict[str, object]]


class MultiStreamOrchestrator:
    def __init__(
        self,
        model: PDTModel,
        tokenizer: PreTrainedTokenizerBase,
        config: PDTConfig,
        *,
        counterfactual: Optional[CounterfactualConfig] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.counterfactual = counterfactual or CounterfactualConfig(mode="none")
        self.device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")

        self.streams: Tuple[str, ...] = tuple(config.runtime.streams)
        if len(self.streams) != config.sidecar.num_streams:
            raise ValueError(
                f"runtime.streams ({len(self.streams)}) != sidecar.num_streams "
                f"({config.sidecar.num_streams})"
            )
        self.topology = TopologyMask(self.streams, config.runtime.topology)
        self.window_builder = NotesWindowBuilder(
            notes_dim=config.sidecar.notes_dim,
            topology_mask=self.topology,
            read_lag=config.runtime.notes_bus.lag,
            max_snapshots=config.runtime.notes_bus.max_snapshots,
            device=self.device,
            self_only_tokens=config.runtime.self_only_tokens,
        )

        if self.counterfactual.seed is not None:
            self._rng = torch.Generator(device="cpu")
            self._rng.manual_seed(self.counterfactual.seed)
        else:
            self._rng = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        ownership_override: Optional[torch.Tensor] = None,
    ) -> OrchestrationResult:
        """Run the full multi-stream decoding loop for one prompt.

        Args:
            prompt: User input text.
            max_new_tokens: Per-stream token budget.
            ownership_override: Optional ``(1, K, S)`` bool override of the
                planner's disjoint-ownership assignment. When ``None`` a
                default round-robin assignment is used.
        """
        tokenized = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_ids = tokenized["input_ids"]
        prompt_mask = tokenized["attention_mask"]

        # -------- Planner pass on the prompt -------- #
        trunk_out = self.model.trunk_adapter.forward(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            use_cache=True,
            output_hidden_states=True,
        )
        prompt_hidden = trunk_out.hidden_states[-1]
        planner_logits = self.model.sidecar.planner_head(
            prompt_hidden, attention_mask=prompt_mask
        )
        slot_ids = planner_logits.argmax(dim=-1)  # (1, S)
        slot_embeddings = self.model.sidecar.plan_embedding(slot_ids)  # (1, S, H)

        if ownership_override is None:
            ownership = _default_ownership(
                batch=1,
                num_streams=self.config.sidecar.num_streams,
                num_slots=self.config.sidecar.planner_head.num_slots,
                device=self.device,
            )
        else:
            ownership = ownership_override.to(self.device)

        snapshot0 = self.model.sidecar.plan_notes_proj(
            slot_embeddings, ownership
        )  # (1, K, d_notes)

        # -------- Seed per-stream DNB buses + state -------- #
        bus_by_stream: Dict[str, DynamicNotesBus] = {
            stream: DynamicNotesBus(self.config.runtime.notes_bus, device=str(self.device))
            for stream in self.streams
        }
        for idx, stream in enumerate(self.streams):
            anchor = snapshot0[0, idx].unsqueeze(0)  # (1, d_notes)
            snap = bus_by_stream[stream].push(anchor.squeeze(0), stride=0)
            # Seed state against this snapshot.

        # Apply anchor-swap counterfactual if requested.
        if self.counterfactual.mode == "anchor_swap":
            if self.counterfactual.alt_prompt_anchors is None:
                raise ValueError("anchor_swap requires alt_prompt_anchors in config.")
            apply_anchor_swap(
                bus_by_stream,
                self.counterfactual.alt_prompt_anchors.to(self.device),
                self.streams,
            )

        # Per-stream states: each stream starts with the prompt duplicated.
        states: Dict[str, StreamState] = {}
        for stream in self.streams:
            states[stream] = StreamState(
                stream=stream,
                input_ids=prompt_ids.clone(),
                attention_mask=prompt_mask.clone(),
                commit_stride=self.config.runtime.block_size,
                commit_horizon=self.config.runtime.commit_horizon,
                past_key_values=None,  # Re-encoded below, per-stream.
            )

        # -------- Per-stream prompt encode (stream-conditioned) -------- #
        for stream, state in states.items():
            ctx = LayerRuntimeContext(
                stream=stream, notes=None, notes_mask=None, snc_force_gate=None
            )
            self._set_context(ctx)
            out = self.model.trunk_adapter.forward(
                input_ids=state.input_ids,
                attention_mask=state.attention_mask,
                use_cache=True,
                output_hidden_states=False,
            )
            state.past_key_values = out.past_key_values

        self._clear_context()

        # -------- Generation loop -------- #
        agreement_history: List[Dict[str, float]] = []
        rollback_events: List[Dict[str, object]] = []
        stride_scores: Dict[str, float] = {stream: 1.0 for stream in self.streams}
        stride_counter = 0

        for step in range(max_new_tokens):
            for stream in self.streams:
                state = states[stream]

                # Build the visible window.
                window = self.window_builder.build(state, bus_by_stream)
                notes_tensor = window.notes
                mask_tensor = window.mask

                # Apply norm-scramble counterfactual if active.
                if self.counterfactual.mode == "norm_scramble" and notes_tensor.numel() > 0:
                    notes_tensor = apply_norm_scramble(notes_tensor, generator=self._rng)

                # Wire runtime context into every instrumented layer.
                force_gate = (
                    apply_gate_ablation()
                    if self.counterfactual.mode == "gate_zero"
                    else None
                )
                ctx = LayerRuntimeContext(
                    stream=stream,
                    notes=notes_tensor,
                    notes_mask=mask_tensor,
                    snc_force_gate=force_gate,
                )
                self._set_context(ctx)
                state.update_notes_window(notes_tensor, mask_tensor)
                state.update_last_seen_version_from_window(window)

                # Single-token forward (step mode).
                out = self.model.trunk_adapter.forward(
                    input_ids=state.input_ids[:, -1:],
                    attention_mask=state.attention_mask,
                    past_key_values=state.past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                )
                hidden = out.hidden_states[-1][:, -1:, :]
                logits = out.logits[:, -1, :]
                next_token = int(logits.argmax(dim=-1).item())
                piece = self.tokenizer.decode([next_token])

                # Per-token agreement readout (over the same hidden, no
                # coverage signal yet -- the paper-level agreement fires at
                # block boundaries below).
                stride_scores[stream] = float(
                    torch.sigmoid(hidden.mean()).item()
                )

                state.append_token(
                    next_token, past_key_values=out.past_key_values, token_text=piece
                )

                # Block boundary?
                if state.tokens_since_snapshot >= self.config.runtime.block_size:
                    self._emit_note_snapshot(state, hidden, bus_by_stream[stream])
                    state.reset_snapshot_counter()

                if state.tokens_since_commit >= self.config.runtime.block_size:
                    state.register_commit()

            # After a full round across streams, run the agreement gate.
            stride_counter += 1
            if stride_counter % self.config.runtime.block_size == 0:
                scores_snapshot = dict(stride_scores)
                agreement_history.append(scores_snapshot)
                min_score = min(scores_snapshot.values())
                if min_score < self.config.runtime.agreement_threshold:
                    for stream, score in scores_snapshot.items():
                        if score < self.config.runtime.agreement_threshold:
                            state = states[stream]
                            if state.can_rollback():
                                removed, _ = state.rollback()
                                rollback_events.append(
                                    {
                                        "stream": stream,
                                        "tokens_removed": len(removed),
                                        "score": score,
                                        "step": step,
                                    }
                                )

        self._clear_context()

        return OrchestrationResult(
            text_by_stream={s: states[s].generated_text for s in self.streams},
            tokens_by_stream={s: list(states[s].generated_tokens) for s in self.streams},
            plan_slot_ids=slot_ids,
            planner_logits=planner_logits,
            snapshot0_anchors=snapshot0,
            agreement_history=agreement_history,
            rollback_events=rollback_events,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _set_context(self, context: LayerRuntimeContext) -> None:
        for layer in self.model.instrumented_layers:
            layer.set_runtime_context(context)

    def _clear_context(self) -> None:
        for layer in self.model.instrumented_layers:
            layer.set_runtime_context(None)

    def _emit_note_snapshot(
        self,
        state: StreamState,
        block_hidden: torch.Tensor,
        bus: DynamicNotesBus,
    ) -> None:
        """Run SpeculationHead on the block-end hidden, push onto the bus."""
        spec = self.model.sidecar.speculation_head(block_hidden)
        # (1, 1, notes_dim) -> (notes_dim,)
        vec = spec[0, -1]
        snapshot = bus.push(vec, stride=state.total_tokens)
        state.mark_snapshot_version(snapshot.version)


def _default_ownership(
    batch: int,
    num_streams: int,
    num_slots: int,
    device: torch.device,
) -> torch.Tensor:
    """Round-robin assign slots to streams.

    slot s is owned by stream (s % num_streams). Shape: (B, K, S) bool.
    """
    ownership = torch.zeros(batch, num_streams, num_slots, dtype=torch.bool, device=device)
    for s in range(num_slots):
        ownership[:, s % num_streams, s] = True
    return ownership


# Monkey-patch a helper onto StreamState since this is the only consumer. #
def _update_last_seen_version_from_window(self: StreamState, window: NotesWindow) -> None:
    """Advance last_seen_version per producer, silently skipping replays."""
    for producer, version in zip(window.producers, window.versions.tolist()):
        cached = self.last_seen_version.get(producer)
        if cached is None or version > cached:
            self.last_seen_version[producer] = int(version)


StreamState.update_last_seen_version_from_window = _update_last_seen_version_from_window  # type: ignore[attr-defined]
