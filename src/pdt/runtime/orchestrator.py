"""Multi-stream inference orchestrator.

Responsibilities:

- Run the planner on the prompt, sample per-slot plan IDs, seed per-stream
  snapshot-0 on the Dynamic Notes Bus via ``plan_notes_proj``.
- Advance all stream frontier tokens in one packed trunk call per round.
- Assemble the visible notes window per stream via ``NotesWindowBuilder``.
- Thread ``LayerRuntimeContext`` into every instrumented trunk layer so
  SNC + per-stream adapter deltas execute correctly.
- At block boundaries (every ``\u03c4`` tokens), synchronously publish one
  finite product-VQ code tuple per stream. Commit control remains out of scope
  until a trained, validated controller exists.

This is the runtime-only path (inference + ablation). Training reuses a
small subset (prompt encode + planner forward) via
``pdt.training.trainer``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Dict, List, Mapping, Optional, Tuple

import torch
from transformers import PreTrainedTokenizerBase

from pdt.config.schemas import PDTConfig
from pdt.model import PDTModel
from pdt.prompts import planner_user_text, stream_user_text
from pdt.runtime.counterfactuals import (
    CounterfactualConfig,
    apply_anchor_swap,
    apply_bus_mutation,
    apply_gate_ablation,
    apply_norm_scramble,
    apply_source_swap,
)
from pdt.runtime.dnb_bus import DynamicNotesBus
from pdt.runtime.state import PackedFrontierState, StreamState, pack_token_rows
from pdt.runtime.window import NotesWindowBuilder
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
    dynamic_codes_by_stream: Dict[str, List[Tuple[int, ...]]]


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
        if config.instrumentation.coordination_source != "bus":
            raise ValueError(
                "MultiStreamOrchestrator is the physical bus runtime and requires "
                "instrumentation.coordination_source='bus'. The self-only control "
                "is trained and scored through PDTTrainer."
            )
        self.counterfactual = counterfactual or CounterfactualConfig(mode="none")
        first_parameter = next(model.parameters(), None)
        self.device = first_parameter.device if first_parameter is not None else torch.device("cpu")

        self.streams: Tuple[str, ...] = tuple(config.runtime.streams)
        if len(self.streams) != config.sidecar.num_streams:
            raise ValueError(
                f"runtime.streams ({len(self.streams)}) != sidecar.num_streams "
                f"({config.sidecar.num_streams})"
            )
        self.window_builder = NotesWindowBuilder(
            producers=self.streams,
            notes_dim=config.sidecar.notes_dim,
            block_size=config.runtime.block_size,
            device=self.device,
        )

        if self.counterfactual.mode == "bus_mutation":
            mutation_producer = (self.counterfactual.mutation_producer or self.streams[0]).lower()
            if mutation_producer not in self.streams:
                raise ValueError(
                    f"Unknown mutation producer {mutation_producer!r}; "
                    f"expected one of {self.streams}."
                )
            if self.counterfactual.mutation_block < 0:
                raise ValueError("mutation_block must be non-negative.")
            code_offset = self.counterfactual.mutation_code_offset
            code_count = self.config.sidecar.speculation_head.codes_per_codebook
            if type(code_offset) is not int or not 0 < code_offset < code_count:
                raise ValueError(
                    "mutation_code_offset must be an integer in "
                    f"[1, {code_count}); got {code_offset!r}."
                )

        self._rng: Optional[torch.Generator]
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
        prompt_ids, prompt_mask = _tokenize_user_prompt(
            self.tokenizer,
            prompt,
            device=self.device,
        )
        stream_prompts = {
            stream: (prompt_ids.clone(), prompt_mask.clone()) for stream in self.streams
        }
        return self._generate_from_tokenized_prompts(
            planner_prompt_ids=prompt_ids,
            planner_prompt_attention_mask=prompt_mask,
            stream_prompts=stream_prompts,
            max_new_tokens=max_new_tokens,
            ownership_override=ownership_override,
        )

    @torch.no_grad()
    def generate_structured(
        self,
        shared_prompt: str,
        stream_local_prompts: Mapping[str, str],
        stream_block_transition_ids: Mapping[str, Sequence[Sequence[int]]],
        *,
        max_new_tokens: int = 128,
        ownership_override: Optional[torch.Tensor] = None,
    ) -> OrchestrationResult:
        """Decode a temporal mechanism example with one private reveal per block.

        The planner sees only ``shared_prompt``. Each continuation's prompt
        contains block-0's addressed observation. At every later boundary the
        decoder consumes the exact chat-template suffix stored by canonical
        retokenization (close assistant, add the next private user observation,
        reopen assistant) under that block's newly visible notes context.
        """

        normalized = {str(key).lower(): value for key, value in stream_local_prompts.items()}
        normalized_transitions = {
            str(key).lower(): value for key, value in stream_block_transition_ids.items()
        }
        if max_new_tokens <= 0 or max_new_tokens % self.config.runtime.block_size != 0:
            raise ValueError(
                "Structured mechanism decoding requires a positive whole number of "
                f"tau-token blocks; max_new_tokens={max_new_tokens}, "
                f"tau={self.config.runtime.block_size}."
            )
        expected = set(self.streams)
        if set(normalized) != expected:
            missing = sorted(expected - set(normalized))
            extra = sorted(set(normalized) - expected)
            raise ValueError(
                "stream_local_prompts must contain exactly runtime.streams; "
                f"missing={missing}, extra={extra}."
            )
        if set(normalized_transitions) != expected:
            missing = sorted(expected - set(normalized_transitions))
            extra = sorted(set(normalized_transitions) - expected)
            raise ValueError(
                "stream_block_transition_ids must contain exactly runtime.streams; "
                f"missing={missing}, extra={extra}."
            )
        block_count = max_new_tokens // self.config.runtime.block_size
        transitions = {
            stream: _validate_structured_transitions(
                normalized_transitions[stream],
                stream=stream,
                block_count=block_count,
                device=self.device,
            )
            for stream in self.streams
        }
        shared = planner_user_text(shared_prompt)
        planner_ids, planner_mask = _tokenize_user_prompt(
            self.tokenizer,
            shared,
            device=self.device,
        )
        stream_prompts = {}
        for stream in self.streams:
            local = normalized[stream]
            if not isinstance(local, str) or not local.strip():
                raise ValueError(f"Private prompt for {stream!r} must be non-empty text.")
            user_text = stream_user_text(shared, stream, local)
            stream_prompts[stream] = _tokenize_user_prompt(
                self.tokenizer,
                user_text,
                device=self.device,
            )
        return self._generate_from_tokenized_prompts(
            planner_prompt_ids=planner_ids,
            planner_prompt_attention_mask=planner_mask,
            stream_prompts=stream_prompts,
            stream_block_transitions=transitions,
            max_new_tokens=max_new_tokens,
            ownership_override=ownership_override,
        )

    @torch.no_grad()
    def _generate_from_tokenized_prompts(
        self,
        *,
        planner_prompt_ids: torch.Tensor,
        planner_prompt_attention_mask: torch.Tensor,
        stream_prompts: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
        stream_block_transitions: Optional[Mapping[str, tuple[torch.Tensor, ...]]] = None,
        max_new_tokens: int,
        ownership_override: Optional[torch.Tensor],
    ) -> OrchestrationResult:
        """The single canonical generation loop for natural and structured input."""

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if set(stream_prompts) != set(self.streams):
            raise ValueError("Tokenized stream prompts must exactly match runtime.streams.")
        if stream_block_transitions is not None and set(stream_block_transitions) != set(
            self.streams
        ):
            raise ValueError("Tokenized block transitions must exactly match runtime.streams.")
        _validate_tokenized_prompt(
            planner_prompt_ids,
            planner_prompt_attention_mask,
            name="planner prompt",
        )
        for stream in self.streams:
            ids, mask = stream_prompts[stream]
            _validate_tokenized_prompt(ids, mask, name=f"{stream} prompt")

        prompt_ids = planner_prompt_ids
        prompt_mask = planner_prompt_attention_mask

        # -------- Planner pass on the prompt -------- #
        self._clear_context()
        trunk_out = self.model.trunk_adapter.forward(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        prompt_hidden = trunk_out.hidden_states[-1]
        planner = self.model.sidecar.planner_head(prompt_hidden, attention_mask=prompt_mask)
        slot_ids = planner.indices  # (1, S)

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
            planner.quantized, ownership
        )  # (1, K, d_notes)

        # -------- Seed the shared addressed DNB + per-stream state -------- #
        bus = DynamicNotesBus(
            self.config.runtime.notes_bus,
            producers=self.streams,
            device=self.device,
            codec=self.model.sidecar.speculation_head,
        )
        for idx, stream in enumerate(self.streams):
            bus.seed_anchor(stream, snapshot0[0, idx])

        # Apply anchor-swap counterfactual if requested.
        if self.counterfactual.mode == "anchor_swap":
            if self.counterfactual.alt_prompt_anchors is None:
                raise ValueError("anchor_swap requires alt_prompt_anchors in config.")
            apply_anchor_swap(
                bus,
                self.counterfactual.alt_prompt_anchors.to(self.device),
                self.streams,
            )

        # Per-stream states start with their addressed prompt. Natural inference
        # supplies K identical prompts through this same path.
        states: Dict[str, StreamState] = {}
        for stream in self.streams:
            stream_ids, stream_mask = stream_prompts[stream]
            states[stream] = StreamState(
                stream=stream,
                input_ids=stream_ids.clone(),
                attention_mask=stream_mask.clone(),
            )

        # -------- One packed K-stream prefill (stream-conditioned) -------- #
        # Prefill owns the first generated-token logits. The final prompt token
        # is already present in its cache and must never be fed a second time.
        pad_token_id = _require_pad_token_id(self.tokenizer)
        packed_prefill = pack_token_rows(
            self.streams,
            {stream: states[stream].input_ids for stream in self.streams},
            pad_token_id=pad_token_id,
        )
        self._set_context(
            self._prepare_frontier_context(
                states,
                bus,
                consumer_block=0,
            )
        )
        out = self.model.trunk_adapter.forward(
            input_ids=packed_prefill.input_ids,
            attention_mask=packed_prefill.valid_mask,
            position_ids=packed_prefill.position_ids,
            cache_position=packed_prefill.cache_position,
            use_cache=True,
            output_hidden_states=False,
        )
        if out.past_key_values is None:
            raise RuntimeError("Frozen trunk dropped the KV cache during packed prefill.")
        frontier = PackedFrontierState(
            streams=self.streams,
            attention_mask=packed_prefill.valid_mask,
            past_key_values=out.past_key_values,
        )
        next_logits = {
            stream: out.logits[index : index + 1, -1, :]
            for index, stream in enumerate(self.streams)
        }

        self._clear_context()

        # -------- Generation loop -------- #
        block_size = self.config.runtime.block_size

        for step in range(max_new_tokens):
            # Snapshot every stream's addressed window at the same pre-append
            # generated count. These contexts remain fixed for the whole round.
            consumer_block = step // block_size
            round_context = self._prepare_frontier_context(
                states,
                bus,
                consumer_block=consumer_block,
            )

            # Sample a full synchronous stream round from already-computed
            # logits. No trunk input is duplicated here.
            for stream in self.streams:
                state = states[stream]
                next_token = int(next_logits[stream].argmax(dim=-1).item())
                piece = self.tokenizer.decode([next_token])
                state.append_token(
                    next_token,
                    token_text=piece,
                )

            # Consume each newly generated token exactly once. Its hidden state
            # is therefore the state used for a boundary write when this round
            # completes a tau-token block.
            packed_step = frontier.prepare_append(
                {stream: states[stream].input_ids[:, -1:] for stream in self.streams},
                pad_token_id=pad_token_id,
            )
            boundary = (step + 1) % block_size == 0
            self._set_context(round_context)
            out = self.model.trunk_adapter.forward(
                input_ids=packed_step.rows.input_ids,
                attention_mask=packed_step.attention_mask,
                past_key_values=frontier.past_key_values,
                position_ids=packed_step.rows.position_ids,
                cache_position=packed_step.rows.cache_position,
                use_cache=True,
                output_hidden_states=boundary,
            )
            if out.past_key_values is None:
                raise RuntimeError(
                    f"Frozen trunk dropped the KV cache during packed decode step {step}."
                )
            frontier.commit(packed_step, past_key_values=out.past_key_values)
            for index, stream in enumerate(self.streams):
                next_logits[stream] = out.logits[index : index + 1, -1, :]

            # Publish only after all K streams have consumed their tau-th token,
            # preventing within-round stream-order leakage.
            if boundary:
                if out.hidden_states is None:
                    raise RuntimeError("Packed boundary decode omitted required hidden states.")
                final_hidden = out.hidden_states[-1]
                for index, stream in enumerate(self.streams):
                    state = states[stream]
                    self._emit_note_snapshot(
                        state,
                        final_hidden[index : index + 1, -1:, :],
                        bus,
                    )
                    state.reset_snapshot_counter()
                completed_block = step // block_size
                next_block = completed_block + 1
                if (
                    stream_block_transitions is not None
                    and next_block * block_size < max_new_tokens
                ):
                    # All K writes are present before any transition is consumed.
                    # Delta=1 therefore exposes block-m writes while the private
                    # observation for block m+1 enters each stream's cache.
                    transition_rows = {
                        stream: stream_block_transitions[stream][next_block]
                        for stream in self.streams
                    }
                    packed_transition = frontier.prepare_append(
                        transition_rows,
                        pad_token_id=pad_token_id,
                    )
                    self._set_context(
                        self._prepare_frontier_context(
                            states,
                            bus,
                            consumer_block=next_block,
                        )
                    )
                    transition_out = self.model.trunk_adapter.forward(
                        input_ids=packed_transition.rows.input_ids,
                        attention_mask=packed_transition.attention_mask,
                        past_key_values=frontier.past_key_values,
                        position_ids=packed_transition.rows.position_ids,
                        cache_position=packed_transition.rows.cache_position,
                        use_cache=True,
                        output_hidden_states=False,
                    )
                    if transition_out.past_key_values is None:
                        raise RuntimeError(
                            "Frozen trunk dropped the KV cache during packed structured "
                            f"transition into block {next_block}."
                        )
                    frontier.commit(
                        packed_transition,
                        past_key_values=transition_out.past_key_values,
                    )
                    for index, stream in enumerate(self.streams):
                        states[stream].append_context_tokens(transition_rows[stream])
                        next_logits[stream] = transition_out.logits[index : index + 1, -1, :]

        self._clear_context()

        return OrchestrationResult(
            text_by_stream={s: states[s].generated_text for s in self.streams},
            tokens_by_stream={s: list(states[s].generated_tokens) for s in self.streams},
            plan_slot_ids=slot_ids,
            planner_logits=planner.logits,
            snapshot0_anchors=snapshot0,
            dynamic_codes_by_stream={
                stream: [
                    update.code_indices
                    for update in bus.all_updates()
                    if update.kind == "dynamic"
                    and update.producer == stream
                    and update.code_indices is not None
                ]
                for stream in self.streams
            },
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _prepare_stream_context(
        self,
        stream: str,
        state: StreamState,
        bus: DynamicNotesBus,
        *,
        consumer_block: int,
    ) -> LayerRuntimeContext:
        window = self.window_builder.build_for_block(
            state,
            bus,
            consumer_block=consumer_block,
        )
        notes_tensor = window.notes
        mask_tensor = window.mask
        if self.counterfactual.mode == "source_swap":
            notes_tensor, mask_tensor = apply_source_swap(
                notes_tensor,
                mask_tensor,
                anchor_mask=window.anchor_mask,
                producer_indices=window.producer_indices,
                consumer_index=self.streams.index(stream),
                donor_notes=self.counterfactual.source_swap_donor,
                donor_mask=self.counterfactual.source_swap_donor_mask,
            )
        if self.counterfactual.mode == "norm_scramble" and notes_tensor.numel() > 0:
            sibling_dynamic = (~window.anchor_mask) & (
                window.producer_indices != self.streams.index(stream)
            )
            notes_tensor = apply_norm_scramble(
                notes_tensor,
                generator=self._rng,
                slot_mask=sibling_dynamic,
            )
        force_gate = apply_gate_ablation() if self.counterfactual.mode == "gate_zero" else None
        state.update_notes_window(notes_tensor, mask_tensor)
        return LayerRuntimeContext(
            stream_ids=(stream,),
            notes=notes_tensor,
            notes_mask=mask_tensor,
            note_producer_ids=window.producer_indices.unsqueeze(0),
            note_kind_ids=(~window.anchor_mask).to(dtype=torch.long).unsqueeze(0),
            note_lags=window.lags.unsqueeze(0),
            snc_force_gate=force_gate,
        )

    def _prepare_frontier_context(
        self,
        states: Mapping[str, StreamState],
        bus: DynamicNotesBus,
        *,
        consumer_block: int,
    ) -> LayerRuntimeContext:
        """Build and batch all K receiver-specific contexts in stream order."""

        contexts = tuple(
            self._prepare_stream_context(
                stream,
                states[stream],
                bus,
                consumer_block=consumer_block,
            )
            for stream in self.streams
        )
        return _pack_layer_contexts(contexts)

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
        pre_quantized = self.model.sidecar.speculation_head.project(block_hidden)
        published_block = (state.generated_count // self.config.runtime.block_size) - 1
        mutation_producer = (self.counterfactual.mutation_producer or self.streams[0]).lower()
        spec = self.model.sidecar.speculation_head.quantize(pre_quantized)
        transmitted_indices = spec.indices[0, -1]
        if (
            self.counterfactual.mode == "bus_mutation"
            and state.stream == mutation_producer
            and published_block == self.counterfactual.mutation_block
        ):
            transmitted_indices = apply_bus_mutation(
                transmitted_indices,
                codes_per_codebook=self.model.sidecar.speculation_head.codes_per_codebook,
                code_offset=self.counterfactual.mutation_code_offset,
            )
        # The bus receives only the (M,) code tuple and decodes it itself.
        code_indices = tuple(int(index) for index in transmitted_indices.tolist())
        snapshot = bus.publish(
            state.stream,
            published_block=published_block,
            stride=state.total_tokens,
            code_indices=code_indices,
            metadata={
                "capacity_bits": spec.capacity_bits,
            },
        )
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


def _pack_layer_contexts(
    contexts: Sequence[LayerRuntimeContext],
) -> LayerRuntimeContext:
    """Concatenate one receiver context per row into one frontier context."""

    if not contexts:
        raise ValueError("Packed frontier context requires at least one stream context.")
    stream_ids: list[str] = []
    for context in contexts:
        if context.stream_ids is None or len(context.stream_ids) != 1:
            raise ValueError(
                "Each context entering a packed frontier must address exactly one stream row."
            )
        stream_ids.append(context.stream_ids[0].lower())
    if len(set(stream_ids)) != len(stream_ids):
        raise ValueError("Packed frontier context stream IDs must be unique.")

    note_presence = tuple(context.notes is not None for context in contexts)
    if any(note_presence) and not all(note_presence):
        raise ValueError("Packed frontier contexts must agree on notes presence.")
    notes = (
        torch.cat([context.notes for context in contexts if context.notes is not None], dim=0)
        if all(note_presence)
        else None
    )

    packed_metadata: dict[str, Optional[torch.Tensor]] = {}
    for name in ("notes_mask", "note_producer_ids", "note_kind_ids", "note_lags"):
        values = tuple(getattr(context, name) for context in contexts)
        presence = tuple(value is not None for value in values)
        if any(presence) and not all(presence):
            raise ValueError(f"Packed frontier contexts must agree on {name} presence.")
        packed_metadata[name] = (
            torch.cat([value for value in values if value is not None], dim=0)
            if all(presence)
            else None
        )

    gate_values = tuple(context.snc_force_gate for context in contexts)
    if all(value is None for value in gate_values):
        force_gate: Optional[object] = None
    elif any(value is None for value in gate_values):
        raise ValueError("Packed frontier contexts must agree on SNC gate override presence.")
    else:
        device = notes.device if notes is not None else torch.device("cpu")
        normalized_gates: list[bool] = []
        for value in gate_values:
            override = torch.as_tensor(value, device=device)
            if override.numel() != 1:
                raise ValueError("Per-stream SNC gate overrides must be scalar before packing.")
            normalized_gates.append(bool(override.item()))
        force_gate = torch.tensor(normalized_gates, dtype=torch.bool, device=device)

    return LayerRuntimeContext(
        stream_ids=tuple(stream_ids),
        notes=notes,
        notes_mask=packed_metadata["notes_mask"],
        note_producer_ids=packed_metadata["note_producer_ids"],
        note_kind_ids=packed_metadata["note_kind_ids"],
        note_lags=packed_metadata["note_lags"],
        snc_force_gate=force_gate,
    )


def _require_pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise RuntimeError("Packed PDT requires a tokenizer with a non-negative pad_token_id.")
    return pad_token_id


def _validate_structured_transitions(
    rows: Sequence[Sequence[int]],
    *,
    stream: str,
    block_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Validate canonical row-0-empty temporal suffixes and move them to device."""

    if isinstance(rows, (str, bytes)) or len(rows) != block_count:
        actual = len(rows) if not isinstance(rows, (str, bytes)) else "non-sequence"
        raise ValueError(
            f"{stream} block transitions must have exactly {block_count} rows; got {actual}."
        )
    prepared: list[torch.Tensor] = []
    for block_idx, row in enumerate(rows):
        if isinstance(row, (str, bytes)):
            raise ValueError(f"{stream} transition row {block_idx} must be integer token IDs.")
        values = list(row)
        if block_idx == 0 and values:
            raise ValueError(f"{stream} transition row 0 must be empty; block 0 is in prefill.")
        if block_idx > 0 and not values:
            raise ValueError(f"{stream} transition row {block_idx} must be non-empty.")
        if any(type(token_id) is not int or token_id < 0 for token_id in values):
            raise ValueError(
                f"{stream} transition row {block_idx} must contain non-negative integer IDs."
            )
        prepared.append(torch.tensor([values], dtype=torch.long, device=device))
    return tuple(prepared)


def _tokenize_user_prompt(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render one user turn with the frozen Instruct checkpoint's template."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(
            "The configured Instruct tokenizer has no chat template; refusing "
            "to encode a raw prompt with checkpoint-mismatched semantics."
        )
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    if not isinstance(encoded, Mapping):
        raise TypeError("apply_chat_template(return_dict=True) must return a mapping.")
    input_ids = encoded.get("input_ids")
    attention_mask = encoded.get("attention_mask")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
        raise RuntimeError(
            "Chat-template tokenization must return input_ids and attention_mask tensors."
        )
    _validate_tokenized_prompt(input_ids, attention_mask, name="chat-template prompt")
    return input_ids.to(device), attention_mask.to(device)


def _validate_tokenized_prompt(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    name: str,
) -> None:
    if input_ids.dim() != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError(f"{name} ids/mask must have matching [batch, tokens] shape.")
    if input_ids.size(0) != 1 or input_ids.size(1) == 0:
        raise ValueError(f"{name} must contain exactly one non-empty prompt.")
    if not attention_mask.to(dtype=torch.bool).any():
        raise ValueError(f"{name} attention mask contains no active tokens.")
    if not bool(attention_mask.to(dtype=torch.bool).all()):
        raise ValueError(f"{name} must be an unpadded row; packed PDT owns left padding centrally.")
