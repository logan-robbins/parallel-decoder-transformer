"""Multi-stream inference orchestrator.

Responsibilities:

- Run the planner once and retain one persistent read-only outline per lane.
- Advance all stream frontier tokens through one shared-lower and grouped
  physical-upper call per round.
- Assemble either the addressed sibling-note window or the parameter-matched
  receiver-owned history, according to the checkpoint's coordination source.
- Thread ``LayerRuntimeContext`` into every physical branch layer so its
  hard-routed Plan-KV and delayed-memory reads execute correctly.
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

from pdt.baselines.self_only import build_self_only_memory
from pdt.config.schemas import PDTConfig
from pdt.model import PDTModel
from pdt.prompts import planner_user_text, stream_user_text
from pdt.runtime.counterfactuals import (
    CounterfactualConfig,
    apply_bus_mutation,
    apply_gate_ablation,
    apply_norm_scramble,
    apply_plan_intervention,
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
    plan_nodes: torch.Tensor  # (1, K, N, planner_width)
    plan_node_mask: torch.Tensor  # (1, K, N)
    planner_node_validity_logits: torch.Tensor  # (1, K, N)
    presentation_order_logits: torch.Tensor  # (1, K)
    plan_memory: torch.Tensor  # (1, K, N, notes_dim)
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
        self.counterfactual = counterfactual or CounterfactualConfig(mode="none")
        if (
            config.instrumentation.coordination_source == "self_only"
            and self.counterfactual.mode
            in {"bus_mutation", "norm_scramble", "source_swap"}
        ):
            raise ValueError(
                f"Counterfactual {self.counterfactual.mode!r} requires a bus checkpoint; "
                "the self-only runtime has no sibling-note channel."
            )
        frozen_trunk_parameters = model.trunk_adapter.frozen_parameters()
        if not frozen_trunk_parameters:
            raise RuntimeError("Packed inference requires a non-empty frozen trunk.")
        trunk_parameter = frozen_trunk_parameters[0]
        if not trunk_parameter.is_floating_point():
            raise TypeError("Frozen trunk parameters must use a floating-point dtype.")
        self.device = trunk_parameter.device
        self.trunk_dtype = trunk_parameter.dtype
        sidecar_parameter = next(model.sidecar.parameters(), None)
        if sidecar_parameter is None or sidecar_parameter.device != self.device:
            raise RuntimeError(
                "Packed inference requires the trunk and sidecar on the same device."
            )

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
            history_blocks=config.runtime.notes_bus.history_blocks,
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
        plan_nodes_override: Optional[torch.Tensor] = None,
        plan_mask_override: Optional[torch.Tensor] = None,
    ) -> OrchestrationResult:
        """Run the full multi-stream decoding loop for one prompt.

        Args:
            prompt: User input text.
            max_new_tokens: Per-stream token budget.
            plan_nodes_override: Optional oracle plan with shape
                ``(1, K, N, planner_width)``.
            plan_mask_override: Required validity mask for an oracle plan.
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
            plan_nodes_override=plan_nodes_override,
            plan_mask_override=plan_mask_override,
        )

    @torch.no_grad()
    def generate_structured(
        self,
        shared_prompt: str,
        stream_local_prompts: Mapping[str, str],
        stream_block_transition_ids: Mapping[str, Sequence[Sequence[int]]],
        *,
        max_new_tokens: int = 128,
        plan_nodes_override: Optional[torch.Tensor] = None,
        plan_mask_override: Optional[torch.Tensor] = None,
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
            plan_nodes_override=plan_nodes_override,
            plan_mask_override=plan_mask_override,
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
        plan_nodes_override: Optional[torch.Tensor],
        plan_mask_override: Optional[torch.Tensor],
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
        prompt_hidden = self.model.encode_planner_prompt(
            prompt_ids,
            prompt_mask,
        )
        planner = self.model.sidecar.planner_head(prompt_hidden, attention_mask=prompt_mask)
        plan_nodes, plan_mask = _resolve_plan(
            planner.nodes,
            planner.node_validity_logits,
            plan_nodes_override=plan_nodes_override,
            plan_mask_override=plan_mask_override,
            device=self.device,
        )
        presentation_order_logits = planner.presentation_order_logits
        plan_intervention = self.counterfactual.mode
        if plan_intervention == "plan_swap":
            plan_nodes = apply_plan_intervention(
                plan_nodes,
                mode="plan_swap",
                alternate=self.counterfactual.alt_prompt_plan_nodes,
                generator=self._rng,
            )
        elif plan_intervention == "lane_swap":
            plan_nodes = apply_plan_intervention(
                plan_nodes,
                mode="lane_swap",
                lane_pair=self.counterfactual.plan_swap_lanes,
            )
            plan_mask = apply_plan_intervention(
                plan_mask,
                mode="lane_swap",
                lane_pair=self.counterfactual.plan_swap_lanes,
            )
            presentation_order_logits = apply_plan_intervention(
                presentation_order_logits.unsqueeze(-1),
                mode="lane_swap",
                lane_pair=self.counterfactual.plan_swap_lanes,
            ).squeeze(-1)
        elif plan_intervention == "plan_zero":
            plan_nodes = apply_plan_intervention(
                plan_nodes,
                mode="plan_zero",
                generator=self._rng,
            )
        elif plan_intervention == "random_plan":
            plan_nodes = apply_plan_intervention(
                plan_nodes,
                mode="random_plan",
                generator=self._rng,
            )
        plan_memory = self.model.sidecar.plan_memory_proj(plan_nodes, plan_mask)

        # -------- Initialize the selected coordination state -------- #
        bus = (
            DynamicNotesBus(
                self.config.runtime.notes_bus,
                producers=self.streams,
                device=self.device,
                codec=self.model.sidecar.speculation_head,
            )
            if self.config.instrumentation.coordination_source == "bus"
            else None
        )
        self_only_states: list[torch.Tensor] = []
        self_only_validity: list[torch.Tensor] = []
        self_only_positions: list[torch.Tensor] = []

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
                self_only_states=self_only_states,
                self_only_validity=self_only_validity,
                self_only_positions=self_only_positions,
                query_positions=packed_prefill.position_ids,
                plan_nodes=plan_nodes,
                plan_memory=plan_memory,
                plan_mask=plan_mask,
            )
        )
        out = self.model.forward_frontier(
            input_ids=packed_prefill.input_ids,
            attention_mask=packed_prefill.valid_mask,
            position_ids=packed_prefill.position_ids,
            cache_position=packed_prefill.cache_position,
            use_cache=True,
            output_hidden_states=False,
            logits_to_keep=1,
        )
        if out.past_key_values is None:
            raise RuntimeError("Physical frontier dropped its cache during packed prefill.")
        if out.logits is None:
            raise RuntimeError("Physical frontier omitted logits during packed prefill.")
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
        stop_on_eos = stream_block_transitions is None
        eos_token_id = self.tokenizer.eos_token_id
        if stop_on_eos and (type(eos_token_id) is not int or eos_token_id < 0):
            raise ValueError("Natural generation requires one non-negative EOS token ID.")
        finished: set[str] = set()

        for step in range(max_new_tokens):
            active_streams = tuple(
                stream for stream in self.streams if stream not in finished
            )
            if not active_streams:
                break
            consumer_block = step // block_size

            # Sample a full synchronous stream round from already-computed
            # logits. No trunk input is duplicated here.
            for stream in active_streams:
                state = states[stream]
                next_token = int(next_logits[stream].argmax(dim=-1).item())
                state.append_token(
                    next_token,
                )
                if stop_on_eos and next_token == eos_token_id:
                    finished.add(stream)

            # Consume each newly generated token exactly once. Its hidden state
            # is therefore the state used for a boundary write when this round
            # completes a tau-token block.
            append_tokens = {
                stream: (
                    states[stream].input_ids[:, -1:]
                    if stream in active_streams
                    else torch.full(
                        (1, 1),
                        pad_token_id,
                        dtype=torch.long,
                        device=self.device,
                    )
                )
                for stream in self.streams
            }
            packed_step = frontier.prepare_append(
                append_tokens,
                pad_token_id=pad_token_id,
                active_streams=active_streams,
            )
            # Every row sees the same pre-round coordination frontier. For the
            # self-only control, query positions are the exact positions of
            # the packed tokens being consumed.
            round_context = self._prepare_frontier_context(
                states,
                bus,
                consumer_block=consumer_block,
                self_only_states=self_only_states,
                self_only_validity=self_only_validity,
                self_only_positions=self_only_positions,
                query_positions=packed_step.rows.position_ids,
                plan_nodes=plan_nodes,
                plan_memory=plan_memory,
                plan_mask=plan_mask,
            )
            boundary = (step + 1) % block_size == 0
            self._set_context(round_context)
            out = self.model.forward_frontier(
                input_ids=packed_step.rows.input_ids,
                attention_mask=packed_step.attention_mask,
                past_key_values=frontier.past_key_values,
                position_ids=packed_step.rows.position_ids,
                cache_position=packed_step.rows.cache_position,
                use_cache=True,
                output_hidden_states=boundary,
                logits_to_keep=1,
            )
            if out.past_key_values is None:
                raise RuntimeError(
                    f"Physical frontier dropped its cache during packed decode step {step}."
                )
            if out.logits is None:
                raise RuntimeError(
                    f"Physical frontier omitted logits during packed decode step {step}."
                )
            frontier.commit(packed_step, past_key_values=out.past_key_values)
            for index, stream in enumerate(self.streams):
                if stream in active_streams:
                    next_logits[stream] = out.logits[index : index + 1, -1, :]

            # Publish only after all K streams have consumed their tau-th token,
            # preventing within-round stream-order leakage.
            if boundary:
                if out.hidden_states is None:
                    raise RuntimeError("Packed boundary decode omitted required hidden states.")
                final_hidden = out.hidden_states[-1]
                if bus is None:
                    complete = torch.tensor(
                        [
                            states[stream].tokens_since_snapshot == block_size
                            for stream in self.streams
                        ],
                        dtype=torch.bool,
                        device=self.device,
                    )
                    block_end_hidden = final_hidden[:, -1, :]
                    block_end_hidden = block_end_hidden * complete.to(
                        dtype=block_end_hidden.dtype
                    ).unsqueeze(-1)
                    block_end_positions = packed_step.rows.position_ids[:, -1].masked_fill(
                        ~complete,
                        -1,
                    )
                    self_only_states.append(block_end_hidden.unsqueeze(0))
                    self_only_validity.append(complete.unsqueeze(0))
                    self_only_positions.append(block_end_positions.unsqueeze(0))
                    for stream, is_complete in zip(
                        self.streams,
                        complete.tolist(),
                        strict=True,
                    ):
                        if is_complete:
                            states[stream].reset_snapshot_counter()
                else:
                    for index, stream in enumerate(self.streams):
                        state = states[stream]
                        if state.tokens_since_snapshot == block_size:
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
                            self_only_states=self_only_states,
                            self_only_validity=self_only_validity,
                            self_only_positions=self_only_positions,
                            query_positions=packed_transition.rows.position_ids,
                            plan_nodes=plan_nodes,
                            plan_memory=plan_memory,
                            plan_mask=plan_mask,
                        )
                    )
                    transition_out = self.model.forward_frontier(
                        input_ids=packed_transition.rows.input_ids,
                        attention_mask=packed_transition.attention_mask,
                        past_key_values=frontier.past_key_values,
                        position_ids=packed_transition.rows.position_ids,
                        cache_position=packed_transition.rows.cache_position,
                        use_cache=True,
                        output_hidden_states=False,
                        logits_to_keep=1,
                    )
                    if transition_out.past_key_values is None:
                        raise RuntimeError(
                            "Physical frontier dropped its cache during packed structured "
                            f"transition into block {next_block}."
                        )
                    if transition_out.logits is None:
                        raise RuntimeError(
                            "Physical frontier omitted logits during packed structured "
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
            text_by_stream={
                stream: self.tokenizer.decode(
                    states[stream].generated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for stream in self.streams
            },
            tokens_by_stream={s: list(states[s].generated_tokens) for s in self.streams},
            plan_nodes=plan_nodes,
            plan_node_mask=plan_mask,
            planner_node_validity_logits=planner.node_validity_logits,
            presentation_order_logits=presentation_order_logits,
            plan_memory=plan_memory,
            dynamic_codes_by_stream={
                stream: [
                    update.code_indices
                    for update in (() if bus is None else bus.all_updates())
                    if update.producer == stream and update.code_indices is not None
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
        plan_nodes: torch.Tensor,
        plan_memory: torch.Tensor,
        plan_mask: torch.Tensor,
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
                producer_indices=window.producer_indices,
                consumer_index=self.streams.index(stream),
                donor_notes=self.counterfactual.source_swap_donor,
                donor_mask=self.counterfactual.source_swap_donor_mask,
            )
        if self.counterfactual.mode == "norm_scramble" and notes_tensor.numel() > 0:
            sibling_dynamic = window.producer_indices != self.streams.index(stream)
            notes_tensor = apply_norm_scramble(
                notes_tensor,
                generator=self._rng,
                slot_mask=sibling_dynamic,
            )
        force_gate = apply_gate_ablation() if self.counterfactual.mode == "gate_zero" else None
        state.update_notes_window(notes_tensor, mask_tensor)
        return LayerRuntimeContext(
            stream_ids=(stream,),
            plan_nodes=plan_nodes,
            plan_mask=plan_mask,
            plan_memory=plan_memory,
            plan_producer_ids=torch.full(
                plan_mask.shape,
                self.streams.index(stream),
                dtype=torch.long,
                device=plan_mask.device,
            ),
            notes=notes_tensor,
            notes_mask=mask_tensor,
            note_producer_ids=window.producer_indices.unsqueeze(0),
            note_kind_ids=torch.ones_like(window.producer_indices).unsqueeze(0),
            note_lags=window.lags.unsqueeze(0),
            snc_force_gate=force_gate,
        )

    def _prepare_frontier_context(
        self,
        states: Mapping[str, StreamState],
        bus: Optional[DynamicNotesBus],
        *,
        consumer_block: int,
        self_only_states: list[torch.Tensor],
        self_only_validity: list[torch.Tensor],
        self_only_positions: list[torch.Tensor],
        query_positions: torch.Tensor,
        plan_nodes: torch.Tensor,
        plan_memory: torch.Tensor,
        plan_mask: torch.Tensor,
    ) -> LayerRuntimeContext:
        """Build and batch all K receiver-specific contexts in stream order."""

        if self.config.instrumentation.coordination_source == "self_only":
            if bus is not None:
                raise RuntimeError("Self-only inference cannot own a dynamic notes bus.")
            batch, lanes, nodes, _ = plan_nodes.shape
            if batch != 1 or lanes != len(self.streams):
                raise ValueError("Self-only runtime requires one complete physical frontier.")
            if query_positions.size(0) != lanes:
                raise ValueError(
                    "Self-only query-position rows must match the physical lane count."
                )
            lane_ids = torch.arange(
                lanes,
                device=self.device,
                dtype=torch.long,
            ).view(lanes, 1)
            return LayerRuntimeContext(
                stream_ids=self.streams,
                plan_nodes=plan_nodes.reshape(lanes, nodes, -1),
                plan_mask=plan_mask.reshape(lanes, nodes),
                plan_memory=plan_memory.reshape(lanes, nodes, -1),
                plan_producer_ids=lane_ids.expand(lanes, nodes),
                self_only_memory=build_self_only_memory(
                    self_only_states,
                    self_only_validity,
                    self_only_positions,
                    consumer_block=consumer_block,
                    lanes=lanes,
                    history_blocks=self.config.runtime.notes_bus.history_blocks,
                    hidden_size=self.config.sidecar.snc.hidden_size,
                    streams=self.streams,
                    device=self.device,
                    dtype=self.trunk_dtype,
                ),
                self_only_query_positions=query_positions,
                snc_force_gate=(
                    apply_gate_ablation()
                    if self.counterfactual.mode == "gate_zero"
                    else None
                ),
            )
        if bus is None:
            raise RuntimeError("Bus inference requires a dynamic notes bus.")
        if self_only_states or self_only_validity or self_only_positions:
            raise RuntimeError("Bus inference cannot retain self-only hidden history.")
        contexts = tuple(
            self._prepare_stream_context(
                stream,
                states[stream],
                bus,
                consumer_block=consumer_block,
                plan_nodes=plan_nodes[:, index],
                plan_memory=plan_memory[:, index],
                plan_mask=plan_mask[:, index],
            )
            for index, stream in enumerate(self.streams)
        )
        return _pack_layer_contexts(contexts)

    def _set_context(self, context: LayerRuntimeContext) -> None:
        self.model.set_runtime_context(context)

    def _clear_context(self) -> None:
        self.model.set_runtime_context(None)

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


def _resolve_plan(
    predicted_nodes: torch.Tensor,
    validity_logits: torch.Tensor,
    *,
    plan_nodes_override: Optional[torch.Tensor],
    plan_mask_override: Optional[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select either the learned plan or an explicit oracle plan."""

    if predicted_nodes.dim() != 4 or validity_logits.shape != predicted_nodes.shape[:-1]:
        raise ValueError(
            "Planner outputs must have shapes [B, K, N, P] and [B, K, N]."
        )
    if (plan_nodes_override is None) != (plan_mask_override is None):
        raise ValueError(
            "plan_nodes_override and plan_mask_override must be supplied together."
        )
    if plan_nodes_override is None:
        nodes = predicted_nodes
        mask = validity_logits >= 0
    else:
        assert plan_mask_override is not None
        nodes = plan_nodes_override.to(
            device=device,
            dtype=predicted_nodes.dtype,
        )
        mask = plan_mask_override.to(device=device, dtype=torch.bool)
        if nodes.shape != predicted_nodes.shape:
            raise ValueError(
                "plan_nodes_override must match planner output shape; "
                f"expected {tuple(predicted_nodes.shape)}, got {tuple(nodes.shape)}."
            )
        if mask.shape != validity_logits.shape:
            raise ValueError(
                "plan_mask_override must match planner validity shape; "
                f"expected {tuple(validity_logits.shape)}, got {tuple(mask.shape)}."
            )
    if not bool(torch.isfinite(nodes).all()):
        raise ValueError("Plan nodes must be finite.")
    if bool((~mask.any(dim=-1)).any()):
        raise ValueError("Every physical lane requires at least one valid plan node.")
    return nodes, mask


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

    packed_plan: dict[str, Optional[torch.Tensor]] = {}
    for name in ("plan_nodes", "plan_mask", "plan_memory", "plan_producer_ids"):
        values = tuple(getattr(context, name) for context in contexts)
        presence = tuple(value is not None for value in values)
        if any(presence) and not all(presence):
            raise ValueError(f"Packed frontier contexts must agree on {name} presence.")
        packed_plan[name] = (
            torch.cat([value for value in values if value is not None], dim=0)
            if all(presence)
            else None
        )

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
        plan_nodes=packed_plan["plan_nodes"],
        plan_mask=packed_plan["plan_mask"],
        plan_memory=packed_plan["plan_memory"],
        plan_producer_ids=packed_plan["plan_producer_ids"],
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
