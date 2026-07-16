"""Single-GPU / DDP trainer for PDT.

Training loop responsibilities:

1. Per-step curriculum tick (``CurriculumController.on_step``): flip
   requires_grad per stage.
2. Build one batch via ``PDTCollator``.
3. Forward: prompt encode (trunk), VQ planner, planner-seeded bus notes,
   differentiable block rollout with SNC windows, speculation writes.
4. Compute all loss terms; stage-mask via ``compute_pdt_losses``.
5. Backward, clip, step, LR schedule.
6. Periodic eval: codebook diagnostics and curriculum telemetry.

Training mirrors the inference block semantics closely enough for gradients
to flow from receiver LM loss through SNC into visible sibling notes and the
speculation writer that produced them.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from pdt.baselines.self_only import SelfOnlyMemory
from pdt.checkpoint import (
    CheckpointMetadata,
    CheckpointMismatchError,
    resume_checkpoint,
    save_checkpoint,
)
from pdt.config.schemas import PDTConfig
from pdt.diagnostics.architecture import architecture_telemetry
from pdt.diagnostics.codebook import CodebookDiagnostics
from pdt.evaluation.paired_causal import CausalDocumentEvaluator, PairedCausalEvaluator
from pdt.model import PDTModel
from pdt.runtime.counterfactuals import apply_bus_mutation, apply_norm_scramble
from pdt.training.curriculum import CurriculumController
from pdt.training.dataset import PDTCollator, PDTDependencyDataset, SampleBatch
from pdt.training.losses import compute_pdt_losses
from pdt.trunk.instrumentation import LayerRuntimeContext


LOGGER = logging.getLogger("pdt.training.trainer")


__all__ = ["PDTTrainer"]


_RolloutMode = Literal["normal", "gate_zero", "norm_scramble", "bus_mutation"]


@dataclass(frozen=True, slots=True)
class _RolloutIntervention:
    mode: _RolloutMode = "normal"
    seed: int = 0
    mutation_producer: str | None = None
    mutation_block: int = 0
    mutation_code_offset: int = 1


@dataclass(slots=True)
class _StudentRollout:
    lm_logits: torch.Tensor
    lm_labels: torch.Tensor
    lm_label_mask: torch.Tensor
    dependency_mask: torch.Tensor
    nondependency_mask: torch.Tensor
    classifier_hidden: torch.Tensor
    planner: Any
    plan_snapshot: torch.Tensor
    dynamic_vq_commitment_loss: torch.Tensor
    dynamic_vq_codebook_loss: torch.Tensor
    dynamic_note_indices: torch.Tensor
    dynamic_assignment_logits: torch.Tensor


class PDTTrainer:
    def __init__(
        self,
        model: PDTModel,
        config: PDTConfig,
        *,
        telemetry_dir: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.config = config
        if (
            model.config.instrumentation.coordination_source
            != config.instrumentation.coordination_source
        ):
            raise ValueError(
                "Trainer/model coordination source mismatch: "
                f"model={model.config.instrumentation.coordination_source!r}, "
                f"trainer={config.instrumentation.coordination_source!r}."
            )
        self.telemetry_dir = Path(telemetry_dir or config.training.telemetry_dir).resolve()
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._resolve_device()
        # Move both the \u03c6 tree (sidecar + instrumented layers) AND the frozen
        # trunk model to the target device. The trunk is held by composition
        # on a non-Module adapter, so PDTModel.to(device) would not reach it.
        self.model.to(self.device)
        trunk_model: torch.nn.Module = self.model.trunk_adapter.model
        trunk_model.to(self.device)
        _require_cache_compatible_trunk(self.model.trunk_adapter)
        pad_token_id = self.model.trunk_adapter.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("The frozen trunk tokenizer must define pad_token_id.")
        self.pad_token_id = int(pad_token_id)

        self.curriculum = CurriculumController(model, config)
        self.codebook = CodebookDiagnostics(
            vocab_size=config.sidecar.plan_vocab_size,
            num_slots=config.sidecar.planner_head.num_slots,
        )
        self.dynamic_codebook = CodebookDiagnostics(
            vocab_size=config.sidecar.speculation_head.codes_per_codebook,
            num_slots=config.sidecar.speculation_head.num_codebooks,
        )

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        # Optimizer-update count. Gradient-accumulation microbatches do not
        # advance this counter or the curriculum schedule.
        self.global_step = 0
        self._train_loader: Optional[DataLoader] = None
        self._eval_loader: Optional[DataLoader] = None

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def _resolve_device(self) -> torch.device:
        if self.config.training.device:
            return torch.device(self.config.training.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        opt_cfg = self.config.training.optimizer
        params = [p for p in self.model.all_trainable_parameters() if p.requires_grad]
        return AdamW(
            params,
            lr=opt_cfg.learning_rate,
            weight_decay=opt_cfg.weight_decay,
            betas=(0.9, 0.95),
        )

    def _build_scheduler(self):
        opt_cfg = self.config.training.optimizer
        max_steps = self.config.training.max_steps
        warmup = opt_cfg.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return max(step, 1) / max(warmup, 1)
            if opt_cfg.lr_scheduler == "constant":
                return 1.0
            if opt_cfg.lr_scheduler == "linear":
                progress = (step - warmup) / max(1, max_steps - warmup)
                return max(0.0, 1.0 - progress)
            # cosine
            progress = (step - warmup) / max(1, max_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        return LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _build_collator(self) -> PDTCollator:
        return PDTCollator(
            pad_token_id=self.pad_token_id,
            num_streams=self.config.sidecar.num_streams,
            max_planner_prompt_length=self.config.training.max_planner_prompt_length,
            max_stream_prompt_length=self.config.training.max_stream_prompt_length,
            max_block_transition_length=(self.config.training.max_block_transition_length),
            max_teacher_prompt_length=self.config.training.max_teacher_prompt_length,
            max_blocks=self.config.training.max_blocks,
            max_block_length=self.config.runtime.block_size,
        )

    def _build_dataloader(self, path: str, shuffle: bool) -> DataLoader:
        dataset = PDTDependencyDataset(
            path,
            num_streams=self.config.sidecar.num_streams,
            expected_tokenizer=self.config.trunk.base_model,
            expected_tokenizer_revision=self.config.trunk.revision,
        )
        collator = self._build_collator()
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            collate_fn=collator,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
            drop_last=shuffle,
        )

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #

    def resume_from_checkpoint(self, path: Path) -> CheckpointMetadata:
        """Strictly restore phi, optimizer, scheduler, step, and freeze policy."""

        metadata = resume_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.scheduler,
        )
        expected_stage = self.curriculum.determine_stage(metadata.global_step)
        if metadata.stage != expected_stage:
            raise CheckpointMismatchError(
                "Checkpoint curriculum stage does not match its global step: "
                f"checkpoint stage={metadata.stage}, step={metadata.global_step}, "
                f"configured stage={expected_stage}."
            )
        self.global_step = metadata.global_step
        restored_stage = self.curriculum.on_step(self.global_step)
        if restored_stage != metadata.stage:
            raise RuntimeError(
                "Curriculum restoration returned a stage different from the "
                "validated checkpoint metadata."
            )
        LOGGER.info(
            "Resumed checkpoint %s at step=%d stage=%d",
            path,
            metadata.global_step,
            metadata.stage,
        )
        return metadata

    def train(self) -> None:
        self._train_loader = self._build_dataloader(self.config.training.dataset_path, shuffle=True)
        self._eval_loader = self._build_dataloader(
            self.config.training.eval_dataset_path, shuffle=False
        )
        LOGGER.info(
            "Beginning training: %d train batches, device=%s",
            len(self._train_loader),
            self.device,
        )
        self.model.train()
        # Qwen disables use_cache when gradient checkpointing and train mode
        # coincide. The frozen trunk lives outside PDTModel's registered module
        # tree, and this assertion makes that cache-compatible boundary explicit.
        self.model.trunk_adapter.model.eval()
        _require_cache_compatible_trunk(self.model.trunk_adapter)
        t_start = time.time()
        accum = 0
        train_iter = _infinite(self._train_loader)
        while self.global_step < self.config.training.max_steps:
            stage = self.curriculum.on_step(self.global_step)
            batch = next(train_iter)
            losses_dict = self._train_step(batch, stage=stage)
            accum += 1
            if accum >= self.config.training.grad_accumulation:
                self._optimizer_step()
                accum = 0
                self.global_step += 1
                # A checkpoint at an exact stage boundary must carry the
                # policy that will govern the next microbatch, not the policy
                # that governed the optimizer step just completed.
                self.curriculum.on_step(self.global_step)
                if self.global_step % self.config.training.log_interval == 0:
                    elapsed = time.time() - t_start
                    LOGGER.info(
                        "step=%d stage=%d loss=%s | %.2fs elapsed",
                        self.global_step,
                        stage,
                        {
                            k: round(v, 4)
                            for k, v in losses_dict.items()
                            if isinstance(v, (int, float))
                        },
                        elapsed,
                    )
                if self.global_step % self.config.training.save_every == 0:
                    self._save_checkpoint()
                if self.global_step % self.config.training.eval_interval == 0:
                    self._eval()

        # Final artifacts without duplicating a periodic event at the same step.
        if self.global_step % self.config.training.save_every != 0:
            self._save_checkpoint()
        if self.global_step % self.config.training.eval_interval != 0:
            self._eval()

    def optimizer_probe(self) -> dict[str, object]:
        """Execute two real CUDA updates and prove every active phi group has gradients.

        Two updates are required because the canonical zero-initialized output
        projections intentionally block upstream q/k/v and adapter-downstream
        gradients on the first backward pass. The second pass audits the opened
        graph, writes machine-readable telemetry, and saves a resumable checkpoint.
        """

        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The optimizer probe requires a visible CUDA device.")
        if self.config.training.grad_accumulation != 1:
            raise ValueError(
                "The optimizer probe requires training.grad_accumulation=1 so each "
                "audited backward pass is exactly one optimizer update."
            )
        if self.global_step != 0:
            raise RuntimeError("The optimizer probe must start from a fresh step-0 model.")

        loader = self._build_dataloader(self.config.training.dataset_path, shuffle=False)
        if len(loader) == 0:
            raise ValueError("The optimizer probe dataset is empty.")
        iterator = _infinite(loader)
        self.model.train()
        self.model.trunk_adapter.model.eval()
        _require_cache_compatible_trunk(self.model.trunk_adapter)
        torch.cuda.reset_peak_memory_stats(self.device)

        losses_by_step: list[dict[str, float]] = []
        gradient_report: dict[str, object] | None = None
        started = time.perf_counter()
        for probe_step in range(2):
            stage = self.curriculum.on_step(self.global_step)
            batch = next(iterator)
            losses_by_step.append(self._train_step(batch, stage=stage))
            if probe_step == 1:
                gradient_report = _active_phi_gradient_report(self.model)
            self._optimizer_step()
            self.global_step += 1
            self.curriculum.on_step(self.global_step)

        if gradient_report is None:
            raise RuntimeError("The second optimizer-probe gradient audit did not execute.")
        elapsed = time.perf_counter() - started
        metrics: dict[str, object] = {
            "coordination_source": self.config.instrumentation.coordination_source,
            "optimizer_steps": self.global_step,
            "losses": losses_by_step,
            "active_phi_gradients_after_first_update": gradient_report,
            "elapsed_seconds": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
            "device": torch.cuda.get_device_name(self.device),
        }
        destination = self.telemetry_dir / "optimizer_probe.json"
        destination.write_text(json.dumps(metrics, indent=2))
        self._save_checkpoint()
        LOGGER.info("CUDA optimizer probe passed: %s", metrics)
        return metrics

    def _train_step(self, batch: SampleBatch, *, stage: int) -> Dict[str, float]:
        batch = _to_device(batch, self.device)
        rollout = self._student_rollout(batch)
        teacher_logits = self._functional_teacher_logits(batch)
        B = batch.target_block_ids.size(0)
        K = batch.target_block_ids.size(1)
        M = batch.target_block_ids.size(2)
        stream_classifier_logits = self.model.sidecar.stream_classifier(rollout.classifier_hidden)
        stream_targets = torch.arange(K, device=self.device).repeat(M).repeat_interleave(B)

        losses = compute_pdt_losses(
            stage=stage,
            weights=self.curriculum.active_loss_weights(stage),
            lm_logits=rollout.lm_logits,
            lm_labels=rollout.lm_labels,
            lm_label_mask=rollout.lm_label_mask,
            dependency_mask=rollout.dependency_mask,
            nondependency_mask=rollout.nondependency_mask,
            lm_teacher_logits=teacher_logits,
            kd_temperature_lm=self.config.training.kd_temperature_lm,
            planner_vq_commitment_loss=rollout.planner.commitment_loss,
            planner_vq_codebook_loss=rollout.planner.codebook_loss,
            dynamic_vq_commitment_loss=rollout.dynamic_vq_commitment_loss,
            dynamic_vq_codebook_loss=rollout.dynamic_vq_codebook_loss,
            planner_logits=rollout.planner.logits,
            dynamic_vq_logits=rollout.dynamic_assignment_logits,
            stream_logits=stream_classifier_logits,
            stream_targets=stream_targets,
        )

        loss = losses.total / self.config.training.grad_accumulation
        loss.backward()

        return losses.to_dict()

    def _student_rollout(
        self,
        batch: SampleBatch,
        *,
        intervention: _RolloutIntervention | None = None,
    ) -> _StudentRollout:
        """Run the canonical cached student path under one intervention."""

        _require_cache_compatible_trunk(self.model.trunk_adapter)
        intervention = intervention or _RolloutIntervention()
        coordination_source = self.config.instrumentation.coordination_source
        _validate_rollout_intervention(
            intervention,
            streams=self.config.runtime.streams,
            codes_per_codebook=self.config.sidecar.speculation_head.codes_per_codebook,
            coordination_source=coordination_source,
        )
        B = batch.planner_prompt_ids.size(0)
        K = batch.stream_prompt_ids.size(1)
        M = batch.target_block_ids.size(2)
        stream_names = self.config.runtime.streams
        _validate_fixed_tau_blocks(batch, tau=self.config.runtime.block_size)
        _validate_block_transitions(batch)
        if B != 1:
            raise ValueError(
                "Canonical differentiable cached rollout requires training.batch_size=1; "
                f"got batch size {B}. Use gradient accumulation for larger effective batches."
            )
        if K != len(stream_names):
            raise ValueError(
                f"Rollout received K={K} streams but runtime config has {len(stream_names)}."
            )

        scramble_generator: torch.Generator | None = None
        if intervention.mode == "norm_scramble":
            scramble_generator = torch.Generator(device="cpu")
            scramble_generator.manual_seed(intervention.seed)

        _clear_runtime_contexts(self.model.instrumented_layers)
        prompt_out = self.model.trunk_adapter.forward(
            input_ids=batch.planner_prompt_ids,
            attention_mask=batch.planner_prompt_attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        if prompt_out.hidden_states is None:
            raise RuntimeError("Planner prompt forward must return hidden states.")
        prompt_hidden = prompt_out.hidden_states[-1]
        planner = self.model.sidecar.planner_head(
            prompt_hidden,
            attention_mask=batch.planner_prompt_attention_mask.to(prompt_hidden.dtype),
        )
        ownership = _round_robin_ownership(
            batch_size=B,
            num_streams=K,
            num_slots=self.config.sidecar.planner_head.num_slots,
            device=self.device,
        )
        plan_snapshot = self.model.sidecar.plan_notes_proj(planner.quantized, ownership)
        snapshots_by_stream: list[list[torch.Tensor]] = [[plan_snapshot[:, k]] for k in range(K)]

        logits_by_block: list[torch.Tensor] = []
        labels_by_block: list[torch.Tensor] = []
        masks_by_block: list[torch.Tensor] = []
        dep_by_block: list[torch.Tensor] = []
        non_by_block: list[torch.Tensor] = []
        hidden_for_classifier: list[torch.Tensor] = []
        dynamic_commitment_losses: list[torch.Tensor] = []
        dynamic_codebook_losses: list[torch.Tensor] = []
        dynamic_note_indices: list[torch.Tensor] = []
        dynamic_assignment_logits: list[torch.Tensor] = []

        try:
            # One compact prefill per stream. The returned logits predict the
            # first target token and each cache remains attached to the graph.
            past_by_stream: list[object] = []
            attention_by_stream: list[torch.Tensor] = []
            next_logits_by_stream: list[torch.Tensor] = []
            self_histories: list[_SelfOnlyHistory] = []
            for stream_idx, stream in enumerate(stream_names):
                prompt_ids, prompt_mask = _compact_single_prompt(
                    batch.stream_prompt_ids[:, stream_idx],
                    batch.stream_prompt_attention_mask[:, stream_idx],
                )
                if coordination_source == "bus":
                    ctx = _rollout_layer_context(
                        snapshots_by_stream,
                        stream=stream,
                        consumer=stream_idx,
                        block_idx=0,
                        lag=self.config.runtime.notes_bus.lag,
                        history_blocks=self.config.runtime.notes_bus.history_blocks,
                        intervention=intervention,
                        scramble_generator=scramble_generator,
                    )
                else:
                    capture_context = LayerRuntimeContext(stream_ids=(stream,) * B)
                    _set_runtime_contexts(
                        self.model.instrumented_layers,
                        capture_context,
                    )
                    capture = self.model.trunk_adapter.forward(
                        input_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        use_cache=False,
                        output_hidden_states=True,
                    )
                    if capture.hidden_states is None:
                        raise RuntimeError(
                            "Self-only prompt capture must return final hidden states."
                        )
                    history = _SelfOnlyHistory.from_prompt(
                        stream=stream,
                        prompt_hidden=capture.hidden_states[-1],
                        prompt_mask=prompt_mask.bool(),
                        slots=K,
                        lag=self.config.runtime.notes_bus.lag,
                        history_blocks=self.config.runtime.notes_bus.history_blocks,
                    )
                    self_histories.append(history)
                    ctx = _self_only_layer_context(
                        history,
                        stream=stream,
                        block_idx=0,
                        query_positions=_sequence_positions(
                            batch=B,
                            start=prompt_ids.size(1),
                            length=prompt_ids.size(1),
                            device=prompt_ids.device,
                        ),
                        intervention=intervention,
                    )
                _set_runtime_contexts(self.model.instrumented_layers, ctx)
                out = self.model.trunk_adapter.forward(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    use_cache=True,
                    output_hidden_states=False,
                )
                if out.past_key_values is None:
                    raise RuntimeError(
                        "Frozen trunk did not return a differentiable KV cache during "
                        f"{stream} prefill."
                    )
                past_by_stream.append(out.past_key_values)
                attention_by_stream.append(prompt_mask)
                next_logits_by_stream.append(out.logits[:, -1, :])
            if coordination_source == "self_only" and len(self_histories) != K:
                raise RuntimeError("Self-only rollout failed to initialize one history per stream.")

            for block_idx in range(M):
                block_pre_writes: list[torch.Tensor] = []
                block_contexts: list[LayerRuntimeContext] = []
                if coordination_source == "bus":
                    for stream_idx in range(K):
                        stream = stream_names[stream_idx]
                        ctx = _rollout_layer_context(
                            snapshots_by_stream,
                            stream=stream,
                            consumer=stream_idx,
                            block_idx=block_idx,
                            lag=self.config.runtime.notes_bus.lag,
                            history_blocks=self.config.runtime.notes_bus.history_blocks,
                            intervention=intervention,
                            scramble_generator=scramble_generator,
                        )
                        block_contexts.append(ctx)

                for stream_idx in range(K):
                    stream = stream_names[stream_idx]
                    if coordination_source == "bus":
                        _set_runtime_contexts(
                            self.model.instrumented_layers,
                            block_contexts[stream_idx],
                        )

                    attention = attention_by_stream[stream_idx]
                    past = past_by_stream[stream_idx]
                    next_logits = next_logits_by_stream[stream_idx]
                    if block_idx > 0:
                        transition_ids, transition_mask = _compact_single_prompt(
                            batch.block_transition_ids[:, stream_idx, block_idx],
                            batch.block_transition_attention_mask[:, stream_idx, block_idx],
                        )
                        if coordination_source == "self_only":
                            transition_context = _self_only_layer_context(
                                self_histories[stream_idx],
                                stream=stream,
                                block_idx=block_idx,
                                query_positions=_sequence_positions(
                                    batch=B,
                                    start=attention.size(1),
                                    length=transition_ids.size(1),
                                    device=transition_ids.device,
                                ),
                                intervention=intervention,
                            )
                            _set_runtime_contexts(
                                self.model.instrumented_layers,
                                transition_context,
                            )
                        attention = torch.cat((attention, transition_mask), dim=1)
                        transition_out = self.model.trunk_adapter.forward(
                            input_ids=transition_ids,
                            attention_mask=attention,
                            past_key_values=past,
                            use_cache=True,
                            output_hidden_states=False,
                        )
                        if transition_out.past_key_values is None:
                            raise RuntimeError(
                                "Frozen trunk dropped the differentiable KV cache "
                                f"during stream {stream_idx} block {block_idx} "
                                "private-observation transition."
                            )
                        if transition_out.logits.size(1) != transition_ids.size(1):
                            raise RuntimeError(
                                "Frozen trunk returned a transition logit length "
                                f"mismatch for stream {stream_idx} block {block_idx}."
                            )
                        past = transition_out.past_key_values
                        next_logits = transition_out.logits[:, -1, :]

                    target_ids = batch.target_block_ids[:, stream_idx, block_idx]
                    target_start = attention.size(1)
                    target_positions = _sequence_positions(
                        batch=B,
                        start=target_start,
                        length=target_ids.size(1),
                        device=target_ids.device,
                    )
                    if coordination_source == "self_only":
                        target_context = _self_only_layer_context(
                            self_histories[stream_idx],
                            stream=stream,
                            block_idx=block_idx,
                            query_positions=target_positions,
                            intervention=intervention,
                        )
                        _set_runtime_contexts(
                            self.model.instrumented_layers,
                            target_context,
                        )
                    attention = torch.cat(
                        (
                            attention,
                            attention.new_ones((B, target_ids.size(1))),
                        ),
                        dim=1,
                    )
                    out = self.model.trunk_adapter.forward(
                        input_ids=target_ids,
                        attention_mask=attention,
                        past_key_values=past,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    if out.past_key_values is None:
                        raise RuntimeError(
                            "Frozen trunk dropped the differentiable KV cache during "
                            f"stream {stream_idx} block {block_idx}."
                        )
                    if out.logits.size(1) != target_ids.size(1):
                        raise RuntimeError(
                            "Frozen trunk returned a block logit length that does not "
                            f"match tau for stream {stream_idx} block {block_idx}: "
                            f"expected {target_ids.size(1)}, got {out.logits.size(1)}."
                        )
                    if out.hidden_states is None or out.hidden_states[-1].size(
                        1
                    ) != target_ids.size(1):
                        raise RuntimeError(
                            "Frozen trunk must return one final hidden state per consumed "
                            f"target token for stream {stream_idx} block {block_idx}."
                        )

                    past_by_stream[stream_idx] = out.past_key_values
                    attention_by_stream[stream_idx] = attention
                    next_logits_by_stream[stream_idx] = out.logits[:, -1, :]
                    # The cached prefix's final logit predicts target token 0;
                    # each target position then predicts its successor. Drop
                    # the final block logit, which predicts the next block.
                    current_logits = torch.cat(
                        (next_logits.unsqueeze(1), out.logits[:, :-1, :]),
                        dim=1,
                    )
                    current_hidden = out.hidden_states[-1]
                    current_mask = batch.target_block_attention_mask[
                        :, stream_idx, block_idx
                    ].bool()
                    logits_by_block.append(current_logits)
                    labels_by_block.append(batch.target_block_labels[:, stream_idx, block_idx])
                    masks_by_block.append(current_mask)
                    dep_by_block.append(batch.dependency_token_mask[:, stream_idx, block_idx])
                    non_by_block.append(batch.nondependency_token_mask[:, stream_idx, block_idx])
                    hidden_for_classifier.append(_masked_mean_hidden(current_hidden, current_mask))

                    last_hidden = _last_valid_hidden(current_hidden, current_mask)
                    block_pre_writes.append(
                        self.model.sidecar.speculation_head.project(last_hidden)
                    )
                    if coordination_source == "self_only":
                        self_histories[stream_idx].publish(
                            block_hidden=current_hidden,
                            block_mask=current_mask,
                            token_positions=target_positions,
                            block_idx=block_idx,
                        )

                # Synchronous publication: no producer's block-m write enters a
                # sibling's block-m context.
                for stream_idx, pre_write in enumerate(block_pre_writes):
                    write = self.model.sidecar.speculation_head.quantize(pre_write)
                    transmitted_indices = write.indices
                    transmitted_note = write.quantized
                    if (
                        intervention.mode == "bus_mutation"
                        and stream_names[stream_idx] == intervention.mutation_producer
                        and block_idx == intervention.mutation_block
                    ):
                        transmitted_indices = apply_bus_mutation(
                            transmitted_indices,
                            codes_per_codebook=(
                                self.model.sidecar.speculation_head.codes_per_codebook
                            ),
                            code_offset=intervention.mutation_code_offset,
                        )
                        transmitted_note = self.model.sidecar.speculation_head.decode(
                            transmitted_indices
                        )
                    snapshots_by_stream[stream_idx].append(transmitted_note)
                    dynamic_commitment_losses.append(write.commitment_loss)
                    dynamic_codebook_losses.append(write.codebook_loss)
                    dynamic_note_indices.append(transmitted_indices)
                    dynamic_assignment_logits.append(write.assignment_logits)
        finally:
            _clear_runtime_contexts(self.model.instrumented_layers)

        return _StudentRollout(
            lm_logits=torch.cat(logits_by_block, dim=0),
            lm_labels=torch.cat(labels_by_block, dim=0),
            lm_label_mask=torch.cat(masks_by_block, dim=0),
            dependency_mask=torch.cat(dep_by_block, dim=0),
            nondependency_mask=torch.cat(non_by_block, dim=0),
            classifier_hidden=torch.cat(hidden_for_classifier, dim=0),
            planner=planner,
            plan_snapshot=plan_snapshot,
            dynamic_vq_commitment_loss=torch.stack(dynamic_commitment_losses).mean(),
            dynamic_vq_codebook_loss=torch.stack(dynamic_codebook_losses).mean(),
            dynamic_note_indices=torch.stack(dynamic_note_indices, dim=1),
            dynamic_assignment_logits=torch.stack(dynamic_assignment_logits, dim=1),
        )

    @torch.no_grad()
    def _functional_teacher_logits(self, batch: SampleBatch) -> torch.Tensor:
        """Score every target with the frozen trunk's privileged serialized view.

        The teacher shares the student's exact trunk weights, but receives the
        all-stream prompt and every completed prior block. Clearing every layer
        context disables both SNC reads and stream adapters, so no coordination
        sidecar participates in the target distribution.
        """

        _clear_runtime_contexts(self.model.instrumented_layers)
        blocks = batch.target_block_ids.size(2)
        streams = batch.target_block_ids.size(1)
        logits: list[torch.Tensor] = []
        for block_idx in range(blocks):
            for stream_idx in range(streams):
                packed = _privileged_teacher_block_input(
                    batch,
                    stream_idx=stream_idx,
                    block_idx=block_idx,
                    pad_token_id=self.pad_token_id,
                )
                out = self.model.trunk_adapter.forward(
                    input_ids=packed.input_ids,
                    attention_mask=packed.attention_mask,
                    use_cache=False,
                    output_hidden_states=False,
                )
                logits.append(_gather_sequence_positions(out.logits, packed.prediction_positions))
        if not logits:
            raise ValueError("Functional teacher received a batch with no streams or blocks.")
        return torch.cat(logits, dim=0).detach()

    def _optimizer_step(self) -> None:
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.all_trainable_parameters() if p.requires_grad],
            max_norm=1.0,
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------ #
    # Eval + checkpoint
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _eval(self) -> None:
        if self._eval_loader is None:
            return
        model_was_training = self.model.training
        trunk_model = getattr(self.model.trunk_adapter, "model", None)
        trunk_was_training = bool(trunk_model.training) if trunk_model is not None else None
        coordination_source = self.config.instrumentation.coordination_source
        bus_evaluator = PairedCausalEvaluator() if coordination_source == "bus" else None
        self_evaluator = CausalDocumentEvaluator() if coordination_source == "self_only" else None
        self.codebook.reset()
        self.dynamic_codebook.reset()
        try:
            self.model.eval()
            if trunk_model is not None:
                trunk_model.eval()
            _require_cache_compatible_trunk(self.model.trunk_adapter)
            for batch_idx, raw_batch in enumerate(self._eval_loader):
                batch = _to_device(raw_batch, self.device)
                baseline = self._student_rollout(batch)
                if baseline.planner is None:
                    raise RuntimeError("Evaluation baseline did not return planner diagnostics.")
                self.codebook.observe_selections(baseline.planner.indices.detach().cpu())
                self.codebook.observe_anchors(baseline.plan_snapshot.detach().cpu())
                dynamic_indices = baseline.dynamic_note_indices.reshape(
                    -1, baseline.dynamic_note_indices.size(-1)
                )
                self.dynamic_codebook.observe_selections(dynamic_indices.detach().cpu())
                gate_zero = self._student_rollout(
                    batch,
                    intervention=_RolloutIntervention(mode="gate_zero"),
                )
                document_lag_masks = _dependency_lag_masks(batch)
                document_labels = _document_major_tensor(baseline.lm_labels, batch)
                document_label_mask = _document_major_tensor(
                    baseline.lm_label_mask, batch
                ).bool()
                document_dependency_mask = _document_major_tensor(
                    baseline.dependency_mask, batch
                ).bool()
                document_nondependency_mask = _document_major_tensor(
                    baseline.nondependency_mask, batch
                ).bool()
                if coordination_source == "bus":
                    norm_scramble = self._student_rollout(
                        batch,
                        intervention=_RolloutIntervention(
                            mode="norm_scramble",
                            seed=self.config.training.causal_eval_seed + batch_idx,
                        ),
                    )
                    mutation = self._student_rollout(
                        batch,
                        intervention=_RolloutIntervention(
                            mode="bus_mutation",
                            mutation_producer=(self.config.training.causal_eval_mutation_producer),
                            mutation_block=self.config.training.causal_eval_mutation_block,
                            mutation_code_offset=(
                                self.config.training.causal_eval_mutation_code_offset
                            ),
                        ),
                    )
                    _validate_rollout_alignment(
                        baseline,
                        gate_zero,
                        norm_scramble,
                        mutation,
                    )
                    mutation_dependency_mask = _mutation_dependency_mask(
                        batch,
                        producer=self.config.training.causal_eval_mutation_producer,
                        source_block=self.config.training.causal_eval_mutation_block,
                    )
                    if bus_evaluator is None:
                        raise RuntimeError("Bus evaluator was not initialized for the bus model.")
                    bus_evaluator.update(
                        baseline_logits=_document_major_tensor(baseline.lm_logits, batch),
                        gate_zero_logits=_document_major_tensor(gate_zero.lm_logits, batch),
                        norm_scramble_logits=_document_major_tensor(
                            norm_scramble.lm_logits, batch
                        ),
                        mutation_logits=_document_major_tensor(mutation.lm_logits, batch),
                        mutation_dependency_mask=_document_major_tensor(
                            mutation_dependency_mask, batch
                        ).bool(),
                        labels=document_labels,
                        label_mask=document_label_mask,
                        dependency_mask=document_dependency_mask,
                        nondependency_mask=document_nondependency_mask,
                        dependency_lag_masks=document_lag_masks,
                        example_ids=batch.example_ids,
                    )
                else:
                    _validate_rollout_alignment(baseline, gate_zero)
                    if self_evaluator is None:
                        raise RuntimeError(
                            "Self-only evaluator was not initialized for the control model."
                        )
                    self_evaluator.update(
                        normal_logits=_document_major_tensor(baseline.lm_logits, batch),
                        ablated_logits=_document_major_tensor(gate_zero.lm_logits, batch),
                        labels=document_labels,
                        label_mask=document_label_mask,
                        dependency_mask=document_dependency_mask,
                        nondependency_mask=document_nondependency_mask,
                        dependency_lag_masks=document_lag_masks,
                        example_ids=batch.example_ids,
                    )

            stats = self.codebook.compute()
            dynamic_stats = self.dynamic_codebook.compute()
            if coordination_source == "bus":
                if bus_evaluator is None:
                    raise RuntimeError("Bus evaluator is unavailable at metric finalization.")
                causal_metrics: dict[str, object] = bus_evaluator.compute(
                    bootstrap_samples=self.config.training.causal_eval_bootstrap_samples,
                    confidence_level=self.config.training.causal_eval_confidence_level,
                    seed=self.config.training.causal_eval_seed,
                    minimum_documents=self.config.training.causal_eval_min_documents,
                ).to_dict()
            else:
                if self_evaluator is None:
                    raise RuntimeError("Self-only evaluator is unavailable at finalization.")
                causal_metrics = {
                    "condition": "self_only",
                    "gate_zero": self_evaluator.compute(
                        bootstrap_samples=self.config.training.causal_eval_bootstrap_samples,
                        confidence_level=self.config.training.causal_eval_confidence_level,
                        seed=self.config.training.causal_eval_seed,
                    ).to_dict(),
                }
            metrics = {
                "global_step": self.global_step,
                "stage": self.curriculum.current_stage,
                "trunk_profile": self.config.trunk.profile,
                "coordination_source": coordination_source,
                "codebook": stats.to_dict(),
                "dynamic_codebook": dynamic_stats.to_dict(),
                "architecture": architecture_telemetry(self.model),
                "active_modules": self.curriculum.active_modules_snapshot(),
                "causal": causal_metrics,
            }
            (self.telemetry_dir / f"eval_{self.global_step:07d}.json").write_text(
                json.dumps(metrics, indent=2)
            )
            LOGGER.info(
                "eval@step=%d stage=%d codebook=%s causal=%s",
                self.global_step,
                self.curriculum.current_stage,
                stats.to_dict(),
                metrics["causal"],
            )
        finally:
            self.codebook.reset()
            self.dynamic_codebook.reset()
            _clear_runtime_contexts(self.model.instrumented_layers)
            self.model.train(model_was_training)
            if trunk_model is not None and trunk_was_training is not None:
                trunk_model.train(trunk_was_training)

    def evaluate(self) -> None:
        """Evaluate one loaded checkpoint on the configured document set without updates."""

        if self.global_step <= 0:
            raise RuntimeError("Evaluation-only execution requires a resumed trained checkpoint.")
        self._eval_loader = self._build_dataloader(
            self.config.training.eval_dataset_path,
            shuffle=False,
        )
        self._eval()

    def _save_checkpoint(self) -> None:
        ckpt_dir = self.telemetry_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"step_{self.global_step:07d}.pt"
        expected_stage = self.curriculum.determine_stage(self.global_step)
        if self.curriculum.current_stage != expected_stage:
            raise RuntimeError(
                "Refusing to save checkpoint under a stale curriculum policy: "
                f"current_stage={self.curriculum.current_stage}, "
                f"step={self.global_step}, expected_stage={expected_stage}."
            )
        save_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.scheduler,
            global_step=self.global_step,
            stage=expected_stage,
        )
        LOGGER.info("Saved checkpoint: %s", path)


def _to_device(batch: SampleBatch, device: torch.device) -> SampleBatch:
    return SampleBatch(
        example_ids=batch.example_ids,
        families=batch.families,
        stream_labels=batch.stream_labels,
        planner_prompt_ids=batch.planner_prompt_ids.to(device),
        planner_prompt_attention_mask=batch.planner_prompt_attention_mask.to(device),
        stream_prompt_ids=batch.stream_prompt_ids.to(device),
        stream_prompt_attention_mask=batch.stream_prompt_attention_mask.to(device),
        block_transition_ids=batch.block_transition_ids.to(device),
        block_transition_attention_mask=(batch.block_transition_attention_mask.to(device)),
        teacher_block_prompt_ids=batch.teacher_block_prompt_ids.to(device),
        teacher_block_prompt_attention_mask=(batch.teacher_block_prompt_attention_mask.to(device)),
        target_block_ids=batch.target_block_ids.to(device),
        target_block_labels=batch.target_block_labels.to(device),
        target_block_attention_mask=batch.target_block_attention_mask.to(device),
        dependency_token_mask=batch.dependency_token_mask.to(device),
        nondependency_token_mask=batch.nondependency_token_mask.to(device),
        raw=batch.raw,
    )


def _document_major_tensor(tensor: torch.Tensor, batch: SampleBatch) -> torch.Tensor:
    """Undo rollout's block/stream concatenation while preserving document identity."""

    batch_size, streams, blocks, block_width = batch.target_block_ids.shape
    expected_rows = batch_size * streams * blocks
    if tensor.ndim < 2 or tensor.size(0) != expected_rows or tensor.size(1) != block_width:
        raise ValueError(
            "Rollout tensor cannot be restored to document-major order: "
            f"shape={tuple(tensor.shape)}, expected leading shape "
            f"({expected_rows}, {block_width})."
        )
    trailing = tensor.shape[2:]
    block_stream_batch = tensor.reshape(blocks, streams, batch_size, block_width, *trailing)
    permutation = (2, 0, 1, 3, *range(4, block_stream_batch.ndim))
    document_major = block_stream_batch.permute(permutation)
    return document_major.reshape(batch_size, blocks * streams, block_width, *trailing)


def _dependency_lag_masks(batch: SampleBatch) -> dict[int, torch.Tensor]:
    """Build an exact document-major partition of dependency tokens by source lag."""

    batch_size, streams, blocks, block_width = batch.dependency_token_mask.shape
    if len(batch.raw) != batch_size:
        raise ValueError(
            f"Batch raw metadata has {len(batch.raw)} rows but tensor batch size is {batch_size}."
        )
    masks: dict[int, torch.Tensor] = {}
    for batch_index, record in enumerate(batch.raw):
        stream_rows = record.get("stream_inputs")
        if not isinstance(stream_rows, list) or len(stream_rows) != streams:
            raise ValueError(
                "Lag metadata must contain one stream_inputs row per tensor stream."
            )
        for stream_index, stream_row in enumerate(stream_rows):
            if not isinstance(stream_row, Mapping):
                raise ValueError("Lag metadata stream_inputs entries must be objects.")
            expected_stream = batch.stream_labels[batch_index][stream_index]
            if str(stream_row.get("stream_id", "")).lower() != expected_stream.lower():
                raise ValueError("Lag metadata order does not match collated stream labels.")
            spans = stream_row.get("dependency_spans")
            if not isinstance(spans, list):
                raise ValueError("Lag metadata dependency_spans must be a list.")
            by_target: dict[int, set[int]] = {}
            for span in spans:
                if not isinstance(span, Mapping):
                    raise ValueError("Lag metadata dependency spans must be objects.")
                target_block = span.get("block_index")
                lag = span.get("lag_blocks")
                if type(target_block) is not int or not 0 <= target_block < blocks:
                    raise ValueError(
                        f"Lag metadata target block {target_block!r} is outside [0, {blocks})."
                    )
                if type(lag) is not int or lag <= 0:
                    raise ValueError(f"Lag metadata value must be positive, got {lag!r}.")
                by_target.setdefault(target_block, set()).add(lag)
            for target_block, target_lags in by_target.items():
                if len(target_lags) != 1:
                    raise ValueError(
                        "A combined dependency-token mask cannot isolate multiple lags "
                        f"within block {target_block}; observed {sorted(target_lags)}."
                    )
                lag = next(iter(target_lags))
                mask = masks.setdefault(
                    lag,
                    torch.zeros(
                        (batch_size, streams, blocks, block_width),
                        dtype=torch.bool,
                        device=batch.dependency_token_mask.device,
                    ),
                )
                mask[batch_index, stream_index, target_block] = batch.dependency_token_mask[
                    batch_index, stream_index, target_block
                ].bool()
    if not masks:
        raise ValueError("Causal evaluation requires at least one annotated dependency lag.")
    union = torch.zeros_like(batch.dependency_token_mask, dtype=torch.bool)
    document_masks: dict[int, torch.Tensor] = {}
    for lag, mask in sorted(masks.items()):
        if bool((union & mask).any()):
            raise ValueError("Lag masks overlap; dependency annotations are ambiguous.")
        union |= mask
        document_masks[lag] = mask.permute(0, 2, 1, 3).reshape(
            batch_size, blocks * streams, block_width
        )
    if not torch.equal(union, batch.dependency_token_mask.bool()):
        raise ValueError(
            "Dependency lag annotations must cover dependency_token_mask exactly."
        )
    return document_masks


def _mutation_dependency_mask(
    batch: SampleBatch,
    *,
    producer: str,
    source_block: int,
) -> torch.Tensor:
    """Align one mutated producer/write with its annotated receiver target tokens."""

    B, K, M, tau = batch.dependency_token_mask.shape
    if len(batch.raw) != B:
        raise ValueError(
            f"Batch raw metadata has {len(batch.raw)} rows but tensor batch size is {B}."
        )
    if type(source_block) is not int or source_block < 0:
        raise ValueError("Mutation source_block must be a non-negative integer.")
    normalized_producer = producer.lower()
    aligned = torch.zeros(
        (B, K, M, tau),
        dtype=torch.bool,
        device=batch.dependency_token_mask.device,
    )
    for batch_idx, record in enumerate(batch.raw):
        streams = record.get("stream_inputs")
        if not isinstance(streams, list) or len(streams) != K:
            raise ValueError(
                "Mutation-mask metadata must contain one stream_inputs row per tensor stream."
            )
        for stream_idx, stream_value in enumerate(streams):
            if not isinstance(stream_value, Mapping):
                raise ValueError("Mutation-mask stream metadata entries must be objects.")
            expected_stream = batch.stream_labels[batch_idx][stream_idx]
            if str(stream_value.get("stream_id", "")).lower() != expected_stream.lower():
                raise ValueError(
                    "Mutation-mask stream metadata order does not match collated stream labels."
                )
            spans = stream_value.get("dependency_spans")
            if not isinstance(spans, list):
                raise ValueError("Mutation-mask metadata dependency_spans must be a list.")
            spans_by_target: dict[int, list[Mapping[str, object]]] = {}
            for span in spans:
                if not isinstance(span, Mapping):
                    raise ValueError("Mutation-mask dependency spans must be objects.")
                target_block = span.get("block_index")
                if type(target_block) is not int or not 0 <= target_block < M:
                    raise ValueError(
                        f"Mutation-mask dependency block {target_block!r} is outside [0, {M})."
                    )
                spans_by_target.setdefault(target_block, []).append(span)
            for target_block, target_spans in spans_by_target.items():
                selected = [
                    span
                    for span in target_spans
                    if str(span.get("source_stream", "")).lower() == normalized_producer
                    and span.get("source_block_index") == source_block
                ]
                if not selected:
                    continue
                if len(selected) != len(target_spans):
                    raise ValueError(
                        "Combined dependency_token_mask cannot isolate mixed source spans in "
                        f"batch {batch_idx} stream {stream_idx} block {target_block}."
                    )
                aligned[batch_idx, stream_idx, target_block] = batch.dependency_token_mask[
                    batch_idx, stream_idx, target_block
                ].bool()

    rows = [aligned[:, stream_idx, block_idx] for block_idx in range(M) for stream_idx in range(K)]
    flattened = torch.cat(rows, dim=0)
    if not bool(flattened.any()):
        raise ValueError(
            "Configured bus mutation has zero aligned dependency tokens in this eval batch: "
            f"producer={producer!r}, source_block={source_block}."
        )
    return flattened


def _validate_rollout_alignment(
    baseline: _StudentRollout,
    *interventions: _StudentRollout,
) -> None:
    """Fail before metric accumulation if an intervention changed token alignment."""

    for index, rollout in enumerate(interventions, start=1):
        if rollout.lm_logits.shape != baseline.lm_logits.shape:
            raise RuntimeError(f"Causal rollout {index} logit shape is not aligned with baseline.")
        for field in (
            "lm_labels",
            "lm_label_mask",
            "dependency_mask",
            "nondependency_mask",
        ):
            if not torch.equal(getattr(rollout, field), getattr(baseline, field)):
                raise RuntimeError(
                    f"Causal rollout {index} changed block-major token alignment for {field}."
                )


def _require_cache_compatible_trunk(trunk_adapter) -> None:
    """Reject the HF train/checkpointing state that silently drops KV caches."""

    trunk_model = getattr(trunk_adapter, "model", None)
    if trunk_model is None:
        return
    if trunk_model.training:
        raise RuntimeError(
            "Canonical incremental training requires the frozen trunk in eval mode; "
            "Hugging Face may disable use_cache when train mode and gradient "
            "checkpointing coincide."
        )
    if bool(getattr(trunk_model, "is_gradient_checkpointing", False)):
        raise RuntimeError(
            "Canonical incremental training is incompatible with gradient "
            "checkpointing because every target token must retain and reuse its "
            "differentiable KV cache."
        )


def _infinite(loader):
    while True:
        for batch in loader:
            yield batch


def _round_robin_ownership(
    *,
    batch_size: int,
    num_streams: int,
    num_slots: int,
    device: torch.device,
) -> torch.Tensor:
    ownership = torch.zeros(
        (batch_size, num_streams, num_slots),
        dtype=torch.bool,
        device=device,
    )
    for slot in range(num_slots):
        ownership[:, slot % num_streams, slot] = True
    return ownership


def _validate_fixed_tau_blocks(batch: SampleBatch, *, tau: int) -> None:
    """Require training publications at the identical token cadence as runtime."""

    if tau <= 0:
        raise ValueError(f"Runtime synchronization stride tau must be positive, got {tau}.")
    ids = batch.target_block_ids
    mask = batch.target_block_attention_mask
    if ids.ndim != 4 or mask.shape != ids.shape:
        raise ValueError("Target block ids/mask must have matching [B, K, M, tau] shape.")
    if ids.size(-1) != tau:
        raise ValueError(
            "Training target width must equal runtime synchronization stride: "
            f"target width={ids.size(-1)}, tau={tau}."
        )
    lengths = mask.to(dtype=torch.long).sum(dim=-1)
    if not torch.all(lengths == tau):
        bad = (lengths != tau).nonzero(as_tuple=False).tolist()
        raise ValueError(
            "Every training block must contain exactly tau valid tokens before its note "
            f"publication; tau={tau}, invalid [batch, stream, block] rows={bad}."
        )


def _validate_block_transitions(batch: SampleBatch) -> None:
    """Require one empty runway row and one non-empty private reveal thereafter."""

    ids = batch.block_transition_ids
    mask = batch.block_transition_attention_mask
    target_shape = batch.target_block_ids.shape[:3]
    if ids.ndim != 4 or mask.shape != ids.shape or ids.shape[:3] != target_shape:
        raise ValueError(
            "Block transition ids/mask must have matching [B, K, M, tokens] "
            "shape aligned with target blocks."
        )
    active = mask.to(dtype=torch.bool)
    if active.size(-1) > 1 and (active[..., 1:] & ~active[..., :-1]).any():
        raise ValueError("Block transition masks must be prefix-contiguous.")
    lengths = active.to(dtype=torch.long).sum(dim=-1)
    if (lengths[:, :, 0] != 0).any():
        raise ValueError("Block transition row 0 must be empty; observation 0 is in the prefill.")
    if lengths.size(-1) > 1 and (lengths[:, :, 1:] == 0).any():
        bad = (lengths[:, :, 1:] == 0).nonzero(as_tuple=False).tolist()
        raise ValueError(
            "Every block after row 0 must reveal a non-empty private observation; "
            f"invalid [batch, stream, block-1] rows={bad}."
        )


@dataclass(slots=True)
class _SelfOnlyHistory:
    """Receiver-owned fixed history with the same topology as the bus window."""

    stream: str
    slots: int
    lag: int
    history_blocks: int
    prompt_tail: torch.Tensor
    prompt_positions: torch.Tensor
    block_tails: list[tuple[torch.Tensor, torch.Tensor]]

    @classmethod
    def from_prompt(
        cls,
        *,
        stream: str,
        prompt_hidden: torch.Tensor,
        prompt_mask: torch.Tensor,
        slots: int,
        lag: int,
        history_blocks: int,
    ) -> _SelfOnlyHistory:
        if type(slots) is not int or slots <= 0:
            raise ValueError("Self-only memory slots must be a positive integer.")
        if type(lag) is not int or lag <= 0:
            raise ValueError("Self-only memory requires a positive causal delivery lag.")
        if type(history_blocks) is not int or history_blocks <= 0:
            raise ValueError("Self-only history_blocks must be a positive integer.")
        positions = _sequence_positions(
            batch=prompt_hidden.size(0),
            start=0,
            length=prompt_hidden.size(1),
            device=prompt_hidden.device,
        )
        tail, tail_positions = _last_n_valid_states(
            prompt_hidden,
            prompt_mask,
            positions,
            count=slots,
            label="self-only prompt",
        )
        return cls(
            stream=stream,
            slots=slots,
            lag=lag,
            history_blocks=history_blocks,
            prompt_tail=tail,
            prompt_positions=tail_positions,
            block_tails=[],
        )

    def window(self, *, consumer_block: int) -> SelfOnlyMemory:
        if type(consumer_block) is not int or consumer_block < 0:
            raise ValueError("Self-only consumer_block must be a non-negative integer.")
        batch, slots, hidden = self.prompt_tail.shape
        if slots != self.slots:
            raise RuntimeError("Self-only prompt tail width changed after initialization.")
        dynamic_rows: list[torch.Tensor] = []
        dynamic_position_rows: list[torch.Tensor] = []
        dynamic_masks: list[torch.Tensor] = []
        for age in range(1, self.history_blocks + 1):
            source_block = consumer_block - age
            if age >= self.lag and source_block >= 0:
                if source_block >= len(self.block_tails):
                    raise RuntimeError(
                        "Self-only history is missing an eligible causal block: "
                        f"consumer_block={consumer_block}, source_block={source_block}, "
                        f"available={len(self.block_tails)}."
                    )
                dynamic, dynamic_positions = self.block_tails[source_block]
                dynamic_mask = torch.ones(
                    (batch, slots),
                    dtype=torch.bool,
                    device=self.prompt_tail.device,
                )
            else:
                dynamic = self.prompt_tail.new_zeros((batch, slots, hidden))
                dynamic_positions = torch.full(
                    (batch, slots),
                    -1,
                    dtype=torch.long,
                    device=self.prompt_tail.device,
                )
                dynamic_mask = torch.zeros(
                    (batch, slots),
                    dtype=torch.bool,
                    device=self.prompt_tail.device,
                )
            dynamic_rows.append(dynamic)
            dynamic_position_rows.append(dynamic_positions)
            dynamic_masks.append(dynamic_mask)
        prompt_mask = torch.ones(
            (batch, slots),
            dtype=torch.bool,
            device=self.prompt_tail.device,
        )
        slot_ids = torch.arange(slots, device=self.prompt_tail.device, dtype=torch.long).repeat(
            self.history_blocks + 1
        )
        kind_ids = torch.cat(
            (
                torch.zeros(slots, device=self.prompt_tail.device, dtype=torch.long),
                torch.ones(
                    slots * self.history_blocks,
                    device=self.prompt_tail.device,
                    dtype=torch.long,
                ),
            )
        )
        lags = torch.cat(
            (
                torch.zeros(slots, device=self.prompt_tail.device, dtype=torch.long),
                torch.arange(
                    self.history_blocks,
                    device=self.prompt_tail.device,
                    dtype=torch.long,
                ).add(1).repeat_interleave(slots),
            )
        )
        return SelfOnlyMemory(
            hidden_states=torch.cat((self.prompt_tail, *dynamic_rows), dim=1),
            mask=torch.cat((prompt_mask, *dynamic_masks), dim=1),
            positions=torch.cat((self.prompt_positions, *dynamic_position_rows), dim=1),
            slot_ids=slot_ids.unsqueeze(0).expand(batch, -1),
            kind_ids=kind_ids.unsqueeze(0).expand(batch, -1),
            lags=lags.unsqueeze(0).expand(batch, -1),
            owner_streams=(self.stream,) * batch,
        )

    def publish(
        self,
        *,
        block_hidden: torch.Tensor,
        block_mask: torch.Tensor,
        token_positions: torch.Tensor,
        block_idx: int,
    ) -> None:
        if block_idx != len(self.block_tails):
            raise RuntimeError(
                "Self-only block publication must be monotone and gap-free: "
                f"received={block_idx}, expected={len(self.block_tails)}."
            )
        tail, positions = _last_n_valid_states(
            block_hidden,
            block_mask,
            token_positions,
            count=self.slots,
            label=f"self-only block {block_idx}",
        )
        self.block_tails.append((tail, positions))


def _last_n_valid_states(
    hidden: torch.Tensor,
    mask: torch.Tensor,
    positions: torch.Tensor,
    *,
    count: int,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or positions.shape != mask.shape:
        raise ValueError(f"{label} requires hidden [B,T,H] and aligned mask/positions.")
    if mask.dtype != torch.bool:
        raise TypeError(f"{label} mask must have dtype torch.bool.")
    lengths = mask.sum(dim=1)
    if bool((lengths < count).any()):
        rows = (lengths < count).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(
            f"{label} must expose at least K={count} receiver-owned states; short rows={rows}."
        )
    state_rows: list[torch.Tensor] = []
    position_rows: list[torch.Tensor] = []
    for batch_idx in range(hidden.size(0)):
        indices = mask[batch_idx].nonzero(as_tuple=False).flatten()[-count:]
        state_rows.append(hidden[batch_idx].index_select(0, indices))
        position_rows.append(positions[batch_idx].index_select(0, indices))
    return torch.stack(state_rows), torch.stack(position_rows)


def _sequence_positions(
    *,
    batch: int,
    start: int,
    length: int,
    device: torch.device,
) -> torch.Tensor:
    if batch <= 0 or start < 0 or length <= 0:
        raise ValueError("Sequence-position construction requires batch>0, start>=0, and length>0.")
    row = torch.arange(start, start + length, dtype=torch.long, device=device)
    return row.unsqueeze(0).expand(batch, -1)


def _self_only_layer_context(
    history: _SelfOnlyHistory,
    *,
    stream: str,
    block_idx: int,
    query_positions: torch.Tensor,
    intervention: _RolloutIntervention,
) -> LayerRuntimeContext:
    if stream != history.stream:
        raise ValueError(
            f"Self-only context ownership mismatch: stream={stream!r}, owner={history.stream!r}."
        )
    return LayerRuntimeContext(
        stream_ids=(stream,) * query_positions.size(0),
        self_only_memory=history.window(consumer_block=block_idx),
        self_only_query_positions=query_positions,
        snc_force_gate=False if intervention.mode == "gate_zero" else None,
    )


def _validate_rollout_intervention(
    intervention: _RolloutIntervention,
    *,
    streams: tuple[str, ...],
    codes_per_codebook: int,
    coordination_source: str,
) -> None:
    if intervention.mode not in {
        "normal",
        "gate_zero",
        "norm_scramble",
        "bus_mutation",
    }:
        raise ValueError(f"Unknown rollout intervention mode {intervention.mode!r}.")
    if type(intervention.seed) is not int or intervention.seed < 0:
        raise ValueError(
            f"Causal rollout seed must be a non-negative integer, got {intervention.seed!r}."
        )
    if coordination_source not in ("bus", "self_only"):
        raise ValueError(f"Unknown coordination source {coordination_source!r}.")
    if coordination_source == "self_only" and intervention.mode not in {"normal", "gate_zero"}:
        raise ValueError(
            f"Intervention {intervention.mode!r} requires a sibling bus and is undefined "
            "for the self-only control."
        )
    if intervention.mode != "bus_mutation":
        return
    if intervention.mutation_producer not in streams:
        raise ValueError(
            "Bus mutation producer must name a configured runtime stream; "
            f"got {intervention.mutation_producer!r}, expected one of {streams}."
        )
    if type(intervention.mutation_block) is not int or intervention.mutation_block < 0:
        raise ValueError("Bus mutation block must be a non-negative integer.")
    if (
        type(intervention.mutation_code_offset) is not int
        or not 0 < intervention.mutation_code_offset < codes_per_codebook
    ):
        raise ValueError(
            "Bus mutation code offset must be an integer in "
            f"[1, {codes_per_codebook}); got {intervention.mutation_code_offset!r}."
        )


def _rollout_layer_context(
    snapshots_by_stream: list[list[torch.Tensor]],
    *,
    stream: str,
    consumer: int,
    block_idx: int,
    lag: int,
    history_blocks: int,
    intervention: _RolloutIntervention,
    scramble_generator: torch.Generator | None,
) -> LayerRuntimeContext:
    """Build one frozen block context and apply only the selected intervention."""

    notes, notes_mask = _visible_notes(
        snapshots_by_stream,
        consumer=consumer,
        block_idx=block_idx,
        lag=lag,
        history_blocks=history_blocks,
    )
    producer_count = len(snapshots_by_stream)
    producer_ids = torch.arange(
        producer_count,
        dtype=torch.long,
        device=notes.device,
    ).repeat(history_blocks + 1)
    kind_ids = torch.cat(
        (
            torch.zeros(producer_count, dtype=torch.long, device=notes.device),
            torch.ones(
                producer_count * history_blocks,
                dtype=torch.long,
                device=notes.device,
            ),
        )
    )
    note_lags = torch.cat(
        (
            torch.zeros(producer_count, dtype=torch.long, device=notes.device),
            torch.arange(
                history_blocks,
                dtype=torch.long,
                device=notes.device,
            ).add(1).repeat_interleave(producer_count),
        )
    )
    if intervention.mode == "norm_scramble":
        if scramble_generator is None:
            raise RuntimeError("Norm scramble requires an initialized CPU generator.")
        sibling_dynamic_slots = (kind_ids == 1) & (producer_ids != consumer)
        notes = apply_norm_scramble(
            notes,
            generator=scramble_generator,
            slot_mask=sibling_dynamic_slots,
        )
    batch = notes.size(0)
    return LayerRuntimeContext(
        stream_ids=(stream,) * batch,
        notes=notes,
        notes_mask=notes_mask,
        note_producer_ids=producer_ids.unsqueeze(0).expand(batch, -1),
        note_kind_ids=kind_ids.unsqueeze(0).expand(batch, -1),
        note_lags=note_lags.unsqueeze(0).expand(batch, -1),
        snc_force_gate=False if intervention.mode == "gate_zero" else None,
    )


def _visible_notes(
    snapshots_by_stream: list[list[torch.Tensor]],
    *,
    consumer: int,
    block_idx: int,
    lag: int,
    history_blocks: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the runtime-equivalent fixed addressed history window."""

    del consumer
    if block_idx < 0:
        raise ValueError("block_idx must be non-negative.")
    if lag < 0:
        raise ValueError("lag must be non-negative.")
    if type(history_blocks) is not int or history_blocks <= 0:
        raise ValueError("history_blocks must be a positive integer.")
    if not snapshots_by_stream or any(not snapshots for snapshots in snapshots_by_stream):
        raise ValueError("Every producer must provide one prompt anchor.")

    anchors = [snapshots[0] for snapshots in snapshots_by_stream]
    reference = anchors[0]
    if reference.ndim != 2:
        raise ValueError("Training notes must have shape [batch, notes_dim].")
    expected_shape = reference.shape
    if any(note.shape != expected_shape for note in anchors):
        raise ValueError("All producer anchors must share [batch, notes_dim] shape.")

    dynamics: list[torch.Tensor] = []
    dynamic_present: list[bool] = []
    for age in range(1, history_blocks + 1):
        source_block = block_idx - age
        for snapshots in snapshots_by_stream:
            dynamic_index = source_block + 1  # index 0 is the prompt anchor
            if age >= lag and source_block >= 0 and dynamic_index < len(snapshots):
                dynamic = snapshots[dynamic_index]
                if dynamic.shape != expected_shape:
                    raise ValueError(
                        "All dynamic notes must share the anchors' [batch, notes_dim] shape."
                    )
                dynamics.append(dynamic)
                dynamic_present.append(True)
            else:
                dynamics.append(torch.zeros_like(reference))
                dynamic_present.append(False)

    notes = torch.stack((*anchors, *dynamics), dim=1)
    slot_mask = torch.tensor(
        [True] * len(anchors) + dynamic_present,
        dtype=torch.bool,
        device=reference.device,
    )
    mask = slot_mask.unsqueeze(0).expand(reference.size(0), -1)
    return notes, mask


@dataclass(slots=True)
class _PackedBlockInput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prediction_positions: torch.Tensor
    target_positions: torch.Tensor


def _compact_single_prompt(
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove right padding before the canonical batch-one cache prefill."""

    if prompt_ids.ndim != 2 or prompt_mask.shape != prompt_ids.shape:
        raise ValueError("Prompt ids and mask must have matching [batch, tokens] shape.")
    if prompt_ids.size(0) != 1:
        raise ValueError(
            "Canonical cached prefill requires one prompt row; "
            f"got batch size {prompt_ids.size(0)}."
        )
    compact_ids = prompt_ids[:, prompt_mask[0].bool()]
    if compact_ids.size(1) == 0:
        raise ValueError("Canonical cached prefill requires at least one prompt token.")
    compact_mask = prompt_mask.new_ones(compact_ids.shape)
    return compact_ids, compact_mask


def _privileged_teacher_block_input(
    batch: SampleBatch,
    *,
    stream_idx: int,
    block_idx: int,
    pad_token_id: int,
) -> _PackedBlockInput:
    """Use the retokenized full privileged prefix for this synchronization block."""

    return _pack_block_input(
        prompt_ids=batch.teacher_block_prompt_ids[:, block_idx],
        prompt_mask=batch.teacher_block_prompt_attention_mask[:, block_idx],
        prior_segments=[],
        target_ids=batch.target_block_ids[:, stream_idx, block_idx],
        target_mask=batch.target_block_attention_mask[:, stream_idx, block_idx],
        pad_token_id=pad_token_id,
    )


def _pack_block_input(
    *,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    prior_segments: list[tuple[torch.Tensor, torch.Tensor]],
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    pad_token_id: int,
) -> _PackedBlockInput:
    """Compact padded segments and record causal prediction/target positions."""

    if prompt_ids.ndim != 2 or prompt_mask.shape != prompt_ids.shape:
        raise ValueError("Prompt ids and mask must have matching [batch, tokens] shape.")
    if target_ids.ndim != 2 or target_mask.shape != target_ids.shape:
        raise ValueError("Target ids and mask must have matching [batch, tokens] shape.")
    if target_ids.size(0) != prompt_ids.size(0):
        raise ValueError("Prompt and target batch sizes must match.")

    batch_size, target_width = target_ids.shape
    sequences: list[torch.Tensor] = []
    prediction_positions = torch.zeros(
        (batch_size, target_width),
        dtype=torch.long,
        device=prompt_ids.device,
    )
    target_positions = torch.zeros_like(prediction_positions)

    for batch_idx in range(batch_size):
        prefix_parts = [prompt_ids[batch_idx][prompt_mask[batch_idx].bool()]]
        for ids, mask in prior_segments:
            if ids.shape != mask.shape or ids.ndim != 2 or ids.size(0) != batch_size:
                raise ValueError("Every prior block ids/mask pair must have [batch, tokens] shape.")
            prefix_parts.append(ids[batch_idx][mask[batch_idx].bool()])
        prefix = torch.cat(prefix_parts)
        if prefix.numel() == 0:
            raise ValueError(
                "A causal target block requires at least one prompt or prior-prefix token."
            )
        target = target_ids[batch_idx][target_mask[batch_idx].bool()]
        if target.numel() == 0:
            raise ValueError(
                "Every configured stream/block must contain at least one target token; "
                f"batch row {batch_idx} is empty."
            )
        sequence = torch.cat((prefix, target))
        sequences.append(sequence)

        target_count = target.numel()
        if target_count:
            offset = torch.arange(
                target_count,
                device=prompt_ids.device,
                dtype=torch.long,
            )
            prediction_positions[batch_idx, :target_count] = prefix.numel() - 1 + offset
            target_positions[batch_idx, :target_count] = prefix.numel() + offset

    max_length = max(sequence.numel() for sequence in sequences)
    packed_ids = prompt_ids.new_full((batch_size, max_length), pad_token_id)
    packed_mask = prompt_mask.new_zeros((batch_size, max_length))
    for batch_idx, sequence in enumerate(sequences):
        packed_ids[batch_idx, : sequence.numel()] = sequence
        packed_mask[batch_idx, : sequence.numel()] = 1

    return _PackedBlockInput(
        input_ids=packed_ids,
        attention_mask=packed_mask,
        prediction_positions=prediction_positions,
        target_positions=target_positions,
    )


def _gather_sequence_positions(
    states: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    if states.ndim != 3 or positions.ndim != 2:
        raise ValueError("Expected states [batch, tokens, dim] and positions [batch, targets].")
    if states.size(0) != positions.size(0):
        raise ValueError("State and position batch sizes must match.")
    if positions.numel() and (
        positions.min().item() < 0 or positions.max().item() >= states.size(1)
    ):
        raise ValueError("Gather position is outside the model sequence length.")
    index = positions.unsqueeze(-1).expand(-1, -1, states.size(-1))
    return states.gather(dim=1, index=index)


def _clear_runtime_contexts(layers) -> None:
    for layer in layers:
        layer.set_runtime_context(None)


def _set_runtime_contexts(layers, context: LayerRuntimeContext) -> None:
    for layer in layers:
        layer.set_runtime_context(context)


def _active_phi_gradient_report(model: PDTModel) -> dict[str, object]:
    """Validate finite nonzero gradients for every active mechanism group."""

    groups: dict[str, list[tuple[str, torch.nn.Parameter]]] = {}

    def add_module(group: str, prefix: str, module: torch.nn.Module) -> None:
        groups.setdefault(group, []).extend(
            (f"{prefix}.{name}", parameter) for name, parameter in module.named_parameters()
        )

    for name in (
        "planner_head",
        "plan_notes_proj",
        "speculation_head",
        "stream_classifier",
    ):
        add_module(name, f"sidecar.{name}", getattr(model.sidecar, name))
    for layer in model.instrumented_layers:
        index = layer.pdt_layer_idx
        if layer.snc is not None:
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                add_module(
                    f"snc_{projection}",
                    f"layer_{index}.snc.{projection}",
                    getattr(layer.snc, projection),
                )
            add_module(
                "snc_headers",
                f"layer_{index}.snc.producer_embedding",
                layer.snc.producer_embedding,
            )
            add_module(
                "snc_headers",
                f"layer_{index}.snc.kind_embedding",
                layer.snc.kind_embedding,
            )
            add_module(
                "snc_headers",
                f"layer_{index}.snc.lag_projection",
                layer.snc.lag_projection,
            )
            groups.setdefault("snc_inner_gate", []).append(
                (f"layer_{index}.snc.gate", layer.snc.gate)
            )
        if layer.stream_adapter is not None:
            add_module(
                "stream_adapters",
                f"layer_{index}.stream_adapter",
                layer.stream_adapter,
            )
        if layer.notes_gate is not None:
            groups.setdefault("snc_outer_gate", []).append(
                (f"layer_{index}.notes_gate", layer.notes_gate)
            )
        if layer.adapter_gate is not None:
            groups.setdefault("adapter_outer_gate", []).append(
                (f"layer_{index}.adapter_gate", layer.adapter_gate)
            )

    report: dict[str, object] = {}
    active_group_count = 0
    for group, named_parameters in groups.items():
        active = [
            (name, parameter) for name, parameter in named_parameters if parameter.requires_grad
        ]
        if not active:
            report[group] = {"active_parameters": 0}
            continue
        active_group_count += 1
        missing = [name for name, parameter in active if parameter.grad is None]
        if missing:
            raise RuntimeError(f"CUDA optimizer probe found missing {group} gradients: {missing}.")
        nonfinite = [
            name
            for name, parameter in active
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
        ]
        if nonfinite:
            raise RuntimeError(
                f"CUDA optimizer probe found non-finite {group} gradients: {nonfinite}."
            )
        gradient_l1 = sum(
            float(parameter.grad.detach().float().abs().sum().item())
            for _, parameter in active
            if parameter.grad is not None
        )
        if not math.isfinite(gradient_l1) or gradient_l1 <= 0:
            raise RuntimeError(f"CUDA optimizer probe found a zero {group} gradient path.")
        report[group] = {
            "active_parameters": len(active),
            "trainable_scalars": sum(parameter.numel() for _, parameter in active),
            "gradient_l1": gradient_l1,
        }
    if active_group_count == 0:
        raise RuntimeError("CUDA optimizer probe found no active phi parameter groups.")
    return report


def _masked_mean_hidden(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return (hidden * weights).sum(dim=1) / denom


def _last_valid_hidden(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return each sample's last generated target state for the bus writer."""

    if hidden.ndim != 3 or mask.shape != hidden.shape[:2]:
        raise ValueError("Expected hidden [batch, tokens, dim] and matching token mask.")
    lengths = mask.to(device=hidden.device, dtype=torch.long).sum(dim=1)
    if (lengths == 0).any():
        rows = (lengths == 0).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"Cannot publish a block write for empty target rows: {rows}.")
    positions = lengths - 1
    batch_indices = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch_indices, positions]
