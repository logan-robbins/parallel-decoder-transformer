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

from pdt.checkpoint import (
    CheckpointMetadata,
    CheckpointMismatchError,
    resume_checkpoint,
    save_checkpoint,
)
from pdt.config.schemas import PDTConfig
from pdt.diagnostics.codebook import CodebookDiagnostics
from pdt.evaluation.paired_causal import PairedCausalEvaluator
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
    mutation_magnitude: float = 1.0


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
            import math

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
        dataset = PDTDependencyDataset(path, num_streams=self.config.sidecar.num_streams)
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

        # Final save.
        self._save_checkpoint()
        self._eval()

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
            vq_commitment_loss=rollout.planner.commitment_loss,
            vq_codebook_loss=rollout.planner.codebook_loss,
            planner_logits=rollout.planner.logits,
            stream_logits=stream_classifier_logits,
            stream_targets=stream_targets,
        )

        loss = losses.total / self.config.training.grad_accumulation
        loss.backward()

        with torch.no_grad():
            self.codebook.observe_selections(rollout.planner.indices.detach().cpu())
            self.codebook.observe_anchors(rollout.plan_snapshot.detach().cpu())

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
        _validate_rollout_intervention(intervention, streams=self.config.runtime.streams)
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

        try:
            # One compact prefill per stream. The returned logits predict the
            # first target token and each cache remains attached to the graph.
            past_by_stream: list[object] = []
            attention_by_stream: list[torch.Tensor] = []
            next_logits_by_stream: list[torch.Tensor] = []
            for stream_idx, stream in enumerate(stream_names):
                ctx = _rollout_layer_context(
                    snapshots_by_stream,
                    stream=stream,
                    consumer=stream_idx,
                    block_idx=0,
                    lag=self.config.runtime.notes_bus.lag,
                    intervention=intervention,
                    scramble_generator=scramble_generator,
                )
                for layer in self.model.instrumented_layers:
                    layer.set_runtime_context(ctx)
                prompt_ids, prompt_mask = _compact_single_prompt(
                    batch.stream_prompt_ids[:, stream_idx],
                    batch.stream_prompt_attention_mask[:, stream_idx],
                )
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

            for block_idx in range(M):
                block_writes: list[torch.Tensor] = []
                block_contexts: list[LayerRuntimeContext] = []
                for stream_idx in range(K):
                    stream = stream_names[stream_idx]
                    ctx = _rollout_layer_context(
                        snapshots_by_stream,
                        stream=stream,
                        consumer=stream_idx,
                        block_idx=block_idx,
                        lag=self.config.runtime.notes_bus.lag,
                        intervention=intervention,
                        scramble_generator=scramble_generator,
                    )
                    block_contexts.append(ctx)

                for stream_idx in range(K):
                    stream = stream_names[stream_idx]
                    for layer in self.model.instrumented_layers:
                        layer.set_runtime_context(block_contexts[stream_idx])

                    attention = attention_by_stream[stream_idx]
                    past = past_by_stream[stream_idx]
                    next_logits = next_logits_by_stream[stream_idx]
                    if block_idx > 0:
                        transition_ids, transition_mask = _compact_single_prompt(
                            batch.block_transition_ids[:, stream_idx, block_idx],
                            batch.block_transition_attention_mask[:, stream_idx, block_idx],
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
                    block_writes.append(self.model.sidecar.speculation_head(last_hidden))

                # Synchronous publication: no producer's block-m write enters a
                # sibling's block-m context.
                for stream_idx, write in enumerate(block_writes):
                    if (
                        intervention.mode == "bus_mutation"
                        and stream_names[stream_idx] == intervention.mutation_producer
                        and block_idx == intervention.mutation_block
                    ):
                        write = apply_bus_mutation(
                            write,
                            magnitude=intervention.mutation_magnitude,
                        )
                    snapshots_by_stream[stream_idx].append(write)
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
        evaluator = PairedCausalEvaluator()
        try:
            self.model.eval()
            if trunk_model is not None:
                trunk_model.eval()
            _require_cache_compatible_trunk(self.model.trunk_adapter)
            for batch_idx, raw_batch in enumerate(self._eval_loader):
                batch = _to_device(raw_batch, self.device)
                baseline = self._student_rollout(batch)
                gate_zero = self._student_rollout(
                    batch,
                    intervention=_RolloutIntervention(mode="gate_zero"),
                )
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
                        mutation_magnitude=(self.config.training.causal_eval_mutation_magnitude),
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
                evaluator.update(
                    baseline_logits=baseline.lm_logits,
                    gate_zero_logits=gate_zero.lm_logits,
                    norm_scramble_logits=norm_scramble.lm_logits,
                    mutation_logits=mutation.lm_logits,
                    labels=baseline.lm_labels,
                    label_mask=baseline.lm_label_mask.bool(),
                    dependency_mask=baseline.dependency_mask.bool(),
                    nondependency_mask=baseline.nondependency_mask.bool(),
                    mutation_dependency_mask=mutation_dependency_mask,
                )

            stats = self.codebook.compute()
            metrics = {
                "global_step": self.global_step,
                "stage": self.curriculum.current_stage,
                "codebook": stats.to_dict(),
                "codebook_passes_stage0_gate": stats.passes_stage0_gate(),
                "active_modules": self.curriculum.active_modules_snapshot(),
                "causal": evaluator.compute().to_dict(),
            }
            (self.telemetry_dir / f"eval_{self.global_step:07d}.json").write_text(
                json.dumps(metrics, indent=2)
            )
            LOGGER.info(
                "eval@step=%d stage=%d codebook=%s gate=%s causal=%s",
                self.global_step,
                self.curriculum.current_stage,
                stats.to_dict(),
                stats.passes_stage0_gate(),
                metrics["causal"],
            )
            self.codebook.reset()
        finally:
            _clear_runtime_contexts(self.model.instrumented_layers)
            self.model.train(model_was_training)
            if trunk_model is not None and trunk_was_training is not None:
                trunk_model.train(trunk_was_training)

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


def _validate_rollout_intervention(
    intervention: _RolloutIntervention,
    *,
    streams: tuple[str, ...],
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
    if intervention.mode != "bus_mutation":
        return
    if intervention.mutation_producer not in streams:
        raise ValueError(
            "Bus mutation producer must name a configured runtime stream; "
            f"got {intervention.mutation_producer!r}, expected one of {streams}."
        )
    if type(intervention.mutation_block) is not int or intervention.mutation_block < 0:
        raise ValueError("Bus mutation block must be a non-negative integer.")
    if not math.isfinite(intervention.mutation_magnitude) or intervention.mutation_magnitude == 0.0:
        raise ValueError("Bus mutation magnitude must be finite and non-zero.")


def _rollout_layer_context(
    snapshots_by_stream: list[list[torch.Tensor]],
    *,
    stream: str,
    consumer: int,
    block_idx: int,
    lag: int,
    intervention: _RolloutIntervention,
    scramble_generator: torch.Generator | None,
) -> LayerRuntimeContext:
    """Build one frozen block context and apply only the selected intervention."""

    notes, notes_mask = _visible_notes(
        snapshots_by_stream,
        consumer=consumer,
        block_idx=block_idx,
        lag=lag,
    )
    if intervention.mode == "norm_scramble":
        if scramble_generator is None:
            raise RuntimeError("Norm scramble requires an initialized CPU generator.")
        producer_count = len(snapshots_by_stream)
        sibling_dynamic_slots = torch.zeros(
            (2 * producer_count,),
            dtype=torch.bool,
            device=notes.device,
        )
        sibling_dynamic_slots[producer_count:] = True
        sibling_dynamic_slots[producer_count + consumer] = False
        notes = apply_norm_scramble(
            notes,
            generator=scramble_generator,
            slot_mask=sibling_dynamic_slots,
        )
    return LayerRuntimeContext(
        stream=stream,
        notes=notes,
        notes_mask=notes_mask,
        snc_force_gate=False if intervention.mode == "gate_zero" else None,
    )


def _visible_notes(
    snapshots_by_stream: list[list[torch.Tensor]],
    *,
    consumer: int,
    block_idx: int,
    lag: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the runtime-equivalent fixed 2K anchor/LWW training window."""

    del consumer
    if block_idx < 0:
        raise ValueError("block_idx must be non-negative.")
    if lag < 0:
        raise ValueError("lag must be non-negative.")
    if not snapshots_by_stream or any(not snapshots for snapshots in snapshots_by_stream):
        raise ValueError("Every producer must provide one prompt anchor.")

    anchors = [snapshots[0] for snapshots in snapshots_by_stream]
    reference = anchors[0]
    if reference.ndim != 2:
        raise ValueError("Training notes must have shape [batch, notes_dim].")
    expected_shape = reference.shape
    if any(note.shape != expected_shape for note in anchors):
        raise ValueError("All producer anchors must share [batch, notes_dim] shape.")

    eligible_block = block_idx - lag
    dynamics: list[torch.Tensor] = []
    dynamic_present: list[bool] = []
    for snapshots in snapshots_by_stream:
        dynamic_index = eligible_block + 1  # index 0 is the prompt anchor
        if eligible_block >= 0 and dynamic_index < len(snapshots):
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
