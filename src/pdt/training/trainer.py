"""Canonical packed trainer for source-grounded three-lane PDT."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
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
from pdt.config.schemas import LossWeights, PDTConfig
from pdt.diagnostics.architecture import architecture_telemetry
from pdt.evaluation.paired_causal import bootstrap_mean
from pdt.model import PDTModel
from pdt.runtime.state import (
    PackedAppend,
    PackedFrontierState,
    PackedTokenRows,
)
from pdt.sidecar.heads.planner import PlannerOutput
from pdt.training.curriculum import CurriculumController
from pdt.training.dataset import RealPlanCollator, RealPlanDataset, SampleBatch
from pdt.training.losses import (
    LossBundle,
    PermutationMatch,
    compute_pdt_losses,
    match_unordered_plans,
)
from pdt.trunk.instrumentation import LayerRuntimeContext


LOGGER = logging.getLogger("pdt.training.trainer")

__all__ = ["PDTTrainer"]


@dataclass(slots=True)
class _PackedRollout:
    lm_ce: torch.Tensor
    block_hidden: torch.Tensor
    block_token_nll: torch.Tensor
    block_token_ce_sum: torch.Tensor
    block_token_count: torch.Tensor
    dynamic_vq_commitment_loss: torch.Tensor
    dynamic_vq_codebook_loss: torch.Tensor
    dynamic_vq_logits: torch.Tensor
    note_queries: torch.Tensor
    note_keys: torch.Tensor


@dataclass(slots=True)
class _StepOutput:
    losses: LossBundle
    rollout: Optional[_PackedRollout]
    plan_nodes: torch.Tensor
    plan_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class _CausalEffect:
    dependency_delta_nats: float
    dependency_tokens: int
    nondependency_delta_nats: float
    nondependency_tokens: int

    @property
    def dependency_delta_ce(self) -> float:
        return self.dependency_delta_nats / self.dependency_tokens

    @property
    def nondependency_delta_ce(self) -> float:
        return self.nondependency_delta_nats / self.nondependency_tokens

    @property
    def selectivity_delta_ce(self) -> float:
        return self.dependency_delta_ce - self.nondependency_delta_ce

    def to_dict(self) -> dict[str, float | int]:
        return {
            "dependency_delta_nats": self.dependency_delta_nats,
            "dependency_tokens": self.dependency_tokens,
            "nondependency_delta_nats": self.nondependency_delta_nats,
            "nondependency_tokens": self.nondependency_tokens,
            "dependency_delta_ce": self.dependency_delta_ce,
            "nondependency_delta_ce": self.nondependency_delta_ce,
            "selectivity_delta_ce": self.selectivity_delta_ce,
        }


class PDTTrainer:
    """Train every lane in one physical `[B*3, 32]` frontier per block."""

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
            raise ValueError("Trainer and model coordination_source must match.")
        if config.training.batch_size != 1:
            raise ValueError(
                "Canonical packed recurrent training requires batch_size=1; "
                "use gradient accumulation for the effective batch."
            )
        self.telemetry_dir = Path(
            telemetry_dir or config.training.telemetry_dir
        ).resolve()
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._resolve_device()
        self.model.to(self.device)
        trunk_model: torch.nn.Module = self.model.trunk_adapter.model
        trunk_model.to(self.device)
        trunk_model.eval()
        self._require_cache_compatible_trunk()
        pad_token_id = self.model.trunk_adapter.tokenizer.pad_token_id
        if type(pad_token_id) is not int or pad_token_id < 0:
            raise ValueError("The frozen trunk tokenizer must define pad_token_id.")
        self.pad_token_id = pad_token_id
        self.curriculum = CurriculumController(model, config)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.global_step = 0
        self._train_loader: Optional[DataLoader] = None
        self._eval_loader: Optional[DataLoader] = None

    def _resolve_device(self) -> torch.device:
        if self.config.training.device:
            return torch.device(self.config.training.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        parameters = list(self.model.all_trainable_parameters())
        if not parameters:
            raise RuntimeError("PDT exposes no trainable phi parameters.")
        return AdamW(
            parameters,
            lr=self.config.training.optimizer.learning_rate,
            weight_decay=self.config.training.optimizer.weight_decay,
            betas=(0.9, 0.95),
        )

    def _build_scheduler(self) -> LambdaLR:
        optimizer_config = self.config.training.optimizer
        maximum = self.config.training.max_steps
        warmup = optimizer_config.warmup_steps

        def multiplier(step: int) -> float:
            if step < warmup:
                return max(step, 1) / max(warmup, 1)
            if optimizer_config.lr_scheduler == "constant":
                return 1.0
            progress = (step - warmup) / max(1, maximum - warmup)
            if optimizer_config.lr_scheduler == "linear":
                return max(0.0, 1.0 - progress)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        return LambdaLR(self.optimizer, lr_lambda=multiplier)

    def _build_collator(self) -> RealPlanCollator:
        return RealPlanCollator(
            pad_token_id=self.pad_token_id,
            max_planner_prompt_length=self.config.training.max_planner_prompt_length,
            seed=self.config.training.seed,
        )

    def _build_dataloader(self, path: str, *, shuffle: bool) -> DataLoader:
        dataset = RealPlanDataset(
            path,
            expected_tokenizer=self.config.trunk.base_model,
            expected_tokenizer_revision=self.config.trunk.revision,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            collate_fn=self._build_collator(),
            num_workers=0,
            pin_memory=self.device.type == "cuda",
            drop_last=shuffle,
        )

    def resume_from_checkpoint(self, path: Path) -> CheckpointMetadata:
        metadata = resume_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.scheduler,
        )
        expected_stage = self.curriculum.determine_stage(metadata.global_step)
        if metadata.stage != expected_stage:
            raise CheckpointMismatchError(
                "Checkpoint curriculum stage does not match global_step: "
                f"saved={metadata.stage}, expected={expected_stage}."
            )
        self.global_step = metadata.global_step
        if self.curriculum.on_step(self.global_step) != metadata.stage:
            raise RuntimeError("Curriculum state restoration failed.")
        return metadata

    def train(self) -> None:
        self._train_loader = self._build_dataloader(
            self.config.training.dataset_path,
            shuffle=True,
        )
        self._eval_loader = self._build_dataloader(
            self.config.training.eval_dataset_path,
            shuffle=False,
        )
        iterator = _infinite(self._train_loader)
        accumulation = 0
        started = time.time()
        self.model.train()
        self.model.trunk_adapter.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        while self.global_step < self.config.training.max_steps:
            stage = self.curriculum.on_step(self.global_step)
            step_output = self._step(next(iterator), stage=stage, backward=True)
            losses = step_output.losses
            accumulation += 1
            if accumulation < self.config.training.grad_accumulation:
                continue
            self._optimizer_step()
            accumulation = 0
            self.global_step += 1
            self.curriculum.on_step(self.global_step)
            if self.global_step % self.config.training.log_interval == 0:
                LOGGER.info(
                    "step=%d stage=%d elapsed=%.1fs losses=%s",
                    self.global_step,
                    stage,
                    time.time() - started,
                    {name: round(value, 5) for name, value in losses.to_dict().items()},
                )
            if self.global_step % self.config.training.save_every == 0:
                self._save_checkpoint()
            if self.global_step % self.config.training.eval_interval == 0:
                self.evaluate()
        if self.global_step % self.config.training.save_every:
            self._save_checkpoint()
        if self.global_step % self.config.training.eval_interval:
            self.evaluate()

    def evaluate(self) -> dict[str, object]:
        loader = self._eval_loader or self._build_dataloader(
            self.config.training.eval_dataset_path,
            shuffle=False,
        )
        stage = self.curriculum.on_step(self.global_step)
        totals: dict[str, float] = {}
        causal_totals = {
            "dependency_delta_nats": 0.0,
            "dependency_tokens": 0,
            "nondependency_delta_nats": 0.0,
            "nondependency_tokens": 0,
        }
        causal_documents: list[dict[str, object]] = []
        causal_effects: list[_CausalEffect] = []
        count = 0
        with torch.no_grad():
            for batch in loader:
                step_output = self._step(batch, stage=stage, backward=False)
                losses = step_output.losses
                for name, value in losses.to_dict().items():
                    totals[name] = totals.get(name, 0.0) + value
                causal = self._dynamic_note_causal_effect(
                    _to_device(batch, self.device),
                    step_output=step_output,
                )
                if causal is not None:
                    causal_effects.append(causal)
                    causal_totals["dependency_delta_nats"] += (
                        causal.dependency_delta_nats
                    )
                    causal_totals["dependency_tokens"] += causal.dependency_tokens
                    causal_totals["nondependency_delta_nats"] += (
                        causal.nondependency_delta_nats
                    )
                    causal_totals["nondependency_tokens"] += (
                        causal.nondependency_tokens
                    )
                    causal_documents.append(
                        {
                            "example_id": batch.example_ids[0],
                            **causal.to_dict(),
                        }
                    )
                count += 1
        if count == 0:
            raise ValueError("Evaluation dataset is empty.")
        metrics: dict[str, object] = {
            "global_step": self.global_step,
            "stage": stage,
            "documents": count,
            "losses": {name: value / count for name, value in totals.items()},
            "architecture": architecture_telemetry(self.model),
        }
        if causal_documents:
            dependency_tokens = int(causal_totals["dependency_tokens"])
            nondependency_tokens = int(causal_totals["nondependency_tokens"])
            if dependency_tokens <= 0 or nondependency_tokens <= 0:
                raise RuntimeError(
                    "Causal evaluation requires both dependency and nondependency tokens."
                )
            dependency_delta = (
                causal_totals["dependency_delta_nats"] / dependency_tokens
            )
            nondependency_delta = (
                causal_totals["nondependency_delta_nats"] / nondependency_tokens
            )
            dependency_interval = bootstrap_mean(
                [effect.dependency_delta_ce for effect in causal_effects],
                samples=self.config.training.causal_eval_bootstrap_samples,
                confidence_level=self.config.training.causal_eval_confidence_level,
                seed=self.config.training.causal_eval_seed,
            )
            selectivity_interval = bootstrap_mean(
                [effect.selectivity_delta_ce for effect in causal_effects],
                samples=self.config.training.causal_eval_bootstrap_samples,
                confidence_level=self.config.training.causal_eval_confidence_level,
                seed=self.config.training.causal_eval_seed + 1,
            )
            enough_documents = (
                len(causal_documents)
                >= self.config.training.causal_eval_min_documents
            )
            metrics["dynamic_note_ablation"] = {
                "dependency_delta_ce": dependency_delta,
                "nondependency_delta_ce": nondependency_delta,
                "selectivity_delta_ce": dependency_delta - nondependency_delta,
                "dependency_tokens": dependency_tokens,
                "nondependency_tokens": nondependency_tokens,
                "dependency_document_bootstrap": dependency_interval.to_dict(),
                "selectivity_document_bootstrap": selectivity_interval.to_dict(),
                "evidence_gate": {
                    "minimum_documents": (
                        self.config.training.causal_eval_min_documents
                    ),
                    "enough_documents": enough_documents,
                    "dependency_ci_lower_positive": dependency_interval.lower > 0,
                    "selectivity_ci_lower_positive": selectivity_interval.lower > 0,
                    "passes": (
                        enough_documents
                        and dependency_interval.lower > 0
                        and selectivity_interval.lower > 0
                    ),
                },
                "documents": causal_documents,
            }
        destination = self.telemetry_dir / f"evaluation_step_{self.global_step:08d}.json"
        destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        LOGGER.info("Evaluation completed: %s", metrics["losses"])
        return metrics

    def optimizer_probe(self) -> dict[str, object]:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The optimizer probe requires a visible CUDA device.")
        if self.config.training.grad_accumulation != 1 or self.global_step != 0:
            raise ValueError(
                "The optimizer probe requires grad_accumulation=1 and a fresh step-0 model."
            )
        loader = self._build_dataloader(
            self.config.training.dataset_path,
            shuffle=False,
        )
        iterator = _infinite(loader)
        reports: list[dict[str, object]] = []
        torch.cuda.reset_peak_memory_stats(self.device)
        for probe_index in range(2):
            stage = self.curriculum.on_step(self.global_step)
            before = _trainable_snapshot(self.model)
            step_output = self._step(next(iterator), stage=stage, backward=True)
            losses = step_output.losses
            gradient_l1 = _finite_gradient_l1(self.model)
            self._optimizer_step()
            movement = _parameter_movement(self.model, before)
            if probe_index == 1 and movement <= 0:
                raise RuntimeError("Optimizer probe found no trainable phi parameter movement.")
            reports.append(
                {
                    "probe_index": probe_index,
                    "losses": losses.to_dict(),
                    "gradient_l1": gradient_l1,
                    "parameter_l1_movement": movement,
                }
            )
            self.global_step += 1
            self.curriculum.on_step(self.global_step)
        result: dict[str, object] = {
            "optimizer_steps": self.global_step,
            "reports": reports,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
            "device": torch.cuda.get_device_name(self.device),
        }
        destination = self.telemetry_dir / "optimizer_probe.json"
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
        self._save_checkpoint()
        return result

    def _step(
        self,
        batch: SampleBatch,
        *,
        stage: int,
        backward: bool,
    ) -> _StepOutput:
        batch = _to_device(batch, self.device)
        weights = self.curriculum.active_loss_weights(stage)
        planner: Optional[PlannerOutput] = None
        match: Optional[PermutationMatch] = None
        if stage != 0:
            planner = self._planner_forward(batch)
            match = match_unordered_plans(
                predicted_nodes=planner.nodes,
                validity_logits=planner.node_validity_logits,
                presentation_order_logits=planner.presentation_order_logits,
                teacher_nodes=batch.plan_semantic_targets,
                teacher_node_mask=batch.plan_node_mask,
            )
            plan_nodes = match.nodes
            plan_semantic_loss: Optional[torch.Tensor] = match.semantic_loss
            presentation_logits: Optional[torch.Tensor] = (
                match.presentation_order_logits
            )
        else:
            plan_nodes = batch.plan_semantic_targets * math.sqrt(
                batch.plan_semantic_targets.size(-1)
            )
            plan_semantic_loss = None
            presentation_logits = None
        plan_mask = batch.plan_node_mask
        route_logits = self.model.sidecar.semantic_heads.fact_route_logits(
            plan_nodes,
            batch.fact_embeddings,
        )

        rollout: Optional[_PackedRollout] = None
        if _requires_rollout(weights):
            rollout = self._packed_rollout(
                batch,
                plan_nodes=plan_nodes,
                plan_mask=plan_mask,
            )
            progress_logits = (
                self.model.sidecar.semantic_heads.outline_progress_logits(
                    rollout.block_hidden,
                    plan_nodes,
                )
            )
            fact_write_logits = self.model.sidecar.semantic_heads.fact_write_logits(
                rollout.block_hidden,
                batch.fact_embeddings,
            )
        else:
            progress_logits = None
            fact_write_logits = None

        losses = compute_pdt_losses(
            weights=weights,
            lm_ce=None if rollout is None else rollout.lm_ce,
            plan_semantic=plan_semantic_loss,
            fact_route_logits=route_logits,
            fact_route_targets=batch.fact_route_targets,
            plan_node_mask=plan_mask,
            fact_mask=batch.fact_mask,
            outline_progress_logits=progress_logits,
            outline_progress_targets=(
                None
                if rollout is None
                else batch.outline_progress_targets[
                    :, :, : rollout.block_hidden.size(2)
                ]
            ),
            fact_write_logits=fact_write_logits,
            fact_write_targets=(
                None
                if rollout is None
                else batch.fact_write_targets[
                    :, :, : rollout.block_hidden.size(2)
                ]
            ),
            note_queries=None if rollout is None else rollout.note_queries,
            note_keys=None if rollout is None else rollout.note_keys,
            presentation_order_logits=presentation_logits,
            presentation_rank_targets=(
                batch.presentation_rank_targets if presentation_logits is not None else None
            ),
            dynamic_vq_commitment_loss=(
                None if rollout is None else rollout.dynamic_vq_commitment_loss
            ),
            dynamic_vq_codebook_loss=(
                None if rollout is None else rollout.dynamic_vq_codebook_loss
            ),
            dynamic_vq_logits=(
                None if rollout is None else rollout.dynamic_vq_logits
            ),
        )
        if backward:
            (losses.total / self.config.training.grad_accumulation).backward()
        return _StepOutput(
            losses=losses,
            rollout=rollout,
            plan_nodes=plan_nodes,
            plan_mask=plan_mask,
        )

    def _dynamic_note_causal_effect(
        self,
        batch: SampleBatch,
        *,
        step_output: _StepOutput,
    ) -> Optional[_CausalEffect]:
        """Measure dynamic-note removal while preserving the static plan path."""

        if self.config.instrumentation.coordination_source != "bus":
            return None
        normal = step_output.rollout
        if normal is None:
            normal = self._packed_rollout(
                batch,
                plan_nodes=step_output.plan_nodes,
                plan_mask=step_output.plan_mask,
            )
        without_dynamic = self._packed_rollout(
            batch,
            plan_nodes=step_output.plan_nodes,
            plan_mask=step_output.plan_mask,
            dynamic_notes_enabled=False,
        )
        if normal.block_token_nll.shape != without_dynamic.block_token_nll.shape:
            raise RuntimeError("Normal and no-dynamic packed rollouts are misaligned.")
        blocks = normal.block_token_nll.size(2)
        active = batch.target_block_attention_mask[:, :, :blocks]
        dependency = batch.dependency_token_mask[:, :, :blocks] & active
        nondependency = ~batch.dependency_token_mask[:, :, :blocks] & active
        if not bool(dependency.any()) or not bool(nondependency.any()):
            raise ValueError(
                "Every real-plan evaluation example requires dependency and "
                "nondependency target tokens."
            )
        delta = without_dynamic.block_token_nll - normal.block_token_nll
        return _CausalEffect(
            dependency_delta_nats=float(
                delta[dependency].sum().detach().float().item()
            ),
            dependency_tokens=int(
                dependency.sum().item()
            ),
            nondependency_delta_nats=float(
                delta[nondependency].sum().detach().float().item()
            ),
            nondependency_tokens=int(
                nondependency.sum().item()
            ),
        )

    def _planner_forward(self, batch: SampleBatch) -> PlannerOutput:
        self._clear_runtime_context()
        with torch.no_grad():
            prompt_output = self.model.trunk_adapter.forward(
                input_ids=batch.planner_prompt_ids,
                attention_mask=batch.planner_prompt_attention_mask,
                use_cache=False,
                output_hidden_states=True,
            )
        if prompt_output.hidden_states is None:
            raise RuntimeError("Planner prompt forward omitted hidden states.")
        return self.model.sidecar.planner_head(
            prompt_output.hidden_states[-1],
            attention_mask=batch.planner_prompt_attention_mask,
        )

    def _packed_rollout(
        self,
        batch: SampleBatch,
        *,
        plan_nodes: torch.Tensor,
        plan_mask: torch.Tensor,
        dynamic_notes_enabled: bool = True,
    ) -> _PackedRollout:
        batch_size, lanes, blocks, block_width = batch.target_block_ids.shape
        if batch_size != 1 or lanes != 3 or block_width != self.config.runtime.block_size:
            raise ValueError(
                "Packed rollout requires [B=1, K=3, M, tau=32] target blocks."
            )
        streams = tuple(self.config.runtime.streams)
        plan_memory = self.model.sidecar.plan_memory_proj(plan_nodes, plan_mask)
        dynamic_notes: list[torch.Tensor] = []
        dynamic_validity: list[torch.Tensor] = []
        self_only_states: list[torch.Tensor] = []
        self_only_validity: list[torch.Tensor] = []
        self_only_positions: list[torch.Tensor] = []
        block_hidden: list[torch.Tensor] = []
        block_token_nll: list[torch.Tensor] = []
        block_token_ce_sums: list[torch.Tensor] = []
        block_token_counts: list[torch.Tensor] = []
        commitment_terms: list[tuple[torch.Tensor, int]] = []
        codebook_terms: list[tuple[torch.Tensor, int]] = []
        assignment_logits: list[torch.Tensor] = []
        note_queries: list[torch.Tensor] = []
        note_keys: list[torch.Tensor] = []
        ce_sum = plan_nodes.new_zeros((), dtype=torch.float32)
        ce_count = 0

        repeated_prompt_ids = batch.planner_prompt_ids.repeat_interleave(lanes, dim=0)
        repeated_prompt_mask = batch.planner_prompt_attention_mask.repeat_interleave(
            lanes,
            dim=0,
        )
        prompt_positions = repeated_prompt_mask.long().cumsum(dim=1) - 1
        prompt_positions.masked_fill_(~repeated_prompt_mask, 0)
        prompt_cache_position = torch.arange(
            repeated_prompt_ids.size(1),
            device=self.device,
            dtype=torch.long,
        )
        self._set_runtime_context(
            self._packed_layer_context(
                plan_nodes=plan_nodes,
                plan_memory=plan_memory,
                plan_mask=plan_mask,
                dynamic_notes=dynamic_notes,
                dynamic_validity=dynamic_validity,
                consumer_block=0,
                self_only_states=self_only_states,
                self_only_validity=self_only_validity,
                self_only_positions=self_only_positions,
                query_positions=prompt_positions,
                dynamic_notes_enabled=dynamic_notes_enabled,
            )
        )
        prefill = self.model.trunk_adapter.forward(
            input_ids=repeated_prompt_ids,
            attention_mask=repeated_prompt_mask,
            position_ids=prompt_positions,
            cache_position=prompt_cache_position,
            use_cache=True,
            output_hidden_states=False,
        )
        if prefill.past_key_values is None:
            raise RuntimeError("Packed training prefill dropped its KV cache.")
        prompt_last = repeated_prompt_mask.sum(dim=1, dtype=torch.long) - 1
        row_index = torch.arange(lanes, device=self.device)
        next_logits = prefill.logits[row_index, prompt_last]
        frontier = PackedFrontierState(
            streams=streams,
            attention_mask=repeated_prompt_mask,
            past_key_values=prefill.past_key_values,
        )

        try:
            for block_index in range(blocks):
                token_ids = batch.target_block_ids[0, :, block_index]
                valid = batch.target_block_attention_mask[0, :, block_index].bool()
                active_rows = valid.any(dim=1)
                if not bool(active_rows.any()):
                    break
                prior_lengths = frontier.logical_lengths
                positions = prior_lengths[:, None] + valid.cumsum(dim=1).long() - 1
                positions.masked_fill_(~valid, 0)
                self_only_query_positions = prior_lengths[:, None] + torch.arange(
                    block_width,
                    device=self.device,
                    dtype=torch.long,
                )
                cache_positions = torch.arange(
                    frontier.physical_length,
                    frontier.physical_length + block_width,
                    device=self.device,
                    dtype=torch.long,
                )
                rows = PackedTokenRows(
                    input_ids=token_ids,
                    valid_mask=valid,
                    position_ids=positions,
                    cache_position=cache_positions,
                )
                append = PackedAppend(
                    rows=rows,
                    attention_mask=torch.cat((frontier.attention_mask, valid), dim=1),
                )
                self._set_runtime_context(
                    self._packed_layer_context(
                        plan_nodes=plan_nodes,
                        plan_memory=plan_memory,
                        plan_mask=plan_mask,
                        dynamic_notes=dynamic_notes,
                        dynamic_validity=dynamic_validity,
                        consumer_block=block_index,
                        self_only_states=self_only_states,
                        self_only_validity=self_only_validity,
                        self_only_positions=self_only_positions,
                        query_positions=self_only_query_positions,
                        dynamic_notes_enabled=dynamic_notes_enabled,
                    )
                )
                output = self.model.trunk_adapter.forward(
                    input_ids=token_ids,
                    attention_mask=append.attention_mask,
                    past_key_values=frontier.past_key_values,
                    position_ids=positions,
                    cache_position=cache_positions,
                    use_cache=True,
                    output_hidden_states=True,
                )
                if output.past_key_values is None or output.hidden_states is None:
                    raise RuntimeError(
                        f"Packed block {block_index} omitted cache or hidden states."
                    )
                frontier.commit(append, past_key_values=output.past_key_values)
                aligned_logits = torch.cat(
                    (next_logits.unsqueeze(1), output.logits[:, :-1]),
                    dim=1,
                )
                token_losses = F.cross_entropy(
                    aligned_logits.float().reshape(
                        lanes * block_width,
                        aligned_logits.size(-1),
                    ),
                    token_ids.reshape(lanes * block_width).long(),
                    reduction="none",
                ).reshape(lanes, block_width)
                row_ce_sum = (token_losses * valid).sum(dim=1)
                row_token_count = valid.sum(dim=1, dtype=torch.long)
                block_token_nll.append(
                    (token_losses * valid).reshape(
                        batch_size,
                        lanes,
                        block_width,
                    )
                )
                block_token_ce_sums.append(
                    row_ce_sum.reshape(batch_size, lanes)
                )
                block_token_counts.append(
                    row_token_count.reshape(batch_size, lanes)
                )
                ce_sum = ce_sum + row_ce_sum.sum()
                ce_count += int(row_token_count.sum().item())

                valid_counts = valid.sum(dim=1, dtype=torch.long)
                last_index = (valid_counts - 1).clamp_min(0)
                final_hidden = output.hidden_states[-1][row_index, last_index]
                final_hidden = final_hidden * active_rows.to(
                    dtype=final_hidden.dtype
                ).unsqueeze(-1)
                block_hidden.append(final_hidden.reshape(batch_size, lanes, -1))
                final_positions = positions[row_index, last_index]
                final_positions = final_positions.masked_fill(~active_rows, -1)
                self_only_states.append(
                    final_hidden.reshape(batch_size, lanes, -1)
                )
                self_only_validity.append(
                    active_rows.reshape(batch_size, lanes)
                )
                self_only_positions.append(
                    final_positions.reshape(batch_size, lanes)
                )
                candidate_next = output.logits[row_index, last_index]
                next_logits = torch.where(
                    active_rows.unsqueeze(-1),
                    candidate_next,
                    next_logits,
                )

                active_index = torch.nonzero(active_rows, as_tuple=False).flatten()
                active_hidden = final_hidden.index_select(0, active_index)
                projected = self.model.sidecar.speculation_head.project(active_hidden)
                quantized = self.model.sidecar.speculation_head.quantize(projected)
                note_flat = final_hidden.new_zeros(
                    lanes,
                    self.config.sidecar.notes_dim,
                    dtype=quantized.quantized.dtype,
                )
                note_flat = note_flat.index_copy(0, active_index, quantized.quantized)
                dynamic_notes.append(note_flat.reshape(batch_size, lanes, -1))
                dynamic_validity.append(active_rows.reshape(batch_size, lanes))
                active_count = int(active_index.numel())
                commitment_terms.append((quantized.commitment_loss, active_count))
                codebook_terms.append((quantized.codebook_loss, active_count))
                assignment_logits.append(quantized.assignment_logits)
                note_queries.append(quantized.quantized)
                note_keys.append(quantized.pre_quantized)
        finally:
            self._clear_runtime_context()

        if ce_count <= 0 or not block_hidden or not note_queries:
            raise RuntimeError("Packed rollout produced no active target tokens or notes.")
        return _PackedRollout(
            lm_ce=ce_sum / ce_count,
            block_hidden=torch.stack(block_hidden, dim=2),
            block_token_nll=torch.stack(block_token_nll, dim=2),
            block_token_ce_sum=torch.stack(block_token_ce_sums, dim=2),
            block_token_count=torch.stack(block_token_counts, dim=2),
            dynamic_vq_commitment_loss=_weighted_mean(commitment_terms),
            dynamic_vq_codebook_loss=_weighted_mean(codebook_terms),
            dynamic_vq_logits=torch.cat(assignment_logits, dim=0),
            note_queries=torch.cat(note_queries, dim=0),
            note_keys=torch.cat(note_keys, dim=0),
        )

    def _packed_layer_context(
        self,
        *,
        plan_nodes: torch.Tensor,
        plan_memory: torch.Tensor,
        plan_mask: torch.Tensor,
        dynamic_notes: list[torch.Tensor],
        dynamic_validity: list[torch.Tensor],
        consumer_block: int,
        self_only_states: list[torch.Tensor],
        self_only_validity: list[torch.Tensor],
        self_only_positions: list[torch.Tensor],
        query_positions: torch.Tensor,
        dynamic_notes_enabled: bool,
    ) -> LayerRuntimeContext:
        batch, lanes, nodes, _ = plan_nodes.shape
        if batch != 1 or lanes != len(self.config.runtime.streams):
            raise ValueError("Layer context requires one complete physical frontier.")
        lane_ids = torch.arange(lanes, device=self.device).view(1, lanes, 1)
        plan_producers = lane_ids.expand(batch, lanes, nodes).reshape(
            batch * lanes,
            nodes,
        )
        stream_ids = tuple(self.config.runtime.streams)
        packed_plan_nodes = plan_nodes.reshape(batch * lanes, nodes, -1)
        packed_plan_mask = plan_mask.reshape(batch * lanes, nodes)
        packed_plan_memory = plan_memory.reshape(batch * lanes, nodes, -1)
        if self.config.instrumentation.coordination_source == "self_only":
            if not dynamic_notes_enabled:
                raise ValueError(
                    "dynamic_notes_enabled=False is a bus-only causal intervention."
                )
            trunk_dtype = self.model.trunk_adapter.frozen_parameters()[0].dtype
            memory = _self_only_window(
                self_only_states,
                self_only_validity,
                self_only_positions,
                consumer_block=consumer_block,
                lanes=lanes,
                history_blocks=self.config.runtime.notes_bus.history_blocks,
                hidden_size=self.config.sidecar.snc.hidden_size,
                streams=tuple(self.config.runtime.streams),
                device=self.device,
                dtype=trunk_dtype,
            )
            return LayerRuntimeContext(
                stream_ids=stream_ids,
                plan_nodes=packed_plan_nodes,
                plan_mask=packed_plan_mask,
                plan_memory=packed_plan_memory,
                plan_producer_ids=plan_producers,
                self_only_memory=memory,
                self_only_query_positions=query_positions,
            )
        notes, notes_mask, producers, lags = _dynamic_window(
            dynamic_notes,
            dynamic_validity,
            consumer_block=consumer_block,
            producers=lanes,
            history_blocks=self.config.runtime.notes_bus.history_blocks,
            notes_dim=self.config.sidecar.notes_dim,
            device=self.device,
        )
        if not dynamic_notes_enabled:
            notes_mask = torch.zeros_like(notes_mask)
        return LayerRuntimeContext(
            stream_ids=stream_ids,
            plan_nodes=packed_plan_nodes,
            plan_mask=packed_plan_mask,
            plan_memory=packed_plan_memory,
            plan_producer_ids=plan_producers,
            notes=notes,
            notes_mask=notes_mask,
            note_producer_ids=producers,
            note_kind_ids=torch.ones_like(producers),
            note_lags=lags,
        )

    def _set_runtime_context(self, context: LayerRuntimeContext) -> None:
        for layer in self.model.instrumented_layers:
            layer.set_runtime_context(context)

    def _clear_runtime_context(self) -> None:
        for layer in self.model.instrumented_layers:
            layer.set_runtime_context(None)

    def _optimizer_step(self) -> None:
        parameters = [
            parameter
            for parameter in self.model.all_trainable_parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("Curriculum exposed no trainable phi parameters.")
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _save_checkpoint(self) -> None:
        stage = self.curriculum.determine_stage(self.global_step)
        destination = (
            self.telemetry_dir
            / "checkpoints"
            / f"step_{self.global_step:08d}.pt"
        )
        save_checkpoint(
            destination,
            self.model,
            self.optimizer,
            self.scheduler,
            global_step=self.global_step,
            stage=stage,
        )
        LOGGER.info("Saved checkpoint %s.", destination)

    def _require_cache_compatible_trunk(self) -> None:
        trunk = self.model.trunk_adapter.model
        if getattr(trunk, "is_gradient_checkpointing", False):
            raise RuntimeError(
                "Packed recurrent PDT requires gradient checkpointing to be disabled."
            )
        if not bool(getattr(trunk.config, "use_cache", False)):
            raise RuntimeError("Packed recurrent PDT requires trunk.config.use_cache=true.")


def _requires_rollout(weights: LossWeights) -> bool:
    return any(
        value > 0
        for value in (
            weights.lm_ce,
            weights.outline_progress,
            weights.fact_write,
            weights.note_align,
            weights.dynamic_vq_commit,
            weights.dynamic_vq_codebook,
            weights.dynamic_codebook_usage,
        )
    )


def _dynamic_window(
    notes_by_block: list[torch.Tensor],
    validity_by_block: list[torch.Tensor],
    *,
    consumer_block: int,
    producers: int,
    history_blocks: int,
    notes_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(notes_by_block) != len(validity_by_block):
        raise ValueError("Dynamic note payloads and validity masks must align.")
    slots: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    producer_ids: list[int] = []
    lag_ids: list[int] = []
    for age in range(1, history_blocks + 1):
        source_block = consumer_block - age
        for producer in range(producers):
            if source_block >= 0:
                if source_block >= len(notes_by_block):
                    raise RuntimeError("Dynamic window requested an unpublished source block.")
                slots.append(notes_by_block[source_block][:, producer])
                masks.append(validity_by_block[source_block][:, producer])
            else:
                slots.append(torch.zeros(1, notes_dim, device=device))
                masks.append(torch.zeros(1, dtype=torch.bool, device=device))
            producer_ids.append(producer)
            lag_ids.append(age)
    base_notes = torch.stack(slots, dim=1)
    base_mask = torch.stack(masks, dim=1)
    packed_notes = base_notes[:, None].expand(-1, producers, -1, -1).reshape(
        producers,
        len(slots),
        notes_dim,
    )
    packed_mask = base_mask[:, None].expand(-1, producers, -1).reshape(
        producers,
        len(slots),
    )
    producer_tensor = torch.tensor(
        producer_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0).expand(producers, -1)
    lag_tensor = torch.tensor(
        lag_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0).expand(producers, -1)
    return packed_notes, packed_mask, producer_tensor, lag_tensor


def _self_only_window(
    states_by_block: list[torch.Tensor],
    validity_by_block: list[torch.Tensor],
    positions_by_block: list[torch.Tensor],
    *,
    consumer_block: int,
    lanes: int,
    history_blocks: int,
    hidden_size: int,
    streams: tuple[str, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> SelfOnlyMemory:
    if not (
        len(states_by_block)
        == len(validity_by_block)
        == len(positions_by_block)
        == consumer_block
    ):
        raise ValueError(
            "Self-only block states, validity, positions, and consumer index must align."
        )
    if len(streams) != lanes:
        raise ValueError("Self-only stream addresses must match the physical lane count.")
    start = max(0, consumer_block - history_blocks)
    states = states_by_block[start:consumer_block]
    validity = validity_by_block[start:consumer_block]
    positions = positions_by_block[start:consumer_block]
    if not states:
        return SelfOnlyMemory(
            hidden_states=torch.empty(
                lanes,
                0,
                hidden_size,
                device=device,
                dtype=dtype,
            ),
            mask=torch.empty(lanes, 0, device=device, dtype=torch.bool),
            positions=torch.empty(lanes, 0, device=device, dtype=torch.long),
            slot_ids=torch.empty(lanes, 0, device=device, dtype=torch.long),
            kind_ids=torch.empty(lanes, 0, device=device, dtype=torch.long),
            lags=torch.empty(lanes, 0, device=device, dtype=torch.long),
            owner_streams=streams,
        )
    for block_state, block_validity, block_positions in zip(
        states,
        validity,
        positions,
        strict=True,
    ):
        if block_state.shape != (1, lanes, hidden_size):
            raise ValueError("Self-only block hidden states have an invalid shape.")
        if block_validity.shape != (1, lanes) or block_validity.dtype != torch.bool:
            raise ValueError("Self-only block validity has an invalid shape or dtype.")
        if block_positions.shape != (1, lanes):
            raise ValueError("Self-only block positions have an invalid shape.")
    memory_states = torch.stack(states, dim=2).reshape(
        lanes,
        len(states),
        hidden_size,
    )
    memory_mask = torch.stack(validity, dim=2).reshape(lanes, len(states))
    memory_positions = torch.stack(positions, dim=2).reshape(lanes, len(states))
    lane_ids = torch.arange(lanes, device=device, dtype=torch.long).unsqueeze(1)
    slot_ids = lane_ids.expand(lanes, len(states))
    lag_values = torch.arange(
        len(states),
        0,
        -1,
        device=device,
        dtype=torch.long,
    )
    return SelfOnlyMemory(
        hidden_states=memory_states,
        mask=memory_mask,
        positions=memory_positions,
        slot_ids=slot_ids,
        kind_ids=torch.ones_like(slot_ids),
        lags=lag_values.unsqueeze(0).expand(lanes, -1),
        owner_streams=streams,
    )


def _weighted_mean(values: list[tuple[torch.Tensor, int]]) -> torch.Tensor:
    if not values or any(weight <= 0 for _, weight in values):
        raise ValueError("Weighted dynamic losses require positive observations.")
    total_weight = sum(weight for _, weight in values)
    total = values[0][0].new_zeros(())
    for value, weight in values:
        total = total + value * weight
    return total / total_weight


def _to_device(batch: SampleBatch, device: torch.device) -> SampleBatch:
    return SampleBatch(
        example_ids=batch.example_ids,
        planner_prompt_ids=batch.planner_prompt_ids.to(device),
        planner_prompt_attention_mask=batch.planner_prompt_attention_mask.to(device),
        target_block_ids=batch.target_block_ids.to(device),
        target_block_labels=batch.target_block_labels.to(device),
        target_block_attention_mask=batch.target_block_attention_mask.to(device),
        fact_embeddings=batch.fact_embeddings.to(device),
        fact_mask=batch.fact_mask.to(device),
        positive_fact_mask=batch.positive_fact_mask.to(device),
        plan_semantic_targets=batch.plan_semantic_targets.to(device),
        plan_node_mask=batch.plan_node_mask.to(device),
        fact_route_targets=batch.fact_route_targets.to(device),
        outline_progress_targets=batch.outline_progress_targets.to(device),
        fact_write_targets=batch.fact_write_targets.to(device),
        dependency_token_mask=batch.dependency_token_mask.to(device),
        presentation_rank_targets=batch.presentation_rank_targets.to(device),
        raw=batch.raw,
    )


def _infinite(loader: DataLoader) -> Iterator[SampleBatch]:
    while True:
        yield from loader


def _trainable_snapshot(model: PDTModel) -> dict[int, torch.Tensor]:
    return {
        id(parameter): parameter.detach().float().cpu().clone()
        for parameter in model.all_trainable_parameters()
        if parameter.requires_grad
    }


def _finite_gradient_l1(model: PDTModel) -> float:
    total = 0.0
    active = 0
    for parameter in model.all_trainable_parameters():
        if not parameter.requires_grad:
            continue
        active += 1
        if parameter.grad is None:
            raise RuntimeError("Optimizer probe found a missing active phi gradient.")
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError("Optimizer probe found a non-finite active phi gradient.")
        total += float(parameter.grad.detach().float().abs().sum().item())
    if active == 0 or not math.isfinite(total) or total <= 0:
        raise RuntimeError("Optimizer probe found no finite nonzero phi gradients.")
    return total


def _parameter_movement(
    model: PDTModel,
    before: dict[int, torch.Tensor],
) -> float:
    movement = 0.0
    for parameter in model.all_trainable_parameters():
        initial = before.get(id(parameter))
        if initial is None:
            continue
        movement += float(
            (parameter.detach().float().cpu() - initial).abs().sum().item()
        )
    if not math.isfinite(movement):
        raise RuntimeError("Optimizer probe observed non-finite parameter movement.")
    return movement
