"""Single-GPU / DDP trainer for PDT.

Training loop responsibilities:

1. Per-step curriculum tick (``CurriculumController.on_step``): flip
   requires_grad per stage.
2. Build one batch via ``PDTCollator``.
3. Forward: prompt encode (trunk), VQ planner, planner-seeded bus notes,
   differentiable block rollout with SNC windows, speculation writes.
4. Compute all loss terms; stage-mask via ``compute_pdt_losses``.
5. Backward, clip, step, LR schedule.
6. Periodic eval: codebook diagnostics + ROC for coverage / agreement.

Training mirrors the inference block semantics closely enough for gradients
to flow from receiver LM loss through SNC into visible sibling notes and the
speculation writer that produced them.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader

from pdt.config.schemas import PDTConfig
from pdt.diagnostics.codebook import CodebookDiagnostics
from pdt.model import PDTModel
from pdt.training.curriculum import CurriculumController
from pdt.training.dataset import PDTCollator, PDTDependencyDataset, SampleBatch
from pdt.training.losses import compute_pdt_losses
from pdt.trunk.instrumentation import LayerRuntimeContext


LOGGER = logging.getLogger("pdt.training.trainer")


__all__ = ["PDTTrainer"]


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
        self.telemetry_dir = Path(
            telemetry_dir or config.training.telemetry_dir
        ).resolve()
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._resolve_device()
        # Move both the \u03c6 tree (sidecar + instrumented layers) AND the frozen
        # trunk model to the target device. The trunk is held by composition
        # on a non-Module adapter, so PDTModel.to(device) would not reach it.
        self.model.to(self.device)
        self.model.trunk_adapter.model.to(self.device)

        self.curriculum = CurriculumController(model, config)
        self.codebook = CodebookDiagnostics(
            vocab_size=config.sidecar.plan_vocab_size,
            num_slots=config.sidecar.planner_head.num_slots,
        )

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
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
        pad_id = self.model.trunk_adapter.tokenizer.pad_token_id or 0
        return PDTCollator(
            pad_token_id=pad_id,
            num_streams=self.config.sidecar.num_streams,
            max_shared_length=256,
            max_local_length=128,
            max_blocks=self.config.runtime.notes_bus.max_snapshots,
            max_block_length=self.config.runtime.block_size,
            max_snapshots=self.config.runtime.notes_bus.max_snapshots,
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

    def train(self) -> None:
        self._train_loader = self._build_dataloader(
            self.config.training.dataset_path, shuffle=True
        )
        self._eval_loader = self._build_dataloader(
            self.config.training.eval_dataset_path, shuffle=False
        )
        LOGGER.info(
            "Beginning training: %d train batches, device=%s",
            len(self._train_loader),
            self.device,
        )
        self.model.train()
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
                if self.global_step % self.config.training.log_interval == 0:
                    elapsed = time.time() - t_start
                    LOGGER.info(
                        "step=%d stage=%d loss=%s | %.2fs elapsed",
                        self.global_step,
                        stage,
                        {k: round(v, 4) for k, v in losses_dict.items()
                         if isinstance(v, (int, float))},
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
        B = batch.shared_ids.size(0)
        K = batch.local_ids.size(1)
        M = batch.target_block_ids.size(2)
        block_len = batch.target_block_ids.size(3)
        stream_names = self.config.runtime.streams

        prompt_out = self.model.trunk_adapter.forward(
            input_ids=batch.shared_ids,
            attention_mask=batch.shared_attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        prompt_hidden = prompt_out.hidden_states[-1]
        planner = self.model.sidecar.planner_head(
            prompt_hidden,
            attention_mask=batch.shared_attention_mask.to(prompt_hidden.dtype),
        )
        ownership = _round_robin_ownership(
            batch_size=B,
            num_streams=K,
            num_slots=self.config.sidecar.planner_head.num_slots,
            device=self.device,
        )
        plan_snapshot = self.model.sidecar.plan_notes_proj(planner.quantized, ownership)
        snapshots_by_stream: list[list[torch.Tensor]] = [
            [plan_snapshot[:, k]] for k in range(K)
        ]

        logits_by_block: list[torch.Tensor] = []
        labels_by_block: list[torch.Tensor] = []
        masks_by_block: list[torch.Tensor] = []
        dep_by_block: list[torch.Tensor] = []
        non_by_block: list[torch.Tensor] = []
        hidden_for_classifier: list[torch.Tensor] = []

        for block_idx in range(M):
            block_writes: list[torch.Tensor] = []
            for stream_idx in range(K):
                stream = stream_names[stream_idx]
                notes = _visible_notes(
                    snapshots_by_stream,
                    consumer=stream_idx,
                    block_idx=block_idx,
                    lag=self.config.runtime.notes_bus.lag,
                    max_snapshots=self.config.runtime.notes_bus.max_snapshots,
                )
                notes_mask = torch.ones(
                    (B, notes.size(1)),
                    dtype=torch.bool,
                    device=notes.device,
                )
                ctx = LayerRuntimeContext(
                    stream=stream,
                    notes=notes,
                    notes_mask=notes_mask,
                    snc_force_gate=None,
                )
                for layer in self.model.instrumented_layers:
                    layer.set_runtime_context(ctx)

                ids, attn = _teacher_forced_block_input(batch, stream_idx, block_idx)
                out = self.model.trunk_adapter.forward(
                    input_ids=ids,
                    attention_mask=attn,
                    use_cache=False,
                    output_hidden_states=True,
                )
                current_logits = out.logits[:, -block_len:, :]
                current_hidden = out.hidden_states[-1][:, -block_len:, :]
                current_mask = batch.target_block_attention_mask[:, stream_idx, block_idx].bool()
                logits_by_block.append(current_logits)
                labels_by_block.append(batch.target_block_labels[:, stream_idx, block_idx])
                masks_by_block.append(current_mask)
                dep_by_block.append(batch.dependency_token_mask[:, stream_idx, block_idx])
                non_by_block.append(batch.nondependency_token_mask[:, stream_idx, block_idx])
                hidden_for_classifier.append(_masked_mean_hidden(current_hidden, current_mask))

                spec_tokens = self.model.sidecar.speculation_head(current_hidden)
                block_writes.append(_masked_mean_hidden(spec_tokens, current_mask))

            for stream_idx, write in enumerate(block_writes):
                snapshots_by_stream[stream_idx].append(write)

        for layer in self.model.instrumented_layers:
            layer.set_runtime_context(None)

        lm_logits = torch.cat(logits_by_block, dim=0)
        lm_labels = torch.cat(labels_by_block, dim=0)
        lm_label_mask = torch.cat(masks_by_block, dim=0)
        dependency_mask = torch.cat(dep_by_block, dim=0)
        nondependency_mask = torch.cat(non_by_block, dim=0)

        stream_classifier_logits = self.model.sidecar.stream_classifier(
            torch.cat(hidden_for_classifier, dim=0)
        )
        stream_targets = torch.arange(K, device=self.device).repeat(M).repeat_interleave(B)

        losses = compute_pdt_losses(
            stage=stage,
            weights=self.curriculum.active_loss_weights(stage),
            lm_logits=lm_logits,
            lm_labels=lm_labels,
            lm_label_mask=lm_label_mask,
            dependency_mask=dependency_mask,
            nondependency_mask=nondependency_mask,
            kd_temperature_lm=self.config.training.kd_temperature_lm,
            vq_commitment_loss=planner.commitment_loss,
            vq_codebook_loss=planner.codebook_loss,
            planner_logits=planner.logits,
            stream_logits=stream_classifier_logits,
            stream_targets=stream_targets,
        )

        loss = losses.total / self.config.training.grad_accumulation
        loss.backward()

        # Observe codebook selections.
        with torch.no_grad():
            self.codebook.observe_selections(planner.indices.detach().cpu())
            self.codebook.observe_anchors(plan_snapshot.detach().cpu())

        return losses.to_dict()

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
        self.model.eval()
        stats = self.codebook.compute()
        metrics = {
            "global_step": self.global_step,
            "stage": self.curriculum.current_stage,
            "codebook": stats.to_dict(),
            "codebook_passes_stage0_gate": stats.passes_stage0_gate(),
            "active_modules": self.curriculum.active_modules_snapshot(),
        }
        (self.telemetry_dir / f"eval_{self.global_step:07d}.json").write_text(
            json.dumps(metrics, indent=2)
        )
        LOGGER.info(
            "eval@step=%d stage=%d codebook=%s gate=%s",
            self.global_step,
            self.curriculum.current_stage,
            stats.to_dict(),
            stats.passes_stage0_gate(),
        )
        self.codebook.reset()
        self.model.train()

    def _save_checkpoint(self) -> None:
        ckpt_dir = self.telemetry_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"step_{self.global_step:07d}.pt"
        state = {
            "global_step": self.global_step,
            "stage": self.curriculum.current_stage,
            "sidecar": self.model.sidecar.state_dict(),
            "per_layer_phi": {
                f"layer_{layer.pdt_layer_idx}": {
                    "snc": layer.snc.state_dict() if layer.snc else None,
                    "stream_adapter": layer.stream_adapter.state_dict()
                        if layer.stream_adapter else None,
                    "notes_gate": layer.notes_gate.detach().cpu() if layer.notes_gate is not None else None,
                    "adapter_gate": layer.adapter_gate.detach().cpu() if layer.adapter_gate is not None else None,
                }
                for layer in self.model.instrumented_layers
            },
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(state, path)
        LOGGER.info("Saved checkpoint: %s", path)


def _to_device(batch: SampleBatch, device: torch.device) -> SampleBatch:
    return SampleBatch(
        example_ids=batch.example_ids,
        families=batch.families,
        stream_labels=batch.stream_labels,
        shared_ids=batch.shared_ids.to(device),
        shared_attention_mask=batch.shared_attention_mask.to(device),
        local_ids=batch.local_ids.to(device),
        local_attention_mask=batch.local_attention_mask.to(device),
        target_block_ids=batch.target_block_ids.to(device),
        target_block_labels=batch.target_block_labels.to(device),
        target_block_attention_mask=batch.target_block_attention_mask.to(device),
        dependency_token_mask=batch.dependency_token_mask.to(device),
        nondependency_token_mask=batch.nondependency_token_mask.to(device),
        readiness_targets=batch.readiness_targets.to(device),
        readiness_mask=batch.readiness_mask.to(device),
        raw=batch.raw,
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


def _visible_notes(
    snapshots_by_stream: list[list[torch.Tensor]],
    *,
    consumer: int,
    block_idx: int,
    lag: int,
    max_snapshots: int,
) -> torch.Tensor:
    del consumer
    visible: list[torch.Tensor] = []
    cutoff = 1 + max(0, block_idx - lag + 1)
    for stream_snapshots in snapshots_by_stream:
        visible.extend(stream_snapshots[:cutoff])
    if not visible:
        first = snapshots_by_stream[0][0]
        return first.new_zeros((first.size(0), 0, first.size(-1)))
    visible = visible[-max_snapshots:]
    return torch.stack(visible, dim=1)


def _teacher_forced_block_input(
    batch: SampleBatch,
    stream_idx: int,
    block_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids_parts = [
        batch.shared_ids,
        batch.local_ids[:, stream_idx],
    ]
    mask_parts = [
        batch.shared_attention_mask,
        batch.local_attention_mask[:, stream_idx],
    ]
    if block_idx > 0:
        prior_ids = batch.target_block_ids[:, stream_idx, :block_idx].reshape(
            batch.target_block_ids.size(0),
            -1,
        )
        prior_mask = batch.target_block_attention_mask[:, stream_idx, :block_idx].reshape(
            batch.target_block_attention_mask.size(0),
            -1,
        )
        ids_parts.append(prior_ids)
        mask_parts.append(prior_mask)
    ids_parts.append(batch.target_block_ids[:, stream_idx, block_idx])
    mask_parts.append(batch.target_block_attention_mask[:, stream_idx, block_idx])
    return torch.cat(ids_parts, dim=1), torch.cat(mask_parts, dim=1)


def _masked_mean_hidden(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return (hidden * weights).sum(dim=1) / denom
