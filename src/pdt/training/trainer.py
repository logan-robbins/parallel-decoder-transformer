"""Single-GPU / DDP trainer for PDT.

Training loop responsibilities:

1. Per-step curriculum tick (``CurriculumController.on_step``): flip
   requires_grad per stage.
2. Build one batch via ``PDTCollator``.
3. Forward: prompt encode (trunk), planner head, per-stream LM forward,
   notes/speculation heads, coverage, stream classifier.
4. Compute all loss terms; stage-mask via ``compute_pdt_losses``.
5. Backward, clip, step, LR schedule.
6. Periodic eval: codebook diagnostics + ROC for coverage / agreement.

This trainer intentionally uses a single LM forward per sample (the K
streams are implicit in the per-stream student_ids dimension). Multi-stream
inference uses the full orchestrator in ``pdt.runtime.orchestrator``.
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
from pdt.training.dataset import PDTCollator, PDTKDDataset, SampleBatch
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
            num_slots=self.config.sidecar.planner_head.num_slots,
            num_streams=self.config.sidecar.num_streams,
            notes_dim=self.config.sidecar.notes_dim,
            max_length=self.config.training.optimizer.warmup_steps and 2048 or 2048,
            max_plan_items=min(32, self.config.sidecar.planner_head.num_slots * 2),
            max_snapshots=self.config.runtime.notes_bus.max_snapshots,
        )

    def _build_dataloader(self, path: str, shuffle: bool) -> DataLoader:
        dataset = PDTKDDataset(path, num_streams=self.config.sidecar.num_streams)
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
        B = batch.student_ids.size(0)
        K = batch.student_ids.size(1)
        # Flatten (B, K, T) -> (B*K, T) for a single trunk forward pass.
        flat_ids = batch.student_ids.reshape(B * K, -1)
        flat_mask = batch.attention_mask.reshape(B * K, -1)
        flat_labels = batch.student_labels.reshape(B * K, -1)

        # Per-sample-per-stream runtime context isn't strictly needed at
        # training time since we're not accumulating notes on a bus; but we
        # still pass stream identity so the per-stream adapter fires.
        stream_names = self.config.runtime.streams
        # Run K forward passes to carry stream conditioning.
        all_hidden = []
        all_logits = []
        all_planner_logits = []
        all_planner_pooled = []
        for k in range(K):
            stream = stream_names[k]
            ctx = LayerRuntimeContext(
                stream=stream, notes=None, notes_mask=None, snc_force_gate=None
            )
            for layer in self.model.instrumented_layers:
                layer.set_runtime_context(ctx)
            ids = batch.student_ids[:, k]
            mask = batch.attention_mask[:, k]
            out = self.model.trunk_adapter.forward(
                input_ids=ids,
                attention_mask=mask,
                use_cache=False,
                output_hidden_states=True,
            )
            all_hidden.append(out.hidden_states[-1])
            all_logits.append(out.logits)

        for layer in self.model.instrumented_layers:
            layer.set_runtime_context(None)

        # Planner runs on the prompt hidden; for training we treat stream 0's
        # hidden as the prompt representation (in training the per-stream
        # target texts are all built from the same prompt).
        prompt_hidden = all_hidden[0]
        prompt_mask = batch.attention_mask[:, 0].to(prompt_hidden.dtype)
        planner_logits = self.model.sidecar.planner_head(prompt_hidden, attention_mask=prompt_mask)

        # Notes + speculation: averaged over the generated sequence.
        # notes_per_stream: list of (B, notes_dim)
        notes_per_stream = [
            self.model.sidecar.notes_head(h).mean(dim=1) for h in all_hidden
        ]
        spec_per_stream = [
            self.model.sidecar.speculation_head(h).mean(dim=1) for h in all_hidden
        ]
        student_notes = torch.stack(notes_per_stream, dim=1)  # (B, K, d_notes)
        student_spec = torch.stack(spec_per_stream, dim=1)

        # Coverage: use the concatenated hidden across streams as attended source.
        attended = torch.cat(all_hidden, dim=1)  # (B, K*T, H)
        plan_embs = self.model.sidecar.plan_embedding(batch.plan_item_ids)
        coverage_logits = self.model.sidecar.coverage_head(
            attended, plan_embs, batch.plan_item_mask
        )

        # Agreement: (B, 1) per sample. Uses block-end hidden of stream 0.
        readiness = self.model.sidecar.agreement_head(
            all_hidden[0][:, -1, :],
            batch.teacher_notes.to(all_hidden[0]),
            coverage_logits,
            student_notes[:, 0],
        )  # (B, 1)

        # Stream classifier: trained to recover stream id from per-stream hidden.
        # Build (B*K, H) targets.
        stream_classifier_logits = self.model.sidecar.stream_classifier(
            torch.cat(all_hidden, dim=0)  # (B*K, T, H)
        )
        stream_targets = torch.arange(K, device=self.device).repeat_interleave(B)

        # Average LM logits over K streams for the LM loss.
        lm_logits = torch.cat(all_logits, dim=0)  # (B*K, T, V)
        lm_labels = flat_labels
        lm_label_mask = (flat_labels >= 0)

        losses = compute_pdt_losses(
            stage=stage,
            weights=self.curriculum.active_loss_weights(stage),
            planner_logits=planner_logits,
            planner_targets=batch.planner_targets,
            kd_temperature_planner=self.config.training.kd_temperature_planner,
            student_notes=student_notes,
            teacher_notes=batch.teacher_notes,
            student_spec=student_spec,
            teacher_spec=batch.teacher_notes,  # Use teacher notes as spec target proxy.
            lm_logits=lm_logits,
            lm_labels=lm_labels,
            lm_label_mask=lm_label_mask,
            kd_temperature_lm=self.config.training.kd_temperature_lm,
            coverage_logits=coverage_logits,
            coverage_targets=batch.readiness_targets[:, : coverage_logits.size(1)]
                if coverage_logits.size(1) <= batch.readiness_targets.size(1)
                else batch.readiness_targets.new_ones(B, coverage_logits.size(1)),
            coverage_mask=batch.plan_item_mask,
            readiness_logits=readiness.squeeze(-1),
            readiness_targets=batch.readiness_targets[:, 0],
            readiness_mask=batch.readiness_mask[:, 0],
            stream_logits=stream_classifier_logits,
            stream_targets=stream_targets,
        )

        loss = losses.total / self.config.training.grad_accumulation
        loss.backward()

        # Observe codebook selections.
        with torch.no_grad():
            slot_ids = planner_logits.argmax(dim=-1).detach().cpu()
            self.codebook.observe_selections(slot_ids)

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
        sample_ids=batch.sample_ids,
        stream_labels=batch.stream_labels,
        student_ids=batch.student_ids.to(device),
        student_labels=batch.student_labels.to(device),
        attention_mask=batch.attention_mask.to(device),
        planner_targets=batch.planner_targets.to(device),
        planner_mask=batch.planner_mask.to(device),
        teacher_notes=batch.teacher_notes.to(device),
        student_notes=batch.student_notes.to(device),
        plan_item_ids=batch.plan_item_ids.to(device),
        plan_item_mask=batch.plan_item_mask.to(device),
        readiness_targets=batch.readiness_targets.to(device),
        readiness_mask=batch.readiness_mask.to(device),
        raw=batch.raw,
    )


def _infinite(loader):
    while True:
        for batch in loader:
            yield batch
