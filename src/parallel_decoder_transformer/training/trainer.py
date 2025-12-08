"""Minimal fine-tuning loop for the GPT-OSS backed Parallel Decoder Transformer transformer."""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from ..data.collator_kd import (
    TwoBranchKDCollatorConfig,
    TwoBranchKnowledgeDistillationCollator,
)
from ..data.teacher_provider import (
    CachedTeacherNotesProvider,
    DatasetTeacherNotesProvider,
    TeacherNotesProviderBase,
)
from ..data.teacher_runner import DatasetTeacherConfig
from ..models import ParallelDecoderTransformer
from ..utils import resolve_device, seed_everything
from ..utils.nli import NliScorer, NliScorerConfig


_PLAN_SNAPSHOT_FREEZE_KEY = "_plan_snapshot_freeze"


@dataclass(slots=True)
class TeacherBranchConfig:
    enabled: bool = True
    type: str = "stop_grad"  # {stop_grad, ema}
    ema_decay: float = 0.99


@dataclass(slots=True)
class CurriculumConfig:
    B: int = 1
    L: int = 32
    delta: int = 1
    rho_by_stream: Dict[str, float] = field(default_factory=dict)
    cadence_min: int = 8
    cadence_max: int = 128
    steps_per_stage: int = 0
    stage_schedule: Tuple[int, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class LossWeights:
    kd: float = 1.0
    stab: float = 0.1
    use: float = 0.0
    cov: float = 1.0
    nli: float = 0.05
    red: float = 0.0
    spec_kl: float = 0.1
    stream: float = 0.0
    agree: float = 1.0


@dataclass(slots=True)
class NotesNoiseConfig:
    drop_p: float = 0.0
    paraphrase_p: float = 0.0


@dataclass(slots=True)
class StagePolicyConfig:
    name: str = ""
    freeze: Tuple[str, ...] = ()
    unfreeze: Tuple[str, ...] = ()
    bus_mix_prob: Optional[float] = None
    stream_dropout_prob: Optional[float] = None
    notes_noise: Optional[NotesNoiseConfig] = None
    loss_weights: Optional[LossWeights] = None


@dataclass(slots=True)
class MetricsConfig:
    mask_ablation_every: int = 0
    stability_every: int = 0


@dataclass(slots=True)
class NegativeSamplingConfig:
    enabled: bool = False
    start_stage: int = 3
    contradiction_ratio: float = 0.0
    max_contradictions: int = 4
    noise_ratio: float = 0.0
    noise_std: float = 0.02


@dataclass(slots=True)
class GradNormConfig:
    enabled: bool = False
    target_ratio: float = 1.0
    alpha: float = 0.05
    min_scale: float = 0.1
    max_scale: float = 5.0


@dataclass(slots=True)
class TrainingConfig:
    dataset_path: Optional[str] = None
    eval_dataset_path: Optional[str] = None
    telemetry_dir: Optional[str] = None
    batch_size: int = 1
    seed: Optional[int] = None
    grad_accumulation: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_steps: int = 1000
    warmup_steps: int = 0
    log_interval: int = 10
    eval_interval: int = 200
    device: Optional[str] = None
    teacher: TeacherBranchConfig = field(default_factory=TeacherBranchConfig)
    dataset_teacher: DatasetTeacherConfig = field(
        default_factory=DatasetTeacherConfig
    )  # Required: teacher notes must be pre-generated in dataset
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    usage_min_stage: int = 4
    usage_margin: float = 0.0
    coverage_threshold: float = 0.5
    bus_mix_prob: float = 0.0
    stream_dropout_prob: float = 0.0
    parallel_micro_steps: int = 0
    notes_noise: NotesNoiseConfig = field(default_factory=NotesNoiseConfig)
    nli_scorer: Optional[str] = None
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    negative_sampling: NegativeSamplingConfig = field(default_factory=NegativeSamplingConfig)
    gradnorm: GradNormConfig = field(default_factory=GradNormConfig)
    stage_policies: Dict[int, StagePolicyConfig] = field(default_factory=dict)
    nli_margin: float = 0.1
    spec_kl_temperature: float = 1.0
    kd_temperature_planner: float = 1.0
    kd_temperature_lm: float = 1.0
    agreement_threshold: float = 0.15
    sectional_self_mask_tokens: int = 0
    topology: Literal["all_to_all"] = "all_to_all"

    # Robustness & Performance
    max_grad_norm: float = 1.0
    save_every: int = 500
    dataloader_workers: int = 4
    resume_from_checkpoint: bool = True  # Automatically resume from latest checkpoint if available


@dataclass(slots=True)
class TrainerState:
    global_step: int = 0
    epoch: int = 0
    best_eval_loss: float = float("inf")
    stage_index: int = 0
    stage_history: list[Dict[str, float]] = field(default_factory=list)


class Trainer:
    """Orchestrates a lean PEFT-style training loop."""

    def __init__(
        self,
        model: ParallelDecoderTransformer,
        config: TrainingConfig,
        *,
        collator_config: TwoBranchKDCollatorConfig,
        dataset: Optional[Dataset[Dict[str, object]]] = None,
        eval_dataset: Optional[Dataset[Dict[str, object]]] = None,
    ) -> None:
        self.model = model
        self.config = config
        seed_everything(self.config.seed)
        self.state = TrainerState()
        self.logger = logging.getLogger("parallel decoder transformer.trainer")
        preferred = config.device or resolve_device()
        if preferred == "cuda" and not torch.cuda.is_available():
            preferred = "cpu"
        if preferred == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                preferred = "cpu"
        self.device = torch.device(preferred)

        # Check if the trunk has an HF device map (sharded across GPUs)
        # If so, we must NOT call model.to(device) as it would collapse shards to a single device.
        trunk = getattr(self.model.trunk_adapter, "model", None)
        has_device_map = trunk is not None and getattr(trunk, "hf_device_map", None) is not None

        if not has_device_map:
            self.model.to(self.device)
        else:
            self.logger.info("trainer_skip_model_to_device | reason=hf_device_map_detected")

        # DDP Setup
        self.is_ddp = torch.distributed.is_initialized()
        self.rank = 0
        if self.is_ddp:
            self.rank = torch.distributed.get_rank()
            # Wrap model in DDP
            # find_unused_parameters=True is required because we have frozen parameters in early stages
            # The "marked ready twice" bug is caused by calling model() multiple times with plan_item_ids
            # Solution: Only pass plan_item_ids in the MAIN forward pass, set to None in diagnostic passes
            self.model = DDP(self.model, device_ids=[self.device], find_unused_parameters=True)  # type: ignore
            self.logger.info("ddp_model_wrapped | rank=%d | find_unused_parameters=True", self.rank)

        self.metric_history: Dict[str, List[Dict[str, float]]] = {"train": [], "eval": []}
        self._stage_transitions: List[Dict[str, Any]] = []
        self._stage_start_step: int = 0
        self._stage_start_time: float = time.time()
        self._stage_history_finalized: bool = False
        self._agreement_thresholds: Tuple[float, ...] = tuple(
            round(val, 2) for val in torch.linspace(0.05, 0.95, 19).tolist()
        )
        self._agreement_stats: Dict[float, Dict[str, float]] = {
            tau: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0} for tau in self._agreement_thresholds
        }
        self._last_agreement_update_step: int = -1
        if not self.config.teacher.enabled:
            raise ValueError(
                "TrainingConfig.teacher.enabled must be True; the teacher branch is mandatory for KD."
            )
        if self.config.kd_temperature_planner <= 0.0:
            raise ValueError("kd_temperature_planner must be positive.")
        if self.config.kd_temperature_lm <= 0.0:
            raise ValueError("kd_temperature_lm must be positive.")
        schedule = tuple(self.config.curriculum.stage_schedule)
        if schedule:
            if list(schedule) != sorted(schedule):
                raise ValueError("curriculum.stage_schedule must be non-decreasing.")
        if self.config.usage_min_stage < 0:
            raise ValueError("TrainingConfig.usage_min_stage must be non-negative.")
        if self.config.usage_margin < 0.0:
            raise ValueError("TrainingConfig.usage_margin must be non-negative.")
            if schedule[0] != 0:
                raise ValueError("curriculum.stage_schedule must start at step 0.")
            if any(step < 0 for step in schedule):
                raise ValueError("curriculum.stage_schedule cannot contain negative steps.")
        self.teacher_model: Optional[ParallelDecoderTransformer] = None
        if config.teacher.enabled and config.teacher.type == "ema":
            self.teacher_model = copy.deepcopy(model)
            self.teacher_model.to(self.device)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad_(False)
        elif config.teacher.enabled:
            self.teacher_model = model
        self.collator_config = collator_config
        if self.collator_config.commit_horizon <= 0:
            self.collator_config.commit_horizon = config.curriculum.L
        self.collator_config.max_snapshots = max(
            self.collator_config.max_snapshots, config.curriculum.B
        )
        self.teacher_provider = self._build_teacher_provider()
        self.collator = TwoBranchKnowledgeDistillationCollator(
            self.collator_config,
            teacher_provider=self.teacher_provider,
        )
        self.stream_lookup = {
            index: stream for stream, index in collator_config.stream_to_id.items()
        }
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.optimizer = AdamW(
            self._trainable_parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self._lr_lambda)
        self.kd_scale = 1.0
        self.plan_hash_buckets = collator_config.plan_hash_buckets
        self.nli_scorer: Optional[NliScorer] = None
        if config.nli_scorer:
            scorer_config = NliScorerConfig(model_name=config.nli_scorer)
            self.nli_scorer = NliScorer(scorer_config, device=self.device)

    def _trainable_parameters(self) -> Iterable[nn.Parameter]:
        # Handle DDP wrapper
        model = self.model.module if self.is_ddp else self.model
        for param in model.iter_trainable_parameters():
            if param.requires_grad:
                yield param

    def _build_teacher_provider(self) -> TeacherNotesProviderBase:
        """Build teacher provider that reads pre-generated notes from dataset.

        Teacher notes must be generated during dataset pipeline (Stage 3: Notes Generation).
        No LLM calls happen during training - all notes are read from metadata.
        """
        backend = DatasetTeacherNotesProvider(
            self.config.dataset_teacher,
            notes_dim=self.collator_config.notes_dim,
            stream_to_id=self.collator_config.stream_to_id,
        )
        cache_dir = (
            Path(self.config.dataset_teacher.cache_dir)
            if self.config.dataset_teacher.cache_dir
            else None
        )
        return CachedTeacherNotesProvider(
            backend=backend,
            cache_dir=cache_dir,
            id_field=self.config.dataset_teacher.id_field,
            refresh=self.config.dataset_teacher.refresh_cache,
        )

    def _lr_lambda(self, step: int) -> float:
        if self.config.warmup_steps <= 0:
            return 1.0
        return min(1.0, step / float(self.config.warmup_steps))

    def _determine_stage(self) -> int:
        schedule = tuple(self.config.curriculum.stage_schedule)
        steps_per_stage = self.config.curriculum.steps_per_stage
        if schedule:
            computed_stage = 0
            for index, threshold in enumerate(schedule):
                if self.state.global_step >= threshold:
                    computed_stage = index
            computed_stage = min(computed_stage, 4)
        elif steps_per_stage and steps_per_stage > 0:
            computed_stage = min(4, self.state.global_step // steps_per_stage)
        else:
            computed_stage = self.state.stage_index
        previous_stage = self.state.stage_index
        first_transition = not self.state.stage_history
        if first_transition or computed_stage != previous_stage:
            transition_from = previous_stage if not first_transition else -1
            self.state.stage_index = computed_stage
            self.state.stage_history.append(
                {"step": self.state.global_step, "stage": computed_stage}
            )
            self._on_stage_transition(transition_from, computed_stage)
        return self.state.stage_index

    def _on_stage_transition(self, previous_stage: int, new_stage: int) -> None:
        now = datetime.now(timezone.utc)
        if previous_stage >= 0 and self._stage_transitions:
            self._finalize_stage_record(self._stage_transitions[-1], now)
        policy = self.config.stage_policies.get(new_stage)
        stage_name = policy.name if policy and policy.name else f"stage_{new_stage}"
        record = {
            "stage_index": int(new_stage),
            "stage_name": stage_name,
            "start_step": int(self.state.global_step),
            "timestamp": now.isoformat(),
            "actions": {},
        }
        self._stage_transitions.append(record)
        self._stage_start_step = self.state.global_step
        self._stage_start_time = time.time()
        if policy is not None:
            self._apply_stage_policy(policy, record["actions"])
        if not record["actions"]:
            record.pop("actions", None)
        if self.state.stage_history:
            self.state.stage_history[-1]["stage_name"] = stage_name
        origin = previous_stage if previous_stage >= 0 else "init"
        self.logger.info(
            "stage_transition | from=%s | to=%d | step=%d | name=%s",
            origin,
            new_stage,
            self.state.global_step,
            stage_name,
        )

    def _apply_stage_policy(self, policy: StagePolicyConfig, actions: Dict[str, Any]) -> None:
        if policy.bus_mix_prob is not None:
            self.config.bus_mix_prob = float(policy.bus_mix_prob)
            actions["bus_mix_prob"] = self.config.bus_mix_prob
        if policy.stream_dropout_prob is not None:
            self.config.stream_dropout_prob = float(policy.stream_dropout_prob)
            actions["stream_dropout_prob"] = self.config.stream_dropout_prob
        if policy.notes_noise is not None:
            notes_cfg = NotesNoiseConfig(
                drop_p=policy.notes_noise.drop_p,
                paraphrase_p=policy.notes_noise.paraphrase_p,
            )
            self.config.notes_noise = notes_cfg
            actions["notes_noise"] = {
                "drop_p": notes_cfg.drop_p,
                "paraphrase_p": notes_cfg.paraphrase_p,
            }
        if policy.freeze:
            frozen = self._update_trainable(policy.freeze, trainable=False)
            if frozen:
                actions["freeze"] = frozen
        if policy.unfreeze:
            unfrozen = self._update_trainable(policy.unfreeze, trainable=True)
            if unfrozen:
                actions["unfreeze"] = unfrozen

    def _update_trainable(self, identifiers: Tuple[str, ...], *, trainable: bool) -> List[str]:
        applied: List[str] = []
        for resolved_name, module in self._resolve_policy_modules(identifiers):
            if module is None:
                self.logger.warning(
                    "stage_policy_missing_module | stage=%d | module=%s",
                    self.state.stage_index,
                    resolved_name,
                )
                continue
            # Skip non-Module objects (e.g., notes_bus is not nn.Module)
            if not isinstance(module, nn.Module):
                self.logger.debug(
                    "stage_policy_skip_non_module | stage=%d | module=%s | type=%s",
                    self.state.stage_index,
                    resolved_name,
                    type(module).__name__,
                )
                continue
            for param in module.parameters():
                param.requires_grad_(trainable)
            applied.append(resolved_name)
        return applied

    def _resolve_policy_modules(
        self, identifiers: Iterable[str]
    ) -> List[Tuple[str, Optional[nn.Module]]]:
        # Handle DDP wrapper for attribute access
        model = self.model.module if self.is_ddp else self.model

        modules: List[Tuple[str, Optional[nn.Module]]] = []
        for identifier in identifiers:
            key = identifier.strip()
            if not key:
                continue
            lower = key.lower()
            if lower == "trunk":
                modules.append(("trunk", getattr(model.trunk_adapter, "model", None)))
                continue
            if lower in {
                "stream_adapters",
                "cross_attention",
                "notes_bus",
                "planner_head",
                "notes_head",
                "speculation_head",
                "agreement_head",
                "coverage_head",
                "stream_classifier",
                "plan_embedding",
            }:
                modules.append((lower, getattr(model, lower, None)))
                continue
            if lower in {"heads", "all_heads"}:
                modules.extend(
                    [
                        ("planner_head", getattr(model, "planner_head", None)),
                        ("notes_head", getattr(model, "notes_head", None)),
                        ("speculation_head", getattr(model, "speculation_head", None)),
                        ("agreement_head", getattr(model, "agreement_head", None)),
                        ("coverage_head", getattr(model, "coverage_head", None)),
                        ("stream_classifier", getattr(model, "stream_classifier", None)),
                    ]
                )
                continue
            modules.append((lower, getattr(model, lower, None)))
        return modules

    def _finalize_stage_record(self, record: Dict[str, Any], timestamp: datetime) -> None:
        if record.get("steps") is not None:
            return
        elapsed_steps = max(0, self.state.global_step - self._stage_start_step)
        duration = max(0.0, time.time() - self._stage_start_time)
        record["end_step"] = int(self.state.global_step)
        record["steps"] = int(elapsed_steps)
        record["duration"] = float(duration)
        record["completed_at"] = timestamp.isoformat()

    def _finalize_stage_history(self) -> None:
        if self._stage_history_finalized:
            return
        if self._stage_transitions:
            self._finalize_stage_record(self._stage_transitions[-1], datetime.now(timezone.utc))
        self._stage_history_finalized = True

    def write_stage_history(self, telemetry_dir: Path | str | None) -> Optional[Path]:
        if telemetry_dir is None:
            return None
        self._finalize_stage_history()
        if not self._stage_transitions:
            return None
        target = Path(telemetry_dir)
        target.mkdir(parents=True, exist_ok=True)
        path = target / "train_run_stages.json"
        path.write_text(json.dumps(self._stage_transitions, indent=2), encoding="utf-8")
        self.logger.info("stage_history_written | path=%s", path)
        return path

    def write_agreement_threshold(self, telemetry_dir: Path | str | None) -> Optional[Path]:
        if telemetry_dir is None:
            return None
        target = Path(telemetry_dir)
        target.mkdir(parents=True, exist_ok=True)
        summary = self._maybe_recalibrate_agreement_threshold(store_points=True)
        payload: Dict[str, Any] = {"agreement_threshold": float(self.config.agreement_threshold)}
        if summary is not None:
            payload["roc_points"] = summary.get("points", [])
            current = summary.get("current")
            if current:
                payload["selected_point"] = current
        path = target / "agreement_thresholds.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.logger.info(
            "agreement_threshold_written | path=%s | tau=%.3f",
            path,
            self.config.agreement_threshold,
        )
        return path

    def _active_loss_weights(self, stage: int) -> LossWeights:
        weights = self.config.loss_weights
        if self.config.stage_policies is None:
            return weights
        stage_policy = self.config.stage_policies.get(stage)
        if stage_policy is None:
            return weights

        modified_weights = copy.deepcopy(weights)
        if stage_policy.loss_weights is not None:
            # Merge stage-specific loss weights into the global weights
            for key, value in stage_policy.loss_weights.__dict__.items():
                if hasattr(modified_weights, key):
                    setattr(modified_weights, key, value)
        return modified_weights

    def _save_checkpoint(self, step: int) -> None:
        """Save an intermediate adapter checkpoint."""
        if self.rank != 0:
            return
        if not self.config.telemetry_dir:
            return
        try:
            base_dir = Path(self.config.telemetry_dir)
            base_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = base_dir / f"adapters_step_{step}.pt"
            # Handle DDP wrapper
            model = self.model.module if self.is_ddp else self.model
            adapter_state = model.adapter_state_dict()

            # Save training state (step, epoch, stage, optimizer, scheduler)
            training_state = {
                "global_step": self.state.global_step,
                "epoch": self.state.epoch,
                "stage_index": self.state.stage_index,
                "stage_history": self.state.stage_history,
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "best_eval_loss": self.state.best_eval_loss,
                "kd_scale": self.kd_scale,
            }

            checkpoint = {
                "adapters": adapter_state,
                "training_state": training_state,
            }
            torch.save(checkpoint, ckpt_path)
            self.logger.info("checkpoint_saved | step=%d | path=%s", step, ckpt_path)

            # Clean up old checkpoints
            self._cleanup_old_checkpoints(base_dir, current_step=step)

        except Exception as e:
            self.logger.warning("checkpoint_failed | step=%d | error=%s", step, str(e))

    def _cleanup_old_checkpoints(
        self, base_dir: Path, current_step: int, keep_last: int = 3
    ) -> None:
        """Remove old checkpoints, keeping only recent ones and stage transitions."""
        try:
            # Find all checkpoint files
            checkpoint_files = sorted(base_dir.glob("adapters_step_*.pt"))
            if len(checkpoint_files) <= keep_last:
                return

            # Get stage transition steps
            stage_transitions = {
                record["start_step"] for record in self._stage_transitions if "start_step" in record
            }

            # Identify checkpoints to keep
            keep_steps = set()

            # Keep stage transitions
            keep_steps.update(stage_transitions)

            # Keep last N checkpoints
            recent_files = sorted(checkpoint_files, key=lambda p: int(p.stem.split("_")[-1]))[
                -keep_last:
            ]
            for f in recent_files:
                step = int(f.stem.split("_")[-1])
                keep_steps.add(step)

            # Delete old checkpoints
            for ckpt_file in checkpoint_files:
                step = int(ckpt_file.stem.split("_")[-1])
                if step not in keep_steps:
                    ckpt_file.unlink()
                    self.logger.info("checkpoint_cleaned | step=%d | path=%s", step, ckpt_file)

        except Exception as e:
            self.logger.warning("checkpoint_cleanup_failed | error=%s", str(e))

    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint in telemetry_dir."""
        if not self.config.telemetry_dir:
            return None
        base_dir = Path(self.config.telemetry_dir)
        if not base_dir.exists():
            return None

        checkpoint_files = list(base_dir.glob("adapters_step_*.pt"))
        if not checkpoint_files:
            return None

        # Sort by step number (highest first)
        checkpoint_files.sort(key=lambda p: int(p.stem.split("_")[-1]), reverse=True)
        return checkpoint_files[0]

    def _load_checkpoint(self, checkpoint_path: Path) -> bool:
        """Load checkpoint and restore training state. Returns True if successful."""
        try:
            self.logger.info("checkpoint_loading | path=%s", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load adapter weights
            model = self.model.module if self.is_ddp else self.model
            model.load_adapters(checkpoint["adapters"], strict=False)

            # Restore training state
            training_state = checkpoint["training_state"]
            self.state.global_step = training_state["global_step"]
            self.state.epoch = training_state["epoch"]
            self.state.stage_index = training_state["stage_index"]
            self.state.stage_history = training_state["stage_history"]
            self.state.best_eval_loss = training_state.get("best_eval_loss", float("inf"))
            self.kd_scale = training_state.get("kd_scale", 1.0)

            # Restore optimizer and scheduler
            self.optimizer.load_state_dict(training_state["optimizer_state"])
            self.scheduler.load_state_dict(training_state["scheduler_state"])

            self.logger.info(
                "checkpoint_loaded | step=%d | epoch=%d | stage=%d",
                self.state.global_step,
                self.state.epoch,
                self.state.stage_index,
            )
            return True

        except Exception as e:
            self.logger.error(
                "checkpoint_load_failed | path=%s | error=%s", checkpoint_path, str(e)
            )
            return False

    def fit(self) -> None:
        if self.dataset is None:
            raise RuntimeError("Trainer.fit requires a training dataset instance.")

        # Check for existing checkpoint and resume if configured
        # CRITICAL DDP FIX: Run on ALL ranks to ensure weights/optimizer/policy are synced.
        if getattr(self.config, "resume_from_checkpoint", True):
            latest_checkpoint = self._find_latest_checkpoint()
            if latest_checkpoint:
                if self._load_checkpoint(latest_checkpoint):
                    if self.rank == 0:
                        self.logger.info("training_resumed | from_step=%d", self.state.global_step)

                    # FORCE RE-APPLY STAGE POLICY
                    # Resume leaves model fully unfrozen because requires_grad is not saved.
                    # We must re-apply the policy for the current stage.
                    current_policy = self.config.stage_policies.get(self.state.stage_index)
                    if current_policy:
                        if self.rank == 0:
                            self.logger.info(
                                "resuming_stage_policy | stage=%d", self.state.stage_index
                            )
                        self._apply_stage_policy(current_policy, {})
                else:
                    if self.rank == 0:
                        self.logger.warning("checkpoint_resume_failed_starting_fresh")

        # Sync state across all DDP ranks after potential checkpoint load
        if self.is_ddp:
            import torch.distributed as dist

            # Broadcast global_step from rank 0 to all ranks (redundant if all loaded, but safe)
            step_tensor = torch.tensor(self.state.global_step, dtype=torch.long, device=self.device)
            dist.broadcast(step_tensor, src=0)
            self.state.global_step = int(step_tensor.item())

        sampler = None
        shuffle = True
        if self.is_ddp:
            sampler = DistributedSampler(self.dataset, shuffle=True)
            shuffle = False  # Sampler handles shuffling

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=self.collator,
            num_workers=self.config.dataloader_workers,
            pin_memory=True if self.device.type == "cuda" else False,
        )
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        while self.state.global_step < self.config.max_steps:
            if self.is_ddp and sampler is not None:
                sampler.set_epoch(self.state.epoch)

            for batch in dataloader:
                loss, metrics = self._training_step(batch)

                # Normalize loss for gradient accumulation to keep effective LR consistent
                # regardless of accumulation steps.
                if self.config.grad_accumulation > 1:
                    loss = loss / self.config.grad_accumulation

                loss.backward()

                # Gradient clipping
                if self.config.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self._trainable_parameters(), self.config.max_grad_norm
                    )

                if (self.state.global_step + 1) % self.config.grad_accumulation == 0:
                    self.optimizer.step()
                    self._update_teacher_ema()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                self.state.global_step += 1
                if self.state.global_step % self.config.log_interval == 0:
                    self._log_metrics("train", metrics)
                if self.state.global_step % self.config.eval_interval == 0:
                    self.evaluate()
                if (
                    self.config.save_every > 0
                    and self.state.global_step % self.config.save_every == 0
                ):
                    self._save_checkpoint(self.state.global_step)
                if self.state.global_step >= self.config.max_steps:
                    break
            self.state.epoch += 1
        self._finalize_stage_history()

    def _ensure_plan_snapshot_freeze_state(self, batch: Dict[str, torch.Tensor]) -> None:
        freeze_window = max(0, self.config.curriculum.B)
        if freeze_window <= 0:
            return
        sectional = self._sectional_mask(batch)
        if sectional is None:
            return
        if sectional.numel() == 0:
            return
        freeze_tensor = torch.zeros(sectional.size(0), dtype=torch.long, device=self.device)
        freeze_tensor[sectional] = freeze_window
        batch[_PLAN_SNAPSHOT_FREEZE_KEY] = freeze_tensor

    def _sectional_mask(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Return a boolean mask for sectional-independence samples, if present."""
        sectional = batch.get("sectional_independence")
        if sectional is None or not torch.is_tensor(sectional):
            return None
        mask = sectional.to(device=self.device, dtype=torch.bool)
        if mask.dim() == 0:
            mask = mask.view(1)
        return mask

    def _build_sectional_note_mask(
        self,
        *,
        sectional_mask: Optional[torch.Tensor],
        stream_ids: torch.Tensor,
        snapshot_streams: Optional[torch.Tensor],
        notes_mask: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        limit_tokens = max(0, self.config.sectional_self_mask_tokens)
        if (
            limit_tokens <= 0
            or sectional_mask is None
            or snapshot_streams is None
            or notes_mask.numel() == 0
        ):
            return None
        base_mask = notes_mask.to(device=self.device, dtype=torch.bool)
        if base_mask.dim() != 2:
            return None
        sectional_bool = sectional_mask.to(device=self.device, dtype=torch.bool)
        if sectional_bool.dim() == 0:
            sectional_bool = sectional_bool.view(1)
        if sectional_bool.size(0) != base_mask.size(0):
            return None
        if not sectional_bool.any():
            return None
        seq_len = int(input_ids.size(1))
        if seq_len <= 0:
            return None
        note_count = base_mask.size(1)
        if note_count == 0:
            return None
        token_mask = base_mask.unsqueeze(1).expand(-1, seq_len, -1).clone()
        stream_ids_vec = stream_ids.to(device=self.device).view(-1)
        if stream_ids_vec.size(0) != base_mask.size(0):
            return None
        stream_table = snapshot_streams.to(device=self.device)
        if stream_table.size() != base_mask.size():
            return None
        for idx in range(base_mask.size(0)):
            if not sectional_bool[idx]:
                continue
            window = min(limit_tokens, seq_len)
            if window <= 0:
                continue
            matches = stream_table[idx] == stream_ids_vec[idx]
            allowed = base_mask[idx] & matches
            if not allowed.any():
                continue
            token_mask[idx, :window] = allowed.unsqueeze(0).expand(window, -1)
        token_mask = token_mask & base_mask.unsqueeze(1).expand_as(token_mask)
        return token_mask

    @staticmethod
    def _merge_note_masks(
        base_mask: torch.Tensor,
        sectional_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if sectional_mask is None:
            return base_mask
        sec_mask = sectional_mask.to(dtype=torch.bool)
        base_bool = base_mask.to(dtype=torch.bool)
        if sec_mask.dim() == base_bool.dim():
            if sec_mask.shape != base_bool.shape:
                return base_mask
            combined = sec_mask & base_bool
        elif sec_mask.dim() == base_bool.dim() + 1:
            expanded = base_bool.unsqueeze(1).expand(-1, sec_mask.size(1), -1)
            if expanded.shape != sec_mask.shape:
                return base_mask
            combined = sec_mask & expanded
        else:
            return base_mask
        return combined.to(dtype=base_mask.dtype)

    def _training_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        stage = self._determine_stage()
        self._maybe_apply_negative_sampling(batch, stage)
        batch = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        self._ensure_plan_snapshot_freeze_state(batch)
        sectional_mask = self._sectional_mask(batch)
        plan_item_ids = batch.get("plan_item_ids")
        plan_item_mask = batch.get("plan_item_mask")
        if plan_item_ids is not None:
            plan_item_ids = plan_item_ids.to(self.device)
        if plan_item_mask is not None:
            plan_item_mask = plan_item_mask.to(self.device)
        teacher_branch, student_branch = self._prepare_branches_for_batch(
            batch,
            stage,
            sectional_mask,
            batch["input_ids"],
        )
        student_encode_mask = self._merge_note_masks(
            student_branch["notes_mask"],
            student_branch.get("sectional_note_mask"),
        )
        hidden_states = self._encode_trunk(
            self.model,
            batch,
            notes=student_branch["notes"],
            notes_mask=student_encode_mask,
        )
        active_weights = self._active_loss_weights(stage)
        teacher_branch["notes"] = teacher_branch["notes"].to(hidden_states.dtype)
        student_branch["notes"] = student_branch["notes"].to(hidden_states.dtype)
        if "pre_notes" in student_branch:
            student_branch["pre_notes"] = student_branch["pre_notes"].to(hidden_states.dtype)

        if self._uses_separate_teacher():
            teacher_hidden = self._teacher_encode(batch, teacher_branch)
        else:
            teacher_hidden = hidden_states.detach()

        student_outputs = self._run_student_pass(
            hidden_states,
            batch,
            student_branch,
            stage=stage,
            plan_item_ids=plan_item_ids,
            plan_item_mask=plan_item_mask,
            sectional_mask=sectional_mask,
        )
        teacher_notes_mask = self._merge_note_masks(
            teacher_branch["notes_mask"],
            teacher_branch.get("sectional_note_mask"),
        )
        teacher_outputs = self._teacher_forward(
            teacher_hidden,
            stream=batch["stream_ids"],
            notes=teacher_branch["notes"],
            notes_mask=teacher_notes_mask,
            sectional_mask=sectional_mask,
        )

        # CRITICAL DDP FIX: Check if this step is a stage transition
        is_transition_step = self.state.global_step in self.config.curriculum.stage_schedule

        # Determine if stability check is due
        stability_logging_due = self.config.metrics.stability_every <= 0 or (
            self.config.metrics.stability_every > 0
            and (self.state.global_step % self.config.metrics.stability_every == 0)
        )

        need_pre_logits = active_weights.stab > 0.0 or stability_logging_due

        # ==============================================================================
        # CRITICAL FIX: Force disable pre-update pass on transition steps
        # This overrides both logging AND stability loss to prevent DDP double-forward crash
        # when parameters are newly unfrozen (avoiding "marked ready twice" error).
        # ==============================================================================
        if is_transition_step and need_pre_logits:
            if self.rank == 0:
                self.logger.warning(
                    "pre_update_pass_skipped | step=%d | reason=stage_transition_collision",
                    self.state.global_step,
                )
            need_pre_logits = False
            # Note: This effectively drops stability loss for 1 step, which is harmless.
        pre_update_logits: Optional[torch.Tensor] = None
        pre_update_lm_logits: Optional[torch.Tensor] = None
        if need_pre_logits and "pre_notes" in student_branch and "pre_notes_mask" in student_branch:
            with torch.no_grad():
                pre_mask_effective = self._merge_note_masks(
                    student_branch["pre_notes_mask"],
                    student_branch.get("pre_sectional_note_mask"),
                )
                # Unwrap DDP if present to avoid double-forward
                model = self.model
                if hasattr(model, "module"):
                    model = model.module
                # Pre-update pass is diagnostic only; do NOT pass plan_item_ids
                # to avoid duplicate gradients on plan_embedding.weight
                pre_outputs = model(
                    hidden_states,
                    stream=batch["stream_ids"],
                    notes=student_branch["pre_notes"],
                    notes_mask=pre_mask_effective,
                    plan_item_ids=None,
                    plan_item_mask=None,
                    sectional_mask=sectional_mask,
                )
            pre_update_logits = pre_outputs["planner_logits"]
            if "lm_logits" in pre_outputs:
                pre_update_lm_logits = pre_outputs["lm_logits"]

        total_loss, metrics = self._compute_losses(
            batch,
            student_outputs,
            teacher_outputs,
            student_branch=student_branch,
            teacher_branch=teacher_branch,
            hidden_states=hidden_states,
            stage=stage,
            weights=active_weights,
            step=self.state.global_step,
            pre_update_logits=pre_update_logits,
            pre_update_lm_logits=pre_update_lm_logits,
            stability_logging_due=stability_logging_due,
            sectional_mask=sectional_mask,
        )
        metrics["loss"] = float(total_loss.detach().cpu())
        metrics["stage"] = float(stage)
        return total_loss, metrics

    def generate_training_report(self) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        train_history = self.metric_history.get("train", [])
        eval_history = self.metric_history.get("eval", [])
        summary_keys = [
            "mask_ablation",
            "kd_ce_ratio",
            "agreement_precision",
            "rollback_kl",
            "stability_kl",
            "repair_error_rate",
            "stability_error_rate",
            "repair_margin",
            "stability_margin",
            "usage_loss",
            "coverage_precision",
            "coverage_recall",
            "coverage_f1",
            "coverage_cross_stream_fp_rate",
            "coverage_same_stream_recall",
        ]

        def _aggregate(key: str) -> Optional[Dict[str, float]]:
            values = [
                float(entry[key])
                for entry in train_history
                if key in entry
                and isinstance(entry[key], (int, float))
                and not math.isnan(float(entry[key]))
            ]
            if not values:
                return None
            return {
                "last": values[-1],
                "mean": float(sum(values) / len(values)),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }

        train_summary = {}
        for key in summary_keys:
            aggregated = _aggregate(key)
            if aggregated is not None:
                train_summary[key] = aggregated

        if train_history:
            loss_summary = _aggregate("loss") or _aggregate("planner_loss")
            if loss_summary is not None:
                train_summary.setdefault("loss", loss_summary)
            last_stage_value = train_history[-1].get("stage")
            if not isinstance(last_stage_value, (int, float)):
                last_stage_value = self.state.stage_index
            train_summary.setdefault("stage", {"last": float(last_stage_value)})

        eval_summary: Dict[str, float] = {}
        if eval_history:
            last_eval = dict(eval_history[-1])
            last_eval.pop("timestamp", None)
            last_eval.pop("step", None)
            eval_summary = last_eval

        report = {
            "generated_at": now,
            "global_step": self.state.global_step,
            "best_eval_loss": self.state.best_eval_loss,
            "stage": self.state.stage_index,
            "train_history_length": len(train_history),
            "eval_history_length": len(eval_history),
            "train_metrics": train_summary,
            "eval_metrics": eval_summary,
            "agreement_threshold": float(self.config.agreement_threshold),
        }
        return report

    def _maybe_apply_negative_sampling(self, batch: Dict[str, Any], stage: int) -> None:
        cfg = self.config.negative_sampling
        if not cfg.enabled or stage < cfg.start_stage:
            return
        if cfg.contradiction_ratio > 0.0 and cfg.max_contradictions > 0:
            self._inject_negative_plan_items(batch, cfg)
        if cfg.noise_ratio > 0.0 and cfg.noise_std > 0.0:
            self._inject_negative_noise(batch, cfg)

    def _inject_negative_plan_items(
        self, batch: Dict[str, Any], cfg: NegativeSamplingConfig
    ) -> None:
        plan_item_ids = batch.get("plan_item_ids")
        plan_item_mask = batch.get("plan_item_mask")
        if plan_item_ids is None or plan_item_mask is None:
            return
        if plan_item_ids.numel() == 0:
            return
        plan_item_mask = plan_item_mask.to(dtype=torch.bool)
        batch_size, width = plan_item_ids.shape
        neg_counts: List[int] = []
        max_negatives = 0
        for index in range(batch_size):
            positive = int(plan_item_mask[index].sum().item())
            if positive == 0:
                neg_counts.append(0)
                continue
            desired = max(1, math.ceil(positive * cfg.contradiction_ratio))
            desired = min(desired, cfg.max_contradictions)
            if desired <= 0:
                neg_counts.append(0)
                continue
            neg_counts.append(desired)
            max_negatives = max(max_negatives, desired)
        if max_negatives == 0:
            return
        device = plan_item_ids.device
        dtype = plan_item_ids.dtype
        new_width = width + max_negatives
        new_plan_ids = torch.zeros((batch_size, new_width), dtype=dtype, device=device)
        new_plan_ids[:, :width] = plan_item_ids
        new_plan_mask = torch.zeros((batch_size, new_width), dtype=torch.bool, device=device)
        new_plan_mask[:, :width] = plan_item_mask

        coverage_targets = batch.get("coverage_targets")
        coverage_mask = batch.get("coverage_mask")
        if coverage_targets is None:
            base_targets = torch.zeros((batch_size, width), dtype=torch.float32, device=device)
        else:
            base_targets = coverage_targets.to(device=device, dtype=torch.float32)
        if coverage_mask is None:
            base_mask = torch.zeros((batch_size, width), dtype=torch.bool, device=device)
        else:
            base_mask = coverage_mask.to(device=device, dtype=torch.bool)

        new_coverage_targets = torch.zeros(
            (batch_size, new_width), dtype=torch.float32, device=device
        )
        new_coverage_targets[:, :width] = base_targets
        new_coverage_mask = torch.zeros((batch_size, new_width), dtype=torch.bool, device=device)
        new_coverage_mask[:, :width] = base_mask

        plan_text = batch.get("plan_text")
        for index, neg_count in enumerate(neg_counts):
            if neg_count <= 0:
                continue
            existing_ids = set(plan_item_ids[index, plan_item_mask[index]].tolist())
            generated: List[int] = []
            attempts = 0
            while len(generated) < neg_count:
                candidate = int(
                    torch.randint(1, self.plan_hash_buckets, (1,), device=device).item()
                )
                if candidate == 0:
                    continue
                if candidate in existing_ids or candidate in generated:
                    attempts += 1
                    if attempts > neg_count * 8:
                        candidate = (
                            candidate + attempts + len(generated) + 1
                        ) % self.plan_hash_buckets
                        candidate = candidate or 1
                if candidate not in existing_ids and candidate not in generated:
                    generated.append(candidate)
            start = width
            end = width + neg_count
            new_plan_ids[index, start:end] = torch.tensor(generated, dtype=dtype, device=device)
            new_plan_mask[index, start:end] = True
            new_coverage_targets[index, start:end] = 0.0
            new_coverage_mask[index, start:end] = True
            if isinstance(plan_text, list) and index < len(plan_text):
                plan_text[index].extend([f"[negative-{value}]" for value in generated])

        batch["plan_item_ids"] = new_plan_ids
        batch["plan_item_mask"] = new_plan_mask
        batch["coverage_targets"] = new_coverage_targets
        batch["coverage_mask"] = new_coverage_mask

    def _inject_negative_noise(self, batch: Dict[str, Any], cfg: NegativeSamplingConfig) -> None:
        notes_student = batch.get("notes_student")
        if notes_student is None or notes_student.numel() == 0:
            return
        device = notes_student.device
        sample_mask = torch.rand((notes_student.size(0),), device=device) < cfg.noise_ratio
        if not sample_mask.any():
            return
        for index, selected in enumerate(sample_mask.tolist()):
            if not selected:
                continue
            original = notes_student[index].to(dtype=torch.float32)
            noise = torch.randn_like(original) * cfg.noise_std
            perturbed = original + noise
            notes_student[index] = perturbed.to(dtype=notes_student.dtype)

    def _prepare_branch_inputs(
        self,
        batch: Dict[str, Any],
        *,
        branch: str,
        stage: int,
        teacher_branch: Optional[Dict[str, torch.Tensor]] = None,
        sectional_mask: Optional[torch.Tensor],
        input_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if branch not in {"teacher", "student"}:
            raise ValueError(f"Unknown branch specifier: {branch}")
        prefix = "teacher" if branch == "teacher" else "student"
        lag_current = max(0, self.config.curriculum.delta)
        notes, coverage, snapshot_streams, snapshot_stride, snapshot_version = (
            self._extract_notes_from_bus(
                batch,
                notes_bus_key=f"{prefix}_notes_bus",
                mask_key=f"{prefix}_bus_mask",
                coverage_key=f"{prefix}_bus_coverage",
                streams_key=f"{prefix}_bus_streams",
                stride_key=f"{prefix}_bus_stride",
                version_key=f"{prefix}_bus_version",
                lag_override=lag_current,
                sectional_mask=sectional_mask,
            )
        )
        pre_notes: Optional[torch.Tensor] = None
        pre_snapshot_version: Optional[torch.Tensor] = None
        if branch == "student":
            (
                pre_notes,
                _,
                _,
                _,
                pre_snapshot_version,
            ) = self._extract_notes_from_bus(
                batch,
                notes_bus_key=f"{prefix}_notes_bus",
                mask_key=f"{prefix}_bus_mask",
                coverage_key=f"{prefix}_bus_coverage",
                streams_key=f"{prefix}_bus_streams",
                stride_key=f"{prefix}_bus_stride",
                version_key=f"{prefix}_bus_version",
                lag_override=lag_current + 1,
                sectional_mask=sectional_mask,
            )
        if branch == "student":
            if stage == 0 and teacher_branch is not None:
                notes = teacher_branch["notes"].clone()
            elif self.model.training:
                bus_mix_prob = self.config.bus_mix_prob if stage >= 3 else 0.0
                stream_dropout = self.config.stream_dropout_prob if stage >= 3 else 0.0
                noise_cfg = self.config.notes_noise if stage >= 3 else NotesNoiseConfig()
                if bus_mix_prob > 0.0 and teacher_branch is not None:
                    teacher_notes = teacher_branch["notes"]
                    mix_mask = torch.rand((notes.size(0), 1, 1), device=notes.device) < bus_mix_prob
                    notes = torch.where(mix_mask, teacher_notes, notes)
                if stream_dropout > 0.0:
                    drop_mask = (
                        torch.rand((notes.size(0), notes.size(1)), device=notes.device)
                        < stream_dropout
                    )
                    notes = notes.masked_fill(drop_mask.unsqueeze(-1), 0.0)
                if noise_cfg.drop_p > 0.0:
                    noise_drop = (
                        torch.rand((notes.size(0), notes.size(1)), device=notes.device)
                        < noise_cfg.drop_p
                    )
                    notes = notes.masked_fill(noise_drop.unsqueeze(-1), 0.0)
                if noise_cfg.paraphrase_p > 0.0:
                    noise_mask = (
                        torch.rand((notes.size(0), notes.size(1)), device=notes.device)
                        < noise_cfg.paraphrase_p
                    )
                    if noise_mask.any():
                        gaussian = torch.randn_like(notes)
                        notes = notes + gaussian * noise_mask.unsqueeze(-1) * 0.05
        notes_mask = (notes.abs().sum(dim=-1) > 0).long()
        if notes_mask.sum() == 0:
            notes_mask = torch.ones_like(notes_mask)
        branch_payload: Dict[str, torch.Tensor] = {
            "notes": notes,
            "notes_mask": notes_mask,
            "lag": torch.tensor(lag_current, device=notes.device, dtype=torch.long),
        }
        if pre_notes is not None:
            pre_mask = (pre_notes.abs().sum(dim=-1) > 0).long()
            if pre_mask.sum() == 0:
                pre_mask = torch.ones_like(pre_mask)
            branch_payload["pre_notes"] = pre_notes
            branch_payload["pre_notes_mask"] = pre_mask
            if pre_snapshot_version is not None:
                branch_payload["pre_snapshot_version"] = pre_snapshot_version
        if coverage is not None:
            branch_payload["coverage"] = coverage.to(device=notes.device, dtype=torch.float32)
        if snapshot_streams is not None:
            branch_payload["snapshot_streams"] = snapshot_streams
        if snapshot_stride is not None:
            branch_payload["snapshot_stride"] = snapshot_stride
        if snapshot_version is not None:
            branch_payload["snapshot_version"] = snapshot_version
        sectional_note_mask = self._build_sectional_note_mask(
            sectional_mask=sectional_mask,
            stream_ids=batch["stream_ids"],
            snapshot_streams=snapshot_streams,
            notes_mask=notes_mask,
            input_ids=input_ids,
        )
        if sectional_note_mask is not None:
            branch_payload["sectional_note_mask"] = sectional_note_mask
        return branch_payload

    def _prepare_branches_for_batch(
        self,
        batch: Dict[str, torch.Tensor],
        stage: int,
        sectional_mask: Optional[torch.Tensor],
        input_ids: torch.Tensor,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        teacher_branch = self._prepare_branch_inputs(
            batch,
            branch="teacher",
            stage=stage,
            sectional_mask=sectional_mask,
            input_ids=input_ids,
        )
        student_branch = self._prepare_branch_inputs(
            batch,
            branch="student",
            stage=stage,
            teacher_branch=teacher_branch,
            sectional_mask=sectional_mask,
            input_ids=input_ids,
        )
        return teacher_branch, student_branch

    def _encode_trunk(
        self,
        model: ParallelDecoderTransformer,
        batch: Dict[str, torch.Tensor],
        *,
        notes: torch.Tensor,
        notes_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_bool = notes_mask.to(dtype=torch.bool)
        # Handle DDP wrapper
        raw_model = model.module if hasattr(model, "module") else model
        return raw_model.encode_with_notes(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            stream=batch["stream_ids"],
            notes=notes,
            notes_mask=mask_bool,
        )

    def _extract_notes_from_bus(
        self,
        batch: Dict[str, Any],
        *,
        notes_bus_key: str,
        mask_key: str,
        coverage_key: str,
        streams_key: str,
        stride_key: str,
        version_key: str,
        lag_override: Optional[int] = None,
        sectional_mask: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        notes_bus = batch.get(notes_bus_key)
        mask = batch.get(mask_key)
        if notes_bus is None or mask is None:
            batch_size = len(batch["input_ids"])
            # Handle DDP wrapper
            model = self.model.module if self.is_ddp else self.model
            notes_dim = model.config.notes_dim
            return torch.zeros(batch_size, 1, notes_dim, device=self.device), None, None, None, None

        notes_bus = notes_bus.to(self.device)
        mask_bool = mask.to(device=self.device, dtype=torch.bool)
        batch_size, max_snapshots, stream_count, _ = notes_bus.shape
        sectional_bool: Optional[torch.Tensor] = None
        if sectional_mask is not None:
            sectional_t = sectional_mask.to(device=self.device, dtype=torch.bool)
            if sectional_t.dim() == 0:
                sectional_t = sectional_t.view(1)
            if sectional_t.size(0) != batch_size:
                raise ValueError(
                    f"Sectional mask length {sectional_t.size(0)} does not match batch size {batch_size}."
                )
            sectional_bool = sectional_t
        if mask_bool.sum().item() == 0:
            target = len(batch["input_ids"]) if "input_ids" in batch else batch_size
            # Handle DDP wrapper
            model = self.model.module if self.is_ddp else self.model
            notes_dim = model.config.notes_dim
            return torch.zeros(target, 1, notes_dim, device=self.device), None, None, None, None
        lag_value = self.config.curriculum.delta if lag_override is None else lag_override
        lag = max(0, lag_value)
        if lag > 0:
            for batch_index in range(batch_size):
                valid_indices = torch.nonzero(mask_bool[batch_index], as_tuple=False).view(-1)
                if valid_indices.numel() == 0:
                    continue
                drop_count = min(lag, valid_indices.numel())
                if sectional_bool is not None and bool(sectional_bool[batch_index].item()):
                    drop_count = 0
                if drop_count > 0:
                    drop_indices = valid_indices[-drop_count:]
                    mask_bool[batch_index, drop_indices] = False

        expanded_mask = mask_bool.unsqueeze(-1).expand(-1, -1, stream_count)
        mask_float = expanded_mask.unsqueeze(-1).to(dtype=notes_bus.dtype)
        filtered_notes = notes_bus * mask_float
        flat_notes = filtered_notes.view(batch_size, -1, notes_bus.size(-1))

        coverage_bus = batch.get(coverage_key)
        gathered_coverage: Optional[torch.Tensor]
        if coverage_bus is None:
            gathered_coverage = None
        else:
            coverage_bus = coverage_bus.to(self.device)
            filtered_coverage = coverage_bus * expanded_mask.to(dtype=coverage_bus.dtype)
            gathered_coverage = filtered_coverage.view(batch_size, -1)

        streams_bus = batch.get(streams_key)
        gathered_streams: Optional[torch.Tensor]
        if streams_bus is None:
            gathered_streams = None
        else:
            streams_bus = streams_bus.to(self.device)
            expanded_streams = streams_bus.unsqueeze(-1).expand(-1, -1, stream_count)
            flat_streams = expanded_streams.reshape(batch_size, -1)
            mask_flat = expanded_mask.reshape(batch_size, -1)
            gathered_streams = flat_streams.masked_fill(~mask_flat, -1)

        stride_bus = batch.get(stride_key)
        gathered_stride: Optional[torch.Tensor]
        if stride_bus is None:
            gathered_stride = None
        else:
            stride_bus = stride_bus.to(self.device)
            expanded_stride = stride_bus.unsqueeze(-1).expand(-1, -1, stream_count)
            mask_flat = expanded_mask.reshape(batch_size, -1)
            flat_stride = expanded_stride.reshape(batch_size, -1)
            gathered_stride = flat_stride.masked_fill(~mask_flat, 0)

        version_bus = batch.get(version_key)
        gathered_version: Optional[torch.Tensor]
        if version_bus is None:
            gathered_version = None
        else:
            version_bus = version_bus.to(self.device)
            expanded_version = version_bus.unsqueeze(-1).expand(-1, -1, stream_count)
            mask_flat = expanded_mask.reshape(batch_size, -1)
            flat_version = expanded_version.reshape(batch_size, -1)
            gathered_version = flat_version.masked_fill(~mask_flat, 0)

        return (
            flat_notes,
            gathered_coverage,
            gathered_streams,
            gathered_stride,
            gathered_version,
        )

    def _run_student_pass(
        self,
        hidden_states: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        student_branch: Dict[str, torch.Tensor],
        *,
        stage: int,
        plan_item_ids: Optional[torch.Tensor],
        plan_item_mask: Optional[torch.Tensor],
        sectional_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        student_branch["notes"] = student_branch["notes"].to(hidden_states.dtype)
        notes_mask = student_branch["notes_mask"]
        sectional_note_mask = student_branch.get("sectional_note_mask")

        # DDP Fix: Only pass plan_item_ids when coverage_head has trainable parameters
        # If passed when frozen, DDP can mark coverage_head params as "ready" multiple times
        # when they transition from frozen→trainable at stage boundaries
        model = self.model.module if hasattr(self.model, "module") else self.model
        coverage_trainable = False
        if hasattr(model, "coverage_head") and model.coverage_head is not None:
            coverage_trainable = any(p.requires_grad for p in model.coverage_head.parameters())

        # CRITICAL: Only use plan_item_ids in the MAIN student pass, never in diagnostic passes
        # This is the ONLY place where we should pass these to avoid duplicate DDP hooks
        effective_plan_ids = plan_item_ids if coverage_trainable else None
        effective_plan_mask = plan_item_mask if coverage_trainable else None

        if self.config.parallel_micro_steps > 0 and stage >= 2 and self.model.training:
            micro_notes = student_branch["notes"]
            last_outputs: Optional[Dict[str, torch.Tensor]] = None
            for micro_step in range(self.config.parallel_micro_steps):
                effective_mask = self._merge_note_masks(notes_mask, sectional_note_mask)
                last_outputs = self.model(
                    hidden_states,
                    stream=batch["stream_ids"],
                    notes=micro_notes,
                    notes_mask=effective_mask,
                    plan_item_ids=effective_plan_ids,
                    plan_item_mask=effective_plan_mask,
                    sectional_mask=sectional_mask,
                )
                new_notes = last_outputs["speculative_notes"].detach()
                coverage_tensor = student_branch.get("coverage")
                streams_tensor = student_branch.get("snapshot_streams")
                meta = self._update_student_bus(
                    batch,
                    new_notes,
                    snapshot_streams=streams_tensor,
                    coverage=coverage_tensor,
                )
                if "version" in meta:
                    student_branch["snapshot_version"] = meta["version"]
                if "stride" in meta:
                    student_branch["snapshot_stride"] = meta["stride"]
                if micro_step < self.config.parallel_micro_steps - 1:
                    micro_notes = new_notes
                    notes_mask = (micro_notes.abs().sum(dim=-1) > 0).long()
                    sectional_note_mask = student_branch.get("sectional_note_mask")
                self._advance_commit_mask(batch)
            if last_outputs is None:
                raise RuntimeError("Parallel micro-steps requested but no outputs produced.")
            student_branch["notes"] = last_outputs["speculative_notes"]
            student_branch["notes_mask"] = (student_branch["notes"].abs().sum(dim=-1) > 0).long()
            self._refresh_pre_notes(batch, student_branch, sectional_mask=sectional_mask)
            return last_outputs
        effective_mask = self._merge_note_masks(notes_mask, sectional_note_mask)
        return self.model(
            hidden_states,
            stream=batch["stream_ids"],
            notes=student_branch["notes"],
            notes_mask=effective_mask,
            plan_item_ids=effective_plan_ids,
            plan_item_mask=effective_plan_mask,
            sectional_mask=sectional_mask,
        )

    def _refresh_pre_notes(
        self,
        batch: Dict[str, torch.Tensor],
        student_branch: Dict[str, torch.Tensor],
        *,
        sectional_mask: Optional[torch.Tensor],
    ) -> None:
        lag_tensor = student_branch.get("lag")
        lag_value = (
            int(lag_tensor.item())
            if isinstance(lag_tensor, torch.Tensor)
            else max(0, self.config.curriculum.delta)
        )
        pre_notes, pre_coverage, pre_streams, pre_stride, pre_version = (
            self._extract_notes_from_bus(
                batch,
                notes_bus_key="student_notes_bus",
                mask_key="student_bus_mask",
                coverage_key="student_bus_coverage",
                streams_key="student_bus_streams",
                stride_key="student_bus_stride",
                version_key="student_bus_version",
                lag_override=lag_value + 1,
                sectional_mask=sectional_mask,
            )
        )
        student_branch["pre_notes"] = pre_notes.to(
            device=self.device, dtype=student_branch["notes"].dtype
        )
        pre_mask = (pre_notes.abs().sum(dim=-1) > 0).long()
        if pre_mask.sum() == 0:
            pre_mask = torch.ones_like(pre_mask)
        student_branch["pre_notes_mask"] = pre_mask
        if pre_version is not None:
            student_branch["pre_snapshot_version"] = pre_version
        if pre_stride is not None:
            student_branch["pre_snapshot_stride"] = pre_stride
        if pre_streams is not None:
            student_branch["pre_snapshot_streams"] = pre_streams
        if pre_coverage is not None:
            student_branch["pre_snapshot_coverage"] = pre_coverage
        pre_sectional_mask = self._build_sectional_note_mask(
            sectional_mask=sectional_mask,
            stream_ids=batch["stream_ids"],
            snapshot_streams=pre_streams,
            notes_mask=pre_mask,
            input_ids=batch["input_ids"],
        )
        if pre_sectional_mask is not None:
            student_branch["pre_sectional_note_mask"] = pre_sectional_mask

    def _update_student_bus(
        self,
        batch: Dict[str, torch.Tensor],
        new_notes: torch.Tensor,
        *,
        snapshot_streams: Optional[torch.Tensor],
        coverage: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        bus = batch.get("student_notes_bus")
        mask = batch.get("student_bus_mask")
        if bus is None or mask is None:
            return {}
        coverage_bus = batch.get("student_bus_coverage")
        streams_bus = batch.get("student_bus_streams")
        stride_bus = batch.get("student_bus_stride")
        version_bus = batch.get("student_bus_version")
        bus = bus.to(self.device)
        mask = mask.to(self.device)
        batch_size, max_snapshots, _, _ = bus.shape
        new_notes = new_notes.to(bus.device)
        meta: Dict[str, torch.Tensor] = {}
        if version_bus is not None:
            meta["version"] = torch.zeros(
                batch_size, dtype=version_bus.dtype, device=version_bus.device
            )
        if stride_bus is not None:
            meta["stride"] = torch.zeros(
                batch_size, dtype=stride_bus.dtype, device=stride_bus.device
            )
        stride_meta = meta.get("stride")
        version_meta = meta.get("version")
        streams_per_snapshot = 0
        streams_view: Optional[torch.Tensor] = None
        coverage_view: Optional[torch.Tensor] = None
        if bus.size(1) > 0:
            if snapshot_streams is not None:
                streams_per_snapshot = max(1, snapshot_streams.size(1) // bus.size(1))
                if snapshot_streams.size(1) % streams_per_snapshot == 0:
                    streams_view = snapshot_streams.view(batch_size, -1, streams_per_snapshot)
            if coverage is not None and streams_per_snapshot > 0:
                if coverage.size(1) % streams_per_snapshot == 0:
                    coverage_view = coverage.view(batch_size, -1, streams_per_snapshot)

        freeze_window = max(0, self.config.curriculum.B)
        freeze_tensor = batch.get(_PLAN_SNAPSHOT_FREEZE_KEY)
        if freeze_tensor is not None:
            freeze_tensor = freeze_tensor.to(device=bus.device, dtype=torch.long)
            batch[_PLAN_SNAPSHOT_FREEZE_KEY] = freeze_tensor

        for index in range(batch_size):
            active_count = int(mask[index].sum().item())
            current_max_version = 0
            if version_bus is not None and active_count > 0:
                current_mask = mask[index].clone()
                current_max_version = int(version_bus[index][current_mask].max().item())
            freeze_active = False
            if (
                freeze_tensor is not None
                and bool(freeze_tensor[index].item() > 0)
                and freeze_window > 0
            ):
                freeze_active = True
            if active_count >= max_snapshots:
                shift_start = 1 if (freeze_active and max_snapshots > 1) else 0
                if shift_start >= max_snapshots:
                    continue
                target_slice = slice(shift_start, -1)
                source_slice = slice(shift_start + 1, None)
                if shift_start == 0:
                    bus[index, :-1] = bus[index, 1:]
                    mask[index, :-1] = mask[index, 1:]
                else:
                    bus[index, target_slice] = bus[index, source_slice]
                    mask[index, target_slice] = mask[index, source_slice]
                mask[index, -1] = False
                if coverage_bus is not None:
                    if shift_start == 0:
                        coverage_bus[index, :-1] = coverage_bus[index, 1:]
                    else:
                        coverage_bus[index, target_slice] = coverage_bus[index, source_slice]
                    coverage_bus[index, -1].zero_()
                if streams_bus is not None:
                    if shift_start == 0:
                        streams_bus[index, :-1] = streams_bus[index, 1:]
                    else:
                        streams_bus[index, target_slice] = streams_bus[index, source_slice]
                    streams_bus[index, -1] = -1
                if stride_bus is not None:
                    if shift_start == 0:
                        stride_bus[index, :-1] = stride_bus[index, 1:]
                    else:
                        stride_bus[index, target_slice] = stride_bus[index, source_slice]
                    stride_bus[index, -1] = 0
                if version_bus is not None:
                    if shift_start == 0:
                        version_bus[index, :-1] = version_bus[index, 1:]
                    else:
                        version_bus[index, target_slice] = version_bus[index, source_slice]
                    version_bus[index, -1] = 0
                active_count = max_snapshots - 1
            insert_idx = active_count
            bus[index, insert_idx] = new_notes[index]
            mask[index, insert_idx] = True
            if coverage_bus is not None:
                if coverage_view is not None and coverage_view.size(1) > insert_idx:
                    coverage_slice = coverage_view[index, insert_idx]
                    coverage_bus[index, insert_idx] = coverage_slice.to(
                        device=coverage_bus.device,
                        dtype=coverage_bus[index, insert_idx].dtype,
                    )
                elif coverage is not None and streams_per_snapshot > 0:
                    start = insert_idx * streams_per_snapshot
                    end = min(start + streams_per_snapshot, coverage.size(1))
                    coverage_slice = coverage[index, start:end]
                    if coverage_slice.numel() == coverage_bus.size(-1):
                        target = coverage_slice
                    elif coverage_slice.numel() > 0:
                        padded = torch.zeros_like(coverage_bus[index, insert_idx])
                        padded_flat = padded.view(-1)
                        slice_len = min(padded_flat.numel(), coverage_slice.numel())
                        padded_flat[:slice_len] = coverage_slice[:slice_len]
                        target = padded
                    else:
                        target = None
                    if target is not None:
                        coverage_bus[index, insert_idx] = target.to(
                            device=coverage_bus.device,
                            dtype=coverage_bus[index, insert_idx].dtype,
                        )
                    else:
                        coverage_bus[index, insert_idx].zero_()
                else:
                    coverage_bus[index, insert_idx].zero_()
            if streams_bus is not None:
                if streams_view is not None and streams_view.size(1) > insert_idx:
                    stream_value = streams_view[index, insert_idx, 0]
                    streams_bus[index, insert_idx] = stream_value.to(
                        device=streams_bus.device,
                        dtype=streams_bus.dtype,
                    )
                else:
                    streams_bus[index, insert_idx] = -1
            if stride_bus is not None:
                stride_value = max(1, self.config.curriculum.B)
                stride_bus[index, insert_idx] = stride_value
                if stride_meta is not None:
                    stride_meta[index] = stride_value
            if version_bus is not None:
                new_version = current_max_version + 1
                version_bus[index, insert_idx] = new_version
                if version_meta is not None:
                    version_meta[index] = new_version
        batch["student_notes_bus"] = bus
        batch["student_bus_mask"] = mask
        if coverage_bus is not None:
            batch["student_bus_coverage"] = coverage_bus
        if streams_bus is not None:
            batch["student_bus_streams"] = streams_bus
        if stride_bus is not None:
            batch["student_bus_stride"] = stride_bus
        if version_bus is not None:
            batch["student_bus_version"] = version_bus
        return meta

    def _advance_commit_mask(self, batch: Dict[str, torch.Tensor]) -> None:
        commit_mask = batch.get("commit_mask")
        if commit_mask is None:
            return
        stride = max(1, self.config.curriculum.B)
        commit_mask = commit_mask.to(device=self.device, dtype=torch.bool)
        for index in range(commit_mask.size(0)):
            active_positions = commit_mask[index].nonzero(as_tuple=False).flatten()
            count = active_positions.numel()
            if count <= 1:
                continue
            max_shift = max(count - 1, 0)
            if max_shift == 0:
                continue
            shift = min(stride, max_shift)
            if shift <= 0:
                continue
            release_indices = active_positions[:shift]
            commit_mask[index, release_indices] = False
        batch["commit_mask"] = commit_mask
        if self.config.curriculum.B > 0:
            freeze_tensor = batch.get(_PLAN_SNAPSHOT_FREEZE_KEY)
            if freeze_tensor is not None:
                stride = max(1, self.config.curriculum.B)
                freeze_tensor = freeze_tensor.to(device=self.device, dtype=torch.long)
                freeze_tensor = torch.clamp(freeze_tensor - stride, min=0)
                batch[_PLAN_SNAPSHOT_FREEZE_KEY] = freeze_tensor

    def _uses_separate_teacher(self) -> bool:
        return bool(
            self.config.teacher.enabled
            and self.teacher_model is not None
            and self.teacher_model is not self.model
        )

    def _teacher_encode(
        self,
        batch: Dict[str, torch.Tensor],
        teacher_branch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        model = self.teacher_model if self.teacher_model is not None else self.model
        notes_mask = self._merge_note_masks(
            teacher_branch["notes_mask"],
            teacher_branch.get("sectional_note_mask"),
        )
        with torch.no_grad():
            return self._encode_trunk(
                model,
                batch,
                notes=teacher_branch["notes"],
                notes_mask=notes_mask,
            )

    def _teacher_forward(
        self,
        hidden_states: torch.Tensor,
        *,
        stream: torch.Tensor,
        notes: torch.Tensor,
        notes_mask: torch.Tensor,
        sectional_mask: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not self.config.teacher.enabled and self.teacher_model is None:
            return None
        model = self.teacher_model if self.teacher_model is not None else self.model
        # Unwrap DDP if present to avoid double-forward
        if hasattr(model, "module"):
            model = model.module
        with torch.no_grad():
            return model(
                hidden_states,
                stream=stream,
                notes=notes,
                notes_mask=notes_mask,
                sectional_mask=sectional_mask,
            )

    def _compute_losses(
        self,
        batch: Dict[str, torch.Tensor],
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Optional[Dict[str, torch.Tensor]],
        *,
        student_branch: Dict[str, torch.Tensor],
        teacher_branch: Dict[str, torch.Tensor],
        hidden_states: torch.Tensor,
        stage: int,
        weights: LossWeights,
        step: int,
        pre_update_logits: Optional[torch.Tensor] = None,
        pre_update_lm_logits: Optional[torch.Tensor] = None,
        stability_logging_due: bool = False,
        sectional_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        label_pad = self.collator_config.label_pad_id
        planner_logits_student = student_outputs["planner_logits"]
        vocab = planner_logits_student.size(-1)
        planner_ids = batch["planner_ids"]
        planner_loss = F.cross_entropy(
            planner_logits_student.view(-1, vocab),
            planner_ids.view(-1),
            ignore_index=label_pad,
        )

        teacher_notes = teacher_branch["notes"].to(student_outputs["notes_logits"].dtype)
        student_notes_pred = student_outputs["notes_logits"]
        if student_notes_pred.dim() != 3 or teacher_notes.dim() != 3:
            raise RuntimeError("Expected rank-3 tensors for student and teacher notes.")
        if student_notes_pred.size(2) != teacher_notes.size(2):
            raise RuntimeError(
                f"Notes dim mismatch: student {student_notes_pred.size()} vs teacher {teacher_notes.size()}"
            )
        if student_notes_pred.size(1) != teacher_notes.size(1):
            pooled = student_notes_pred.mean(dim=1, keepdim=True)
            student_notes_pred = pooled.expand(-1, teacher_notes.size(1), -1)
        notes_loss = F.mse_loss(student_notes_pred, teacher_notes)
        spec_pred = student_outputs["speculative_notes"]
        if spec_pred.dim() != 3:
            raise RuntimeError(f"Expected rank-3 speculative_notes, got {spec_pred.shape}.")
        if spec_pred.size(2) != teacher_notes.size(2):
            raise RuntimeError(
                f"Spec notes dim mismatch: student {spec_pred.size()} vs teacher {teacher_notes.size()}"
            )
        if spec_pred.size(1) != teacher_notes.size(1):
            spec_pred = spec_pred.mean(dim=1, keepdim=True).expand(-1, teacher_notes.size(1), -1)
        spec_loss = F.mse_loss(spec_pred, teacher_notes)

        plan_notes_loss = torch.tensor(0.0, device=self.device)
        plan_spec_loss = torch.tensor(0.0, device=self.device)
        teacher_versions = teacher_branch.get("snapshot_version")
        sectional_plan_mask: Optional[torch.Tensor] = None
        if sectional_mask is not None:
            sectional_plan_mask = sectional_mask.to(device=self.device, dtype=torch.bool)
            if sectional_plan_mask.dim() == 0:
                sectional_plan_mask = sectional_plan_mask.view(1)
        if (
            teacher_versions is not None
            and sectional_plan_mask is not None
            and sectional_plan_mask.any()
        ):
            if teacher_versions.dim() != 2:
                teacher_versions = teacher_versions.view(teacher_versions.size(0), -1)
            plan_mask_tensor = teacher_versions == 0
            notes_mask_bool = teacher_branch["notes_mask"].to(device=self.device, dtype=torch.bool)
            if notes_mask_bool.shape == plan_mask_tensor.shape:
                plan_mask_tensor = plan_mask_tensor & notes_mask_bool
            else:
                plan_mask_tensor = plan_mask_tensor & torch.zeros_like(
                    plan_mask_tensor, dtype=torch.bool
                )
            if plan_mask_tensor.size(0) == sectional_plan_mask.size(0):
                sectional_expanded = sectional_plan_mask.view(-1, 1).expand_as(plan_mask_tensor)
                plan_mask_tensor = plan_mask_tensor & sectional_expanded
            else:
                plan_mask_tensor = torch.zeros_like(plan_mask_tensor, dtype=torch.bool)
            if plan_mask_tensor.any():
                plan_teacher = teacher_notes[plan_mask_tensor]
                plan_student = student_notes_pred[plan_mask_tensor]
                plan_spec = spec_pred[plan_mask_tensor]
                plan_notes_loss = F.mse_loss(plan_student, plan_teacher)
                plan_spec_loss = F.mse_loss(plan_spec, plan_teacher)

        spec_kl_loss = torch.tensor(0.0, device=self.device)
        if weights.spec_kl > 0.0:
            spec_predictions = student_outputs["speculative_notes"].to(self.device)
            notes_mask = student_branch["notes_mask"].to(self.device)
            coverage = student_branch.get("coverage")
            spec_kl_loss = self._interhead_spec_kl(
                spec_predictions,
                notes_mask,
                temperature=self.config.spec_kl_temperature,
                coverage=coverage.to(self.device) if isinstance(coverage, torch.Tensor) else None,
            )

        kd_loss = torch.tensor(0.0, device=self.device)
        stability_loss = torch.tensor(0.0, device=self.device)
        lm_ce_loss = torch.tensor(0.0, device=self.device)
        lm_kd_loss = torch.tensor(0.0, device=self.device)
        lm_stab_loss = torch.tensor(0.0, device=self.device)
        planner_mask = batch["planner_mask"].bool()
        commit_mask_tensor = batch.get("commit_mask")
        if commit_mask_tensor is not None:
            commit_mask_tensor = commit_mask_tensor.bool()
        commit_mask = (
            commit_mask_tensor
            if commit_mask_tensor is not None
            else torch.zeros_like(planner_mask, dtype=torch.bool)
        )
        rollback_mask = planner_mask & commit_mask
        stability_mask = planner_mask & (~commit_mask)

        teacher_logits: Optional[torch.Tensor] = None
        if teacher_outputs is not None:
            teacher_logits = teacher_outputs.get("planner_logits")
            if teacher_logits is not None:
                teacher_logits = teacher_logits.detach()
        if teacher_logits is not None and weights.kd > 0.0:
            kd_loss = self._masked_kl(
                planner_logits_student,
                teacher_logits,
                planner_mask,
                temperature=self.config.kd_temperature_planner,
            )
        if pre_update_logits is not None and weights.stab > 0.0:
            stability_loss = self._masked_kl(
                planner_logits_student,
                pre_update_logits.detach(),
                stability_mask,
            )

        # LM CE + KD + stability on token logits
        lm_student = student_outputs.get("lm_logits")
        if lm_student is not None and stage >= 2:
            lm_device = lm_student.device
            labels = batch.get("labels")
            labels_mask = batch.get("labels_mask")
            if labels is not None and labels_mask is not None and labels_mask.any():
                labels = labels.to(device=lm_device)
                labels_mask = labels_mask.to(device=lm_device)
                effective_labels = self._masked_labels(
                    labels,
                    labels_mask,
                    self.collator_config.label_pad_id,
                )
                vocab = lm_student.size(-1)
                lm_ce_loss = F.cross_entropy(
                    lm_student.view(-1, vocab),
                    effective_labels.view(-1),
                    ignore_index=self.collator_config.label_pad_id,
                )
            if teacher_outputs is not None and weights.kd > 0.0:
                lm_teacher = teacher_outputs.get("lm_logits")
                if lm_teacher is not None:
                    lm_teacher = lm_teacher.to(device=lm_device)
                    # Mask with attention to avoid padding
                    attn = batch.get("attention_mask")
                    labels_mask = batch.get("labels_mask")
                    attn = attn.to(device=lm_device) if attn is not None else None
                    labels_mask = (
                        labels_mask.to(device=lm_device) if labels_mask is not None else None
                    )

                    # Use labels_mask if available to respect sectional independence
                    # Otherwise fall back to attention_mask (padding only)
                    kd_mask = (
                        labels_mask if (labels_mask is not None and labels_mask.any()) else attn
                    )
                    if kd_mask is None:
                        kd_mask = (
                            batch["input_ids"].to(device=lm_device)
                            != self.collator_config.pad_token_id
                        ).long()

                    # Ensure teacher logits are on the same device as student logits
                    # This is critical for sharded models where trunk components may be distributed
                    lm_teacher_aligned = lm_teacher.detach().to(device=lm_device)

                    lm_kd_loss = self._masked_kl(
                        lm_student,
                        lm_teacher_aligned,
                        kd_mask.bool(),
                        temperature=self.config.kd_temperature_lm,
                    )
            if pre_update_lm_logits is not None and weights.stab > 0.0:
                attn = batch.get("attention_mask")
                if attn is None:
                    attn = (batch["input_ids"] != self.collator_config.pad_token_id).long()
                attn = attn.to(device=lm_device)
                lm_stab_mask = (~commit_mask).to(device=attn.device)
                lm_stab_loss = self._masked_kl(
                    lm_student,
                    pre_update_lm_logits.detach().to(device=lm_device),
                    lm_stab_mask.bool(),
                )

        usage_loss = torch.tensor(0.0, device=self.device)
        mask_ablation_delta: Optional[float] = None
        log_usage_metric = (
            self.model.training
            and self.config.metrics.mask_ablation_every > 0
            and (step % self.config.metrics.mask_ablation_every == 0)
        )
        penalise_usage = self._should_penalize_usage(stage, weights)
        should_compute_usage = penalise_usage or log_usage_metric

        # CRITICAL DDP FIX: Skip mask ablation on transition steps to avoid "marked ready twice"
        # because agreement_head (and others) might be used in both Main Pass and Mask Ablation.
        is_transition_step = step in self.config.curriculum.stage_schedule
        if is_transition_step and should_compute_usage:
            if self.rank == 0:
                self.logger.warning(
                    "mask_ablation_skipped | step=%d | reason=stage_transition_collision", step
                )
            should_compute_usage = False

        if should_compute_usage:
            # Mask ablation is diagnostic only; do NOT pass plan_item_ids
            # to avoid duplicate gradients on plan_embedding.weight

            # UNWRAP DDP: The usage pass runs a second forward/backward on the same parameters (planner_head).
            # DDP hooks would fire twice, causing "Marked Ready Twice". We must bypass DDP wrapper.
            model_usage = self.model.module if hasattr(self.model, "module") else self.model

            masked_outputs = model_usage(
                hidden_states,
                stream=batch["stream_ids"],
                notes=torch.zeros_like(student_branch["notes"]),
                notes_mask=torch.zeros_like(student_branch["notes_mask"]),
                plan_item_ids=None,
                plan_item_mask=None,
                sectional_mask=sectional_mask,
            )
            masked_logits = masked_outputs["planner_logits"]
            masked_loss = F.cross_entropy(
                masked_logits.view(-1, vocab),
                planner_ids.view(-1),
                ignore_index=label_pad,
            )
            delta = masked_loss - planner_loss
            mask_ablation_delta = float(delta.detach().cpu())
            if penalise_usage:
                usage_loss = self._usage_penalty(delta, stage, weights)

        coverage_loss = torch.tensor(0.0, device=self.device)
        nli_loss = torch.tensor(0.0, device=self.device)
        redundancy_loss = torch.tensor(0.0, device=self.device)
        stream_loss = torch.tensor(0.0, device=self.device)
        agreement_loss = torch.tensor(0.0, device=self.device)
        agreement_precision = None
        coverage_metrics: Dict[str, float] = {}

        auto_agreement_labels = False
        if stage >= 4 and pre_update_logits is not None:
            existing_labels = batch.get("agreement_labels")
            existing_mask = batch.get("agreement_mask")
            has_existing = bool(
                existing_labels is not None
                and existing_mask is not None
                and existing_mask.to(device=self.device).any()
            )
            if not has_existing:
                derived = self._derive_agreement_targets(
                    pre_update_logits.detach(),
                    planner_logits_student.detach(),
                    commit_mask,
                )
                if derived is not None:
                    labels_tensor, mask_tensor = derived
                    batch["agreement_labels"] = labels_tensor
                    batch["agreement_mask"] = mask_tensor
                    auto_agreement_labels = True

        coverage_logits = student_outputs.get("coverage_logits")
        coverage_mask = batch.get("coverage_mask")
        coverage_targets = batch.get("coverage_targets")

        if (
            coverage_logits is not None
            and coverage_targets is not None
            and coverage_mask is not None
        ):
            coverage_logits = coverage_logits.to(self.device)
            coverage_targets = coverage_targets.to(self.device)
            coverage_mask = coverage_mask.to(self.device).bool()
            if coverage_mask.any():
                if weights.cov > 0.0:
                    coverage_loss = F.binary_cross_entropy_with_logits(
                        coverage_logits[coverage_mask],
                        coverage_targets[coverage_mask],
                    )
                coverage_metrics = self._coverage_metrics(
                    coverage_logits,
                    coverage_targets,
                    coverage_mask,
                    plan_item_streams=batch.get("plan_item_streams"),
                    stream_ids=batch["stream_ids"],
                )
        if weights.nli > 0.0 and self.nli_scorer is not None:
            plan_mask = coverage_mask if coverage_mask is not None else batch.get("plan_item_mask")
            if plan_mask is not None:
                plan_mask = plan_mask.to(self.device).bool()
            nli_loss = self._nli_loss(batch, plan_mask)
        if weights.red > 0.0:
            redundancy_loss = self._redundancy_loss(batch)
        if weights.stream > 0.0:
            stream_logits = student_outputs.get("stream_logits")
            if stream_logits is not None:
                stream_loss = F.cross_entropy(stream_logits, batch["stream_ids"])
        if weights.agree > 0.0 and stage >= 3:
            agreement_loss, agreement_precision = self._agreement_loss(
                batch,
                student_outputs,
            )

        should_log_stability = stability_logging_due
        rollback_kl_value = torch.tensor(0.0, device=self.device)
        stability_kl_value = torch.tensor(0.0, device=self.device)
        repair_error_rate: Optional[float] = None
        stability_error_rate: Optional[float] = None
        repair_margin: Optional[float] = None
        stability_margin: Optional[float] = None
        rollback_ratio: Optional[float] = None
        stability_ratio: Optional[float] = None
        if should_log_stability:
            total_tokens = float(planner_mask.sum().item())
            rollback_tokens = float(rollback_mask.sum().item())
            stability_tokens = float(stability_mask.sum().item())
            denom = max(total_tokens, 1.0)
            rollback_ratio = rollback_tokens / denom
            stability_ratio = stability_tokens / denom
            if teacher_logits is not None and rollback_tokens > 0.0:
                rollback_kl_value = self._masked_kl(
                    planner_logits_student, teacher_logits, rollback_mask
                )
            if pre_update_logits is not None and stability_tokens > 0.0:
                post_detached = planner_logits_student.detach()
                pre_detached = pre_update_logits.detach()
                post_vs_pre = self._masked_kl(post_detached, pre_detached, stability_mask)
                pre_vs_post = self._masked_kl(pre_detached, post_detached, stability_mask)
                stability_kl_value = 0.5 * (post_vs_pre + pre_vs_post)

            student_argmax = planner_logits_student.argmax(dim=-1)
            if teacher_logits is not None:
                teacher_topk = (
                    torch.topk(teacher_logits, k=2, dim=-1) if teacher_logits.size(-1) > 1 else None
                )
            else:
                teacher_topk = None
            if teacher_logits is not None and rollback_tokens > 0.0:
                teacher_argmax = teacher_logits.argmax(dim=-1)
                rollback_diff = (student_argmax != teacher_argmax) & rollback_mask
                repair_error_rate = float(rollback_diff.sum().item() / rollback_tokens)
                if teacher_topk is not None and rollback_mask.any():
                    margins = teacher_topk.values[..., 0] - teacher_topk.values[..., 1]
                    repair_margin = float(margins[rollback_mask].mean().item())
                else:
                    repair_margin = 0.0
            if pre_update_logits is not None and stability_tokens > 0.0:
                pre_argmax = pre_update_logits.argmax(dim=-1)
                stability_diff = (student_argmax != pre_argmax) & stability_mask
                stability_error_rate = float(stability_diff.sum().item() / stability_tokens)
                if stability_mask.any() and pre_update_logits.size(-1) > 1:
                    pre_topk = torch.topk(pre_update_logits, k=2, dim=-1)
                    margins = pre_topk.values[..., 0] - pre_topk.values[..., 1]
                    stability_margin = float(margins[stability_mask].mean().item())
                else:
                    stability_margin = 0.0

        total_loss = (
            planner_loss.to(self.device)
            + notes_loss.to(self.device)
            + 0.5 * spec_loss.to(self.device)
            + plan_notes_loss.to(self.device)
            + 0.5 * plan_spec_loss.to(self.device)
            + weights.kd * kd_loss.to(self.device)
            + weights.stab * stability_loss.to(self.device)
            + lm_ce_loss.to(self.device)
            + weights.kd * lm_kd_loss.to(self.device)
            + weights.stab * lm_stab_loss.to(self.device)
            + weights.use * usage_loss.to(self.device)
            + weights.cov * coverage_loss.to(self.device)
            + weights.nli * nli_loss.to(self.device)
            + weights.red * redundancy_loss.to(self.device)
            + weights.spec_kl * spec_kl_loss.to(self.device)
            + weights.stream * stream_loss.to(self.device)
            + weights.agree * agreement_loss.to(self.device)
        )
        metrics = {
            "planner_loss": float(planner_loss.detach().cpu()),
            "notes_loss": float(notes_loss.detach().cpu()),
            "spec_loss": float(spec_loss.detach().cpu()),
            "plan_snapshot_loss": float(plan_notes_loss.detach().cpu()),
            "plan_snapshot_spec_loss": float(plan_spec_loss.detach().cpu()),
            "spec_kl_loss": float(spec_kl_loss.detach().cpu()),
            "kd_loss": float(kd_loss.detach().cpu()),
            "stability_loss": float(stability_loss.detach().cpu()),
            "lm_ce_loss": float(lm_ce_loss.detach().cpu()),
            "lm_kd_loss": float(lm_kd_loss.detach().cpu()),
            "lm_stability_loss": float(lm_stab_loss.detach().cpu()),
            "usage_loss": float(usage_loss.detach().cpu()),
            "coverage_loss": float(coverage_loss.detach().cpu()),
            "nli_loss": float(nli_loss.detach().cpu()),
            "redundancy_loss": float(redundancy_loss.detach().cpu()),
            "stream_loss": float(stream_loss.detach().cpu()),
            "agreement_loss": float(agreement_loss.detach().cpu()),
        }
        if mask_ablation_delta is not None:
            metrics["mask_ablation"] = mask_ablation_delta
        if should_log_stability:
            metrics["rollback_kl"] = float(rollback_kl_value.detach().cpu())
            metrics["stability_kl"] = float(stability_kl_value.detach().cpu())
            metrics["rollback_ratio"] = float(rollback_ratio or 0.0)
            metrics["stability_ratio"] = float(stability_ratio or 0.0)
            metrics["rollback_tokens"] = float(rollback_mask.sum().item())
            metrics["stability_tokens"] = float(stability_mask.sum().item())
            if repair_error_rate is not None:
                metrics["repair_error_rate"] = repair_error_rate
            if stability_error_rate is not None:
                metrics["stability_error_rate"] = stability_error_rate
            if repair_margin is not None:
                metrics["repair_margin"] = repair_margin
            if stability_margin is not None:
                metrics["stability_margin"] = stability_margin
        planner_value = metrics["planner_loss"]
        ratio_source = "planner"
        ratio_numerator = metrics["kd_loss"]
        ratio_denominator = planner_value
        if metrics["lm_ce_loss"] > 0.0:
            ratio_source = "lm"
            ratio_numerator = metrics["lm_kd_loss"]
            ratio_denominator = metrics["lm_ce_loss"]
        if ratio_denominator > 0:
            metrics["kd_ce_ratio"] = ratio_numerator / max(ratio_denominator, 1e-6)
            metrics["kd_ce_ratio_source"] = ratio_source
        if agreement_precision is not None:
            metrics["agreement_precision"] = agreement_precision
        if auto_agreement_labels:
            metrics["agreement_auto"] = 1.0
        if self.model.training:
            self._maybe_adjust_gradnorm(
                kd_loss,
                planner_loss,
                stage,
                lm_kd_loss=lm_kd_loss,
                lm_ce_loss=lm_ce_loss,
            )
            metrics["kd_scale"] = float(self.kd_scale)
        metrics.update(coverage_metrics)
        return total_loss, metrics

    @staticmethod
    def _masked_labels(
        labels: torch.Tensor,
        mask: torch.Tensor,
        pad_id: int,
    ) -> torch.Tensor:
        if labels.shape != mask.shape:
            raise ValueError(
                f"labels and labels_mask must share shape " f"(got {labels.shape} vs {mask.shape})."
            )
        if not mask.dtype == torch.bool:
            mask = mask.to(dtype=torch.bool)
        if mask.all():
            return labels
        masked = labels.clone()
        masked = masked.masked_fill(~mask, pad_id)
        return masked

    def _should_penalize_usage(self, stage: int, weights: LossWeights) -> bool:
        return weights.use > 0.0 and stage >= self.config.usage_min_stage

    def _usage_penalty(self, delta: torch.Tensor, stage: int, weights: LossWeights) -> torch.Tensor:
        if not self._should_penalize_usage(stage, weights):
            return torch.tensor(0.0, device=self.device)
        margin = max(0.0, self.config.usage_margin)
        return torch.relu(margin - delta)

    def _coverage_metrics(
        self,
        coverage_logits: torch.Tensor,
        coverage_targets: torch.Tensor,
        coverage_mask: torch.Tensor,
        *,
        plan_item_streams: Optional[torch.Tensor],
        stream_ids: torch.Tensor,
    ) -> Dict[str, float]:
        mask = coverage_mask.to(device=self.device, dtype=torch.bool)
        coverage_logits = coverage_logits.to(device=self.device)
        coverage_targets = coverage_targets.to(device=self.device)

        if coverage_logits.shape != coverage_targets.shape:
            raise ValueError(
                "coverage_logits and coverage_targets must share shape "
                f"(got {coverage_logits.shape} vs {coverage_targets.shape})."
            )
        if mask.shape != coverage_targets.shape:
            raise ValueError(
                "coverage_mask must match coverage_targets shape "
                f"(got {mask.shape} vs {coverage_targets.shape})."
            )
        if mask.numel() == 0 or not bool(mask.any()):
            return {}
        probs = torch.sigmoid(coverage_logits)
        predictions = probs >= self.config.coverage_threshold
        targets = coverage_targets >= 0.5

        tp = float((predictions & targets & mask).sum().item())
        fp = float((predictions & (~targets) & mask).sum().item())
        fn = float((~predictions & targets & mask).sum().item())
        tn = float((~predictions & (~targets) & mask).sum().item())

        predicted_positive = tp + fp
        positive_support = tp + fn
        negative_support = fp + tn

        precision = tp / predicted_positive if predicted_positive > 0 else 0.0
        recall = tp / positive_support if positive_support > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0

        metrics: Dict[str, float] = {
            "coverage_precision": precision,
            "coverage_recall": recall,
            "coverage_f1": f1,
            "coverage_tp": tp,
            "coverage_fp": fp,
            "coverage_fn": fn,
            "coverage_support_pos": positive_support,
            "coverage_support_neg": negative_support,
            "coverage_threshold": float(self.config.coverage_threshold),
        }

        if plan_item_streams is not None and plan_item_streams.shape == coverage_targets.shape:
            streams_tensor = plan_item_streams.to(device=self.device, dtype=torch.long)
            stream_ids_tensor = stream_ids.to(device=self.device, dtype=torch.long).view(-1, 1)
            stream_ids_tensor = stream_ids_tensor.expand_as(streams_tensor)
            valid_streams = streams_tensor >= 0
            same_stream_mask = mask & valid_streams & (streams_tensor == stream_ids_tensor)
            cross_stream_mask = mask & valid_streams & (streams_tensor != stream_ids_tensor)

            same_stream_tp = float((predictions & targets & same_stream_mask).sum().item())
            same_stream_positive = float((targets & same_stream_mask).sum().item())
            cross_stream_fp = float((predictions & (~targets) & cross_stream_mask).sum().item())
            cross_stream_total = float(cross_stream_mask.sum().item())

            metrics["coverage_same_stream_recall"] = (
                same_stream_tp / same_stream_positive if same_stream_positive > 0 else 0.0
            )
            metrics["coverage_same_stream_support"] = same_stream_positive
            metrics["coverage_cross_stream_fp_rate"] = (
                cross_stream_fp / cross_stream_total if cross_stream_total > 0 else 0.0
            )
            metrics["coverage_cross_stream_support"] = cross_stream_total

        return metrics

    def _masked_kl(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="none") * (temperature**2)
        kl = kl.sum(dim=-1)
        mask = mask.to(dtype=kl.dtype, device=kl.device)
        denom = mask.sum().clamp_min(1.0)
        return (kl * mask).sum() / denom

    def _interhead_spec_kl(
        self,
        speculative_notes: torch.Tensor,
        notes_mask: torch.Tensor,
        *,
        temperature: float,
        coverage: Optional[torch.Tensor] = None,
        min_overlap: float = 1e-5,
    ) -> torch.Tensor:
        if temperature <= 0.0:
            raise ValueError("spec_kl_temperature must be positive.")
        if speculative_notes.dim() != 3:
            raise ValueError(
                f"Expected speculative_notes to have shape [batch, notes, dim], got {speculative_notes.shape}."
            )
        if notes_mask.dim() != 2:
            raise ValueError(
                f"Expected notes_mask to have shape [batch, notes], got {notes_mask.shape}."
            )
        batch_size, _, feature_dim = speculative_notes.shape
        if feature_dim == 0:
            return torch.tensor(0.0, device=speculative_notes.device)
        target_len = speculative_notes.size(1)
        if notes_mask.size(1) > target_len:
            notes_mask = notes_mask[:, :target_len]
        elif notes_mask.size(1) < target_len:
            pad = torch.zeros(
                notes_mask.size(0),
                target_len - notes_mask.size(1),
                device=notes_mask.device,
                dtype=notes_mask.dtype,
            )
            notes_mask = torch.cat([notes_mask, pad.long()], dim=1)
        coverage_tensor: Optional[torch.Tensor] = None
        if coverage is not None:
            if coverage.dim() != 2:
                raise ValueError(
                    f"Expected coverage to have shape [batch, notes], got {coverage.shape}."
                )
            if coverage.size(1) > target_len:
                coverage = coverage[:, :target_len]
            elif coverage.size(1) < target_len:
                pad_cov = torch.zeros(
                    coverage.size(0),
                    target_len - coverage.size(1),
                    device=coverage.device,
                    dtype=coverage.dtype,
                )
                coverage = torch.cat([coverage, pad_cov], dim=1)
            coverage_tensor = coverage.to(device=speculative_notes.device, dtype=torch.float32)

        total = torch.tensor(0.0, device=speculative_notes.device)
        weight_total = torch.tensor(0.0, device=speculative_notes.device)
        notes_mask = notes_mask.to(device=speculative_notes.device, dtype=torch.bool)

        for batch_index in range(batch_size):
            active_indices = notes_mask[batch_index].nonzero(as_tuple=False).flatten()
            if active_indices.numel() < 2:
                continue
            sample_notes = speculative_notes[batch_index, active_indices]
            log_probs = F.log_softmax(sample_notes / temperature, dim=-1)
            probs = log_probs.exp()
            log_ratio = log_probs.unsqueeze(1) - log_probs.unsqueeze(0)
            kl_matrix = torch.sum(probs.unsqueeze(1) * log_ratio, dim=-1)
            sym_kl = kl_matrix + kl_matrix.transpose(0, 1)
            indices = torch.triu_indices(
                sym_kl.size(0), sym_kl.size(1), offset=1, device=sym_kl.device
            )
            if indices.numel() == 0:
                continue
            pair_values = sym_kl[indices[0], indices[1]]

            if coverage_tensor is not None:
                sample_cov = coverage_tensor[batch_index, active_indices]
                weights = torch.minimum(sample_cov[indices[0]], sample_cov[indices[1]])
                weights = weights.clamp_min(0.0)
                if min_overlap > 0.0:
                    active = weights > min_overlap
                    if not torch.any(active):
                        continue
                    pair_values = pair_values[active]
                    weights = weights[active]
            else:
                weights = torch.ones_like(pair_values, device=pair_values.device)

            total = total + torch.sum(pair_values * weights)
            weight_total = weight_total + torch.sum(weights)

        if weight_total.item() == 0.0:
            return torch.tensor(0.0, device=speculative_notes.device)
        return total / weight_total

    def _update_teacher_ema(self) -> None:
        if self.teacher_model is None or self.teacher_model is self.model:
            return
        decay = self.config.teacher.ema_decay
        with torch.no_grad():
            for t_param, s_param in zip(self.teacher_model.parameters(), self.model.parameters()):
                if not t_param.requires_grad:
                    # Ensure student param is on the same device as teacher param
                    s_data = s_param.data.to(t_param.device)
                    t_param.data.mul_(decay).add_(s_data, alpha=1 - decay)

    def _nli_loss(
        self,
        batch: Dict[str, torch.Tensor],
        plan_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.nli_scorer is None:
            return torch.tensor(0.0, device=self.device)
        notes_text = batch.get("notes_text")
        plan_text = batch.get("plan_text")
        if notes_text is None or plan_text is None:
            return torch.tensor(0.0, device=self.device)
        pairs: List[Tuple[str, str]] = []
        for batch_index, (note, items) in enumerate(zip(notes_text, plan_text)):
            if not note or not items:
                continue
            item_mask = None
            if plan_mask is not None and plan_mask.size(0) > batch_index:
                item_mask = plan_mask[batch_index]
            for item_index, item in enumerate(items):
                if item_mask is not None and item_mask.size(0) > item_index:
                    if not bool(item_mask[item_index]):
                        continue
                pairs.append((note, item))
        if not pairs:
            return torch.tensor(0.0, device=self.device)
        probs = self.nli_scorer.score(pairs)
        if probs.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        contra_idx = self.nli_scorer.label_index.get("contradiction", 0)
        neutral_idx = self.nli_scorer.label_index.get("neutral", 1)
        contradiction = probs[:, contra_idx]
        neutral = probs[:, neutral_idx]
        margin = self.config.nli_margin
        penalties = torch.relu(contradiction - neutral - margin)
        return penalties.mean()

    def _redundancy_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        student_bus = batch.get("student_notes_bus")
        mask = batch.get("student_bus_mask")
        if student_bus is None or mask is None:
            return torch.tensor(0.0, device=self.device)
        notes = student_bus.to(self.device, dtype=torch.float32)
        mask = mask.to(self.device)
        streams = notes.size(-2)
        if streams < 2 or mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        total = torch.tensor(0.0, device=self.device)
        count = torch.tensor(0.0, device=self.device)
        margin = 0.7
        for batch_index in range(notes.size(0)):
            for snapshot_index in range(notes.size(1)):
                if not mask[batch_index, snapshot_index]:
                    continue
                snapshot = notes[batch_index, snapshot_index]
                for i in range(streams):
                    for j in range(i + 1, streams):
                        sim = F.cosine_similarity(
                            snapshot[i].unsqueeze(0), snapshot[j].unsqueeze(0), dim=-1
                        )
                        total = total + torch.relu(sim - margin)
                        count = count + 1
        if count.item() == 0:
            return torch.tensor(0.0, device=self.device)
        return total / count

    def _derive_agreement_targets(
        self,
        pre_logits: torch.Tensor,
        post_logits: torch.Tensor,
        commit_mask: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        commit_mask = commit_mask.to(device=post_logits.device, dtype=torch.bool)
        if commit_mask.sum().item() == 0:
            return None
        pre_logits = pre_logits.to(device=post_logits.device)
        pre_log_probs = F.log_softmax(pre_logits, dim=-1)
        post_log_probs = F.log_softmax(post_logits, dim=-1)
        pre_probs = pre_log_probs.exp()
        post_probs = post_log_probs.exp()
        kl_pre_post = F.kl_div(pre_log_probs, post_probs, reduction="none").sum(dim=-1)
        kl_post_pre = F.kl_div(post_log_probs, pre_probs, reduction="none").sum(dim=-1)
        sym_kl = 0.5 * (kl_pre_post + kl_post_pre)
        same_argmax = pre_logits.argmax(dim=-1) == post_logits.argmax(dim=-1)
        stable = same_argmax & (sym_kl <= self.config.agreement_threshold)
        batch_size, sequence_length = stable.shape
        commit_counts = commit_mask.sum(dim=1)
        max_commit = int(commit_counts.max().item())
        if max_commit == 0:
            return None
        labels = torch.zeros(
            (batch_size, max_commit),
            dtype=torch.float32,
            device=post_logits.device,
        )
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for batch_index in range(batch_size):
            count = int(commit_counts[batch_index].item())
            if count == 0:
                continue
            slot_start = max_commit - count
            token_indices = commit_mask[batch_index].nonzero(as_tuple=False).flatten()
            values = stable[batch_index, token_indices]
            labels[batch_index, slot_start:] = values[-count:].float()
            mask[batch_index, slot_start:] = True
        return labels, mask

    def _agreement_loss(
        self,
        batch: Dict[str, torch.Tensor],
        student_outputs: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[float]]:
        agreement_labels = batch.get("agreement_labels")
        agreement_mask = batch.get("agreement_mask")
        if agreement_labels is None or agreement_mask is None:
            return torch.tensor(0.0, device=self.device), None
        labels = agreement_labels.to(self.device)
        mask = agreement_mask.to(self.device)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device), None
        scores = student_outputs["agreement"].squeeze(-1).to(self.device)
        pooled = torch.zeros_like(labels, dtype=scores.dtype)
        sequence_length = scores.size(1)
        slots = labels.size(1)
        for batch_index in range(scores.size(0)):
            start = max(0, sequence_length - slots)
            selected = scores[batch_index, start:sequence_length]
            pooled[batch_index, -selected.size(0) :] = selected
        flat_mask = mask.view(-1)
        valid_indices = flat_mask.nonzero(as_tuple=False).view(-1)
        if valid_indices.numel() == 0:
            return torch.tensor(0.0, device=self.device), None
        pooled_flat = pooled.view(-1)
        labels_flat = labels.view(-1).float()
        logits = pooled_flat[valid_indices]
        targets = labels_flat[valid_indices]
        loss = F.binary_cross_entropy(logits, targets)
        preds = (logits > 0.5).float()
        predicted_positive = preds.sum()
        if predicted_positive.item() == 0:
            precision = 1.0
        else:
            correct_positive = (((preds == targets) & (preds == 1)).float()).sum()
            precision = float((correct_positive / predicted_positive).detach().cpu())
        self._update_agreement_stats(logits.detach(), targets.detach())
        return loss, precision

    def _update_agreement_stats(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        if logits.numel() == 0:
            return
        logits_cpu = logits.detach().to(device="cpu", dtype=torch.float32)
        targets_cpu = targets.detach().to(device="cpu", dtype=torch.float32)
        for tau, stats in self._agreement_stats.items():
            preds = (logits_cpu > tau).float()
            tp = ((preds == 1.0) & (targets_cpu == 1.0)).sum().item()
            fp = ((preds == 1.0) & (targets_cpu == 0.0)).sum().item()
            fn = ((preds == 0.0) & (targets_cpu == 1.0)).sum().item()
            tn = ((preds == 0.0) & (targets_cpu == 0.0)).sum().item()
            stats["tp"] += tp
            stats["fp"] += fp
            stats["fn"] += fn
            stats["tn"] += tn

    def _compute_agreement_roc(self) -> List[Dict[str, float]]:
        points: List[Dict[str, float]] = []
        for tau in self._agreement_thresholds:
            stats = self._agreement_stats[tau]
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]
            if tp + fp + fn == 0:
                precision = 0.0
                recall = 0.0
            else:
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            points.append(
                {
                    "threshold": float(tau),
                    "precision": float(precision),
                    "recall": float(recall),
                    "tp": float(tp),
                    "fp": float(fp),
                    "fn": float(fn),
                    "tn": float(stats["tn"]),
                }
            )
        points.sort(key=lambda item: item["threshold"])
        return points

    def _maybe_recalibrate_agreement_threshold(
        self, store_points: bool = False
    ) -> Optional[Dict[str, Any]]:
        total_samples = sum(
            stats["tp"] + stats["fp"] + stats["fn"] + stats["tn"]
            for stats in self._agreement_stats.values()
        )
        if total_samples < 10:
            return (
                {"points": self._compute_agreement_roc(), "current": {}} if store_points else None
            )
        roc_points = self._compute_agreement_roc()
        if not roc_points:
            return None
        best = max(
            roc_points,
            key=lambda item: (
                item["precision"],
                item["recall"],
                item["threshold"],
            ),
        )
        new_tau = best["threshold"]
        if abs(new_tau - self.config.agreement_threshold) > 1e-3:
            self.logger.info(
                "agreement_threshold_update | old=%.3f | new=%.3f | precision=%.3f | recall=%.3f",
                self.config.agreement_threshold,
                new_tau,
                best["precision"],
                best["recall"],
            )
            self.config.agreement_threshold = float(new_tau)
            self._last_agreement_update_step = self.state.global_step
        if store_points:
            return {"points": roc_points, "current": best}
        return None

    def _maybe_adjust_gradnorm(
        self,
        kd_loss: torch.Tensor,
        planner_loss: torch.Tensor,
        stage: int,
        *,
        lm_kd_loss: Optional[torch.Tensor] = None,
        lm_ce_loss: Optional[torch.Tensor] = None,
    ) -> None:
        cfg = self.config.gradnorm
        if not cfg.enabled or stage < 2:
            return
        if kd_loss.isnan() or planner_loss.isnan():
            return
        use_lm = (
            lm_kd_loss is not None
            and lm_ce_loss is not None
            and not lm_kd_loss.isnan()
            and not lm_ce_loss.isnan()
            and float(lm_ce_loss.detach().abs().item()) > 0.0
        )
        if use_lm:
            ce_tensor = lm_ce_loss.detach().abs().clamp_min(1e-6).float()
            kd_tensor = lm_kd_loss.detach().abs().float()
        else:
            ce_tensor = planner_loss.detach().abs().clamp_min(1e-6).float()
            kd_tensor = kd_loss.detach().abs().float()
        if torch.isnan(ce_tensor) or torch.isnan(kd_tensor):
            return
        ce_value = max(float(ce_tensor.item()), 1e-6)
        kd_value = float(kd_tensor.item())
        ratio = kd_value / ce_value
        target = cfg.target_ratio
        error = ratio - target
        new_scale = self.kd_scale * (1.0 - cfg.alpha * error)
        new_scale = max(cfg.min_scale, min(cfg.max_scale, new_scale))
        self.kd_scale = new_scale

    def evaluate(self) -> Dict[str, float]:
        if self.eval_dataset is None:
            return {}

        # DDP-safe evaluation: Use DistributedSampler to shard validation set
        # This prevents race conditions on cache writes AND speeds up eval by 8x
        sampler = None
        shuffle = False
        if self.is_ddp:
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(
                self.eval_dataset,
                shuffle=False,  # Keep order consistent across runs
                drop_last=False,  # Important: don't drop incomplete batches in eval
            )
            shuffle = False  # Must be False when sampler is provided

        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=self.collator,
        )
        self.model.eval()
        losses: list[float] = []
        stage = self._determine_stage()
        active_weights = self._active_loss_weights(stage)
        coverage_tp = 0.0
        coverage_fp = 0.0
        coverage_fn = 0.0
        coverage_items = 0.0
        coverage_used_logits = False
        contradiction_pairs = 0
        contradiction_count = 0
        margin_violation_total = 0.0
        redundancy_total = 0.0
        redundancy_pairs = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = {
                    key: value.to(self.device) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                plan_item_ids = batch.get("plan_item_ids")
                plan_item_mask = batch.get("plan_item_mask")
                if plan_item_ids is not None:
                    plan_item_ids = plan_item_ids.to(self.device)
                if plan_item_mask is not None:
                    plan_item_mask = plan_item_mask.to(self.device)
                sectional_mask = self._sectional_mask(batch)
                teacher_branch, student_branch = self._prepare_branches_for_batch(
                    batch,
                    stage,
                    sectional_mask,
                    batch["input_ids"],
                )
                student_encode_mask = self._merge_note_masks(
                    student_branch["notes_mask"],
                    student_branch.get("sectional_note_mask"),
                )
                hidden_states = self._encode_trunk(
                    self.model,
                    batch,
                    notes=student_branch["notes"],
                    notes_mask=student_encode_mask,
                )
                teacher_branch["notes"] = teacher_branch["notes"].to(hidden_states.dtype)
                student_branch["notes"] = student_branch["notes"].to(hidden_states.dtype)
                if "pre_notes" in student_branch:
                    student_branch["pre_notes"] = student_branch["pre_notes"].to(
                        hidden_states.dtype
                    )
                if self._uses_separate_teacher():
                    teacher_hidden = self._teacher_encode(batch, teacher_branch)
                else:
                    teacher_hidden = hidden_states.detach()
                student_outputs = self._run_student_pass(
                    hidden_states,
                    batch,
                    student_branch,
                    stage=stage,
                    plan_item_ids=plan_item_ids,
                    plan_item_mask=plan_item_mask,
                    sectional_mask=sectional_mask,
                )
                teacher_notes_mask = self._merge_note_masks(
                    teacher_branch["notes_mask"],
                    teacher_branch.get("sectional_note_mask"),
                )
                teacher_outputs = self._teacher_forward(
                    teacher_hidden,
                    stream=batch["stream_ids"],
                    notes=teacher_branch["notes"],
                    notes_mask=teacher_notes_mask,
                    sectional_mask=sectional_mask,
                )
                pre_update_logits: Optional[torch.Tensor] = None
                if (
                    active_weights.stab > 0.0
                    and "pre_notes" in student_branch
                    and "pre_notes_mask" in student_branch
                ):
                    pre_mask_effective = self._merge_note_masks(
                        student_branch["pre_notes_mask"],
                        student_branch.get("pre_sectional_note_mask"),
                    )
                    # Pre-update pass in eval is diagnostic only; do NOT pass plan_item_ids
                    pre_outputs = self.model(
                        hidden_states,
                        stream=batch["stream_ids"],
                        notes=student_branch["pre_notes"],
                        notes_mask=pre_mask_effective,
                        plan_item_ids=None,
                        plan_item_mask=None,
                        sectional_mask=sectional_mask,
                    )
                    pre_update_logits = pre_outputs["planner_logits"]
                total_loss, _ = self._compute_losses(
                    batch,
                    student_outputs,
                    teacher_outputs,
                    student_branch=student_branch,
                    teacher_branch=teacher_branch,
                    hidden_states=hidden_states,
                    stage=stage,
                    weights=active_weights,
                    step=self.state.global_step,
                    pre_update_logits=pre_update_logits,
                    stability_logging_due=False,
                    sectional_mask=sectional_mask,
                )
                losses.append(float(total_loss.detach().cpu()))
                coverage_targets = batch.get("coverage_targets")
                coverage_mask = batch.get("coverage_mask")
                notes_text = batch.get("notes_text") or []
                plan_text = batch.get("plan_text") or []
                coverage_threshold = float(
                    self.config.coverage_threshold
                    if self.config.coverage_threshold is not None
                    else 0.5
                )
                logits_used_this_batch = False
                if (
                    student_outputs.get("coverage_logits") is not None
                    and plan_item_mask is not None
                    and coverage_targets is not None
                ):
                    logits = student_outputs["coverage_logits"].detach()
                    mask_bool = plan_item_mask.to(dtype=torch.bool, device=logits.device)
                    if coverage_mask is not None:
                        mask_bool = mask_bool & coverage_mask.to(
                            dtype=torch.bool, device=logits.device
                        )
                    if mask_bool.any():
                        preds = torch.sigmoid(logits) >= coverage_threshold
                        targets_bool = coverage_targets.to(device=preds.device) >= 0.5
                        coverage_tp += float(((preds & targets_bool) & mask_bool).sum().item())
                        coverage_fp += float(((preds & ~targets_bool) & mask_bool).sum().item())
                        coverage_fn += float(((~preds & targets_bool) & mask_bool).sum().item())
                        coverage_items += float(mask_bool.sum().item())
                        logits_used_this_batch = True
                if not logits_used_this_batch and coverage_targets is not None:
                    mask_cpu = (
                        plan_item_mask.detach().cpu().bool() if plan_item_mask is not None else None
                    )
                    targets_cpu = coverage_targets.detach().cpu()
                    coverage_mask_cpu = (
                        coverage_mask.detach().cpu().bool() if coverage_mask is not None else None
                    )
                    for batch_index, items in enumerate(plan_text):
                        if not isinstance(items, (list, tuple)):
                            continue
                        note_str = ""
                        if batch_index < len(notes_text) and isinstance(
                            notes_text[batch_index], str
                        ):
                            note_str = notes_text[batch_index]
                        note_lower = note_str.lower()
                        mask_row = (
                            mask_cpu[batch_index]
                            if mask_cpu is not None and batch_index < mask_cpu.size(0)
                            else None
                        )
                        coverage_row = (
                            coverage_mask_cpu[batch_index]
                            if coverage_mask_cpu is not None
                            and batch_index < coverage_mask_cpu.size(0)
                            else None
                        )
                        target_row = (
                            targets_cpu[batch_index] if batch_index < targets_cpu.size(0) else None
                        )
                        for item_index, raw_item in enumerate(items):
                            if mask_row is not None:
                                if item_index >= mask_row.numel() or not bool(mask_row[item_index]):
                                    continue
                            if coverage_row is not None:
                                if item_index >= coverage_row.numel() or not bool(
                                    coverage_row[item_index]
                                ):
                                    continue
                            target_val = 0.0
                            if target_row is not None and item_index < target_row.numel():
                                target_val = float(target_row[item_index].item())
                            item_text = str(raw_item)
                            predicted = 1 if item_text.lower() in note_lower else 0
                            if target_val >= 0.5:
                                if predicted:
                                    coverage_tp += 1.0
                                else:
                                    coverage_fn += 1.0
                            else:
                                if predicted:
                                    coverage_fp += 1.0
                            coverage_items += 1.0
                if logits_used_this_batch:
                    coverage_used_logits = True

                if self.nli_scorer is not None:
                    pairs: List[Tuple[str, str]] = []
                    mask_cpu = (
                        plan_item_mask.detach().cpu().bool() if plan_item_mask is not None else None
                    )
                    for batch_index, items in enumerate(plan_text):
                        if not isinstance(items, (list, tuple)):
                            continue
                        if batch_index >= len(notes_text):
                            continue
                        note_str = notes_text[batch_index]
                        if not isinstance(note_str, str) or not note_str.strip():
                            continue
                        mask_row = (
                            mask_cpu[batch_index]
                            if mask_cpu is not None and batch_index < mask_cpu.size(0)
                            else None
                        )
                        for item_index, raw_item in enumerate(items):
                            if mask_row is not None:
                                if item_index >= mask_row.numel() or not bool(mask_row[item_index]):
                                    continue
                            item_text = str(raw_item).strip()
                            if not item_text:
                                continue
                            pairs.append((note_str, item_text))
                    if pairs:
                        probs = self.nli_scorer.score(pairs)
                        if probs.numel() > 0:
                            contra_idx = self.nli_scorer.label_index.get("contradiction", 0)
                            neutral_idx = self.nli_scorer.label_index.get("neutral", 1)
                            prediction = probs.argmax(dim=-1)
                            contradiction_count += int((prediction == contra_idx).sum().item())
                            contradiction_pairs += probs.size(0)
                            violations = torch.relu(
                                probs[:, contra_idx]
                                - probs[:, neutral_idx]
                                - self.config.nli_margin
                            )
                            margin_violation_total += float(violations.sum().item())

                notes_bus = batch.get("student_notes_bus")
                notes_mask = batch.get("student_bus_mask")
                if notes_bus is None or notes_mask is None or not bool(notes_mask.sum().item()):
                    notes_bus = batch.get("teacher_notes_bus")
                    notes_mask = batch.get("teacher_bus_mask")
                if (
                    notes_bus is not None
                    and notes_mask is not None
                    and bool(notes_mask.sum().item())
                ):
                    bus_tensor = notes_bus.to(device=self.device, dtype=torch.float32)
                    mask_tensor = notes_mask.to(device=self.device, dtype=torch.bool)
                    streams_dim = bus_tensor.size(-2)
                    margin = 0.7
                    for batch_index in range(bus_tensor.size(0)):
                        for snapshot_index in range(bus_tensor.size(1)):
                            if not mask_tensor[batch_index, snapshot_index]:
                                continue
                            snapshot = bus_tensor[batch_index, snapshot_index]
                            for i in range(streams_dim):
                                for j in range(i + 1, streams_dim):
                                    sim = F.cosine_similarity(
                                        snapshot[i].unsqueeze(0),
                                        snapshot[j].unsqueeze(0),
                                        dim=-1,
                                    )
                                    score = torch.relu(sim - margin)
                                    redundancy_total += float(score.item())
                                    redundancy_pairs += 1

        # Aggregate metrics across all ranks in DDP
        # Each rank only processed 1/Nth of the validation set due to DistributedSampler
        # We need to sum counters and average losses correctly
        if self.is_ddp:
            import torch.distributed as dist

            # Pack all metrics into a tensor for efficient all_reduce
            local_metrics = torch.tensor(
                [
                    sum(losses),  # 0: sum of losses
                    len(losses),  # 1: number of batches
                    coverage_tp,  # 2
                    coverage_fp,  # 3
                    coverage_fn,  # 4
                    coverage_items,  # 5
                    float(coverage_used_logits),  # 6
                    float(contradiction_pairs),  # 7
                    float(contradiction_count),  # 8
                    margin_violation_total,  # 9
                    redundancy_total,  # 10
                    float(redundancy_pairs),  # 11
                ],
                device=self.device,
                dtype=torch.float32,
            )

            # Sum across all ranks
            dist.all_reduce(local_metrics, op=dist.ReduceOp.SUM)

            # Unpack aggregated metrics
            total_loss_sum = local_metrics[0].item()
            total_batches = int(local_metrics[1].item())
            coverage_tp = local_metrics[2].item()
            coverage_fp = local_metrics[3].item()
            coverage_fn = local_metrics[4].item()
            coverage_items = local_metrics[5].item()
            coverage_used_logits = bool(local_metrics[6].item() > 0)
            contradiction_pairs = int(local_metrics[7].item())
            contradiction_count = int(local_metrics[8].item())
            margin_violation_total = local_metrics[9].item()
            redundancy_total = local_metrics[10].item()
            redundancy_pairs = int(local_metrics[11].item())

            # Calculate global average loss
            avg_loss = float(total_loss_sum / max(1, total_batches))
        else:
            # Single-process: just average locally
            avg_loss = float(sum(losses) / max(1, len(losses)))

        self.model.train()
        if self.teacher_model is not None and self.teacher_model is not self.model:
            self.teacher_model.eval()
        metrics = {"eval_loss": avg_loss}
        if contradiction_pairs > 0:
            metrics["contradiction_rate"] = float(contradiction_count) / float(contradiction_pairs)
            metrics["avg_margin_violation"] = margin_violation_total / float(contradiction_pairs)
        else:
            metrics["contradiction_rate"] = None
            metrics["avg_margin_violation"] = None
        metrics["nli_pair_count"] = contradiction_pairs
        if coverage_items > 0:
            precision = (
                coverage_tp / (coverage_tp + coverage_fp)
                if (coverage_tp + coverage_fp) > 0
                else None
            )
            recall = (
                coverage_tp / (coverage_tp + coverage_fn)
                if (coverage_tp + coverage_fn) > 0
                else None
            )
            if precision is not None and recall is not None and (precision + recall) > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = None
            metrics["coverage_precision"] = precision
            metrics["coverage_recall"] = recall
            metrics["coverage_f1"] = f1
            metrics["coverage_support"] = coverage_items
            metrics["coverage_source"] = "logits" if coverage_used_logits else "text"
        else:
            metrics["coverage_precision"] = None
            metrics["coverage_recall"] = None
            metrics["coverage_f1"] = None
            metrics["coverage_support"] = 0
            metrics["coverage_source"] = None
        if redundancy_pairs > 0:
            metrics["redundancy_index"] = redundancy_total / float(redundancy_pairs)
        else:
            metrics["redundancy_index"] = None
        metrics["redundancy_pair_count"] = redundancy_pairs
        if avg_loss < self.state.best_eval_loss:
            self.state.best_eval_loss = avg_loss
        self._log_metrics("eval", metrics)
        return metrics

    def _log_metrics(self, prefix: str, metrics: Dict[str, float]) -> None:
        # Only rank 0 should log to avoid duplicate output in DDP
        if self.rank != 0:
            return

        roc_summary: Optional[Dict[str, Any]] = None
        if prefix == "train" and "agreement_precision" in metrics:
            roc_summary = self._maybe_recalibrate_agreement_threshold(store_points=True)
            if roc_summary is not None:
                current = roc_summary.get("current", {})
                metrics["agreement_tau"] = float(self.config.agreement_threshold)
                metrics["agreement_precision_tau"] = float(current.get("precision", 0.0))
                metrics["agreement_recall_tau"] = float(current.get("recall", 0.0))

        def _format(value: object) -> str:
            if isinstance(value, (int, float)):
                return f"{float(value):.4f}"
            return str(value)

        message = " | ".join(f"{key}={_format(val)}" for key, val in metrics.items())
        print(f"{prefix}: step={self.state.global_step} | {message}")
        record = dict(metrics)
        if roc_summary is not None:
            record["agreement_roc"] = roc_summary.get("points", [])
        record["step"] = self.state.global_step
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        history = self.metric_history.setdefault(prefix, [])
        history.append(record)
