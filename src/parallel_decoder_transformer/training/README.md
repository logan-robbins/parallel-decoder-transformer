# Training Pipeline

Parameter-efficient training for the Parallel Decoder Transformer with a frozen GPT-OSS-20B trunk. The training system attaches lightweight sidecar modules (<5% of total parameters) and trains them through a 4-stage curriculum using two-branch knowledge distillation. All teacher notes are pre-generated during the [dataset pipeline](../datasets/README.md) — training makes no LLM API calls.

See [PAPER.md](../../../docs/arxiv_submission/PAPER.md) Appendix A for the mathematical definitions of all loss terms and the curriculum rationale.

## Pre-Trained Checkpoints

Pre-trained adapter checkpoints are available:

**https://storage.googleapis.com/parallel-decoder-transformer/checkpoints/gpt-oss-8xH100-50000steps/**

```bash
mkdir -p experiments/gpt_oss

# Final checkpoint (50k steps, 77.8% coverage precision)
wget https://storage.googleapis.com/parallel-decoder-transformer/checkpoints/gpt-oss-8xH100-50000steps/adapters_step_50000.pt \
  -O experiments/gpt_oss/adapters_step_50000.pt

# Training metadata
wget https://storage.googleapis.com/parallel-decoder-transformer/checkpoints/gpt-oss-8xH100-50000steps/train_manifest.json \
  -O experiments/gpt_oss/train_manifest.json
wget https://storage.googleapis.com/parallel-decoder-transformer/checkpoints/gpt-oss-8xH100-50000steps/agreement_thresholds.json \
  -O experiments/gpt_oss/agreement_thresholds.json
```

## Prerequisites

1. **Completed dataset pipeline** — `kd_train.jsonl`, `kd_validation.jsonl`, `kd_test.jsonl` in `data/processed/pdt_10k_gpt41/` (or download pre-generated datasets)
2. **GPT-OSS-20B weights** — in `gpt-oss-20b/original/` (see project README)
3. **Hardware** — 8x NVIDIA B200 GPUs (180GB VRAM each) for production training

```bash
uv sync
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Two-Branch Architecture

Every training step runs two forward passes through the frozen trunk:

1. **Teacher branch** — conditioned on ground-truth notes (pre-generated). Produces supervision targets.
2. **Student branch** — conditioned on speculative/noisy notes. Learns to match teacher output despite lower-quality notes.

This enables **Speculative Invariance**: the model produces similar outputs regardless of note quality, matching the paper's objective p(Z | N_hat) ≈ p(Z | N).

The teacher branch operates in one of two modes (configured via `teacher.type`):
- `"stop_grad"` (default) — same model, gradients detached. More memory-efficient.
- `"ema"` — separate model updated via exponential moving average (`ema_decay: 0.995`).

## 4-Stage Curriculum

Training a coordination mechanism on a frozen trunk is unstable if all modules are enabled simultaneously. The curriculum progressively unfreezes modules and activates loss terms.

### Stage 0: Planner Pretrain

**Steps**: 0–3,749 (production) | **Trainable**: planner head, notes head, `plan_notes_proj`

Bootstraps the prompt-time latent planner. The planner learns to predict S = 16 latent slot IDs from a 65,536-entry vocabulary via cross-entropy against hashed plan catalog entries. The `plan_notes_proj` layer learns to project plan embeddings into notes space for snapshot 0.

**Active losses**: `L_plan` (planner CE), `L_notes` (notes MSE), `L_spec` (speculation MSE, weight 0.5), plan projection alignment

### Stage 1: Stream Bootstrap

**Steps**: 3,750–9,999 (production) | **Trainable**: stream adapters, SNC cross-attention, stream classifier

Streams learn stream-specific conditioning under teacher supervision. `bus_mix_prob: 1.0` forces teacher notes while speculation is still frozen.

**Active losses**: All Stage 0 losses + adapter-related losses

### Stage 2: Bus Enablement (Extended)

**Steps**: 10,000–24,999 (production, 15,000 steps) | **Trainable**: speculation head (all others remain trainable)

Streams learn to write latent summaries into the versioned Dynamic Notes Bus. `bus_mix_prob: 0.75` jumpstarts usage with teacher notes.

**Active losses**: All previous + `L_KD` (KD planner KL), `L_spec_kl` (inter-head speculative KL), `L_LM-CE` (language model CE), `L_LM-KD` (language model KD)

### Stage 3: Commit Control (Extended)

**Steps**: 25,000–50,000 (production, 25,000 steps) | **Trainable**: agreement head, coverage head (all others remain trainable)

Ownership consistency and continuation sufficiency become part of the generation policy. `bus_mix_prob: 0.35` (35% teacher notes), `stream_dropout_prob: 0.15` (15% random stream zeroing).

**Active losses**: All previous + `L_cov` (coverage BCE), `L_ready` (agreement/readiness BCE), `L_use` (usage/mask-ablation penalty), `L_stab` (stability KL)

### Stage 4: Trunk Unfreezing (Not Supported)

Defined in the paper and dev config but **not feasible on B200 hardware** (requires >190GB VRAM per GPU). The production config omits Stage 4 entirely. Production achieves 77.8% coverage precision through parameter-efficient training (Stages 0–3 only).

## Loss Functions

The total loss combines all active terms:

```
L_total = L_plan + L_notes + 0.5 * L_spec
        + plan_notes_loss + 0.5 * plan_spec_loss + plan_proj_align_loss
        + w_kd * L_KD + w_stab * L_stab
        + L_LM-CE + w_kd * L_LM-KD + w_stab * L_LM-stab
        + w_use * L_use + w_cov * L_cov
        + w_nli * L_nli + w_red * L_red + w_retain * L_retain
        + w_spec_kl * L_spec_kl + w_stream * L_stream
        + w_agree * L_agree
```

### Paper-Defined Losses (Appendix A)

| Loss | Formula | Purpose |
|------|---------|---------|
| L_plan | CE(planner_logits, planner_ids) | Latent plan slot prediction |
| L_notes | MSE(student_notes_pred, teacher_notes) | Notes head alignment |
| L_spec | MSE(spec_pred, teacher_notes) | Speculation head alignment (weight 0.5) |
| L_LM-CE | CE(lm_logits, labels) | Language model token prediction (stage ≥ 2) |
| L_KD-LM | KL(student_LM ‖ teacher_LM) × T² | LM knowledge distillation |
| L_cov | BCEWithLogits(coverage_logits, coverage_targets) | Plan item ownership prediction |
| L_ready | BCE(agreement_logits, agreement_labels) | Continuation sufficiency / commit readiness |

### Implementation-Specific Losses

| Loss | Purpose | Config Key |
|------|---------|------------|
| L_stab | Pre/post notes-update consistency (Speculative Invariance) | `stab` |
| L_use | Penalizes ignoring notes via mask-ablation forward pass | `use` |
| L_spec_kl | Symmetric KL between speculative note pairs, weighted by coverage overlap | `spec_kl` |
| L_nli | Semantic entailment verification (requires `nli_scorer`) | `nli` |
| L_red | Penalizes cosine similarity > 0.7 between stream notes in same snapshot | `red` |
| L_retain | Penalizes cosine similarity > 0.5 between coverage vectors of different snapshots | `retain` |
| L_stream | Stream classification cross-entropy | `stream` |

### Default Loss Weights

From `configs/gpt_oss_transfer.yaml`:

```yaml
loss_weights:
  kd: 2.0          # Knowledge distillation
  stab: 0.1        # Stability
  use: 1.0         # Usage / mask-ablation
  cov: 1.0         # Coverage
  nli: 0.05        # NLI entailment
  red: 0.0         # Redundancy (off)
  spec_kl: 0.1     # Speculative KL
  stream: 0.0      # Stream classifier (off)
  agree: 1.0       # Agreement / readiness
  retain: 0.0      # Retention (off)
```

Losses activate progressively by stage via `_active_loss_weights()` in `trainer.py`.

## Configuration

### Dev Config (`configs/gpt_oss_transfer.yaml`)

For local development on a single GPU. 1,000 steps, includes Stage 4 in schedule.

Key parameters:
- `hidden_size: 2880`, `vocab_size: 201088` (real GPT-OSS-20B dimensions)
- `notes_dim: 2048`, `num_heads: 32`, `plan_vocab_size: 65536`
- `batch_size: 2`, `grad_accumulation: 4`, `lr: 1e-4`
- `max_steps: 1000`, `warmup_steps: 100`
- Stage schedule: `[0, 150, 400, 700, 900]` (5 stages)
- `attn_implementation: "flex"` (PyTorch Flex Attention)
- Dataset: `pdt_30k/kd_train.jsonl`

### Production Config (`configs/gpt_oss_transfer_production.yaml`)

For 8x B200 distributed training. 50,000 steps, no Stage 4.

| Parameter | Dev | Production |
|-----------|-----|------------|
| `batch_size` | 2 | 1 |
| `grad_accumulation` | 4 | 16 (effective batch = 16) |
| `learning_rate` | 1e-4 | 2e-4 |
| `weight_decay` | 0.0 | 0.01 |
| `max_steps` | 1,000 | 50,000 |
| `warmup_steps` | 100 | 1,250 |
| `save_every` | 500 | 2,500 |
| `eval_interval` | 200 | 10,000 |
| `attn_implementation` | `flex` | `flash_attention_2` |
| `gradient_checkpointing` | absent | `true` |
| `device_map` | `auto` | `null` |
| Stage schedule | `[0, 150, 400, 700, 900]` | `[0, 3750, 10000, 25000]` |

### Stage Policies

Each stage has a `StagePolicyConfig` that controls:
- `freeze` / `unfreeze` — module names to toggle `requires_grad`
- `bus_mix_prob` — probability of substituting teacher notes for student notes
- `stream_dropout_prob` — probability of zeroing out random stream slots
- `notes_noise` — Gaussian perturbation (`paraphrase_p`, sigma=0.05) and zero-out (`drop_p`)
- `loss_weights` — per-stage loss weight overrides

Supported module identifiers: `trunk`, `stream_adapters`, `cross_attention`, `notes_bus`, `planner_head`, `notes_head`, `speculation_head`, `agreement_head`, `coverage_head`, `stream_classifier`, `plan_embedding`, `plan_notes_proj`. Aliases: `heads`, `all_heads`.

## Training Execution

### Local Development

```bash
uv run scripts/train.py --config configs/gpt_oss_transfer.yaml
```

### Production (8x B200 with WandB)

```bash
# One-time WandB setup
uv run wandb login

# SSH to GPU instance, setup, transfer data
ssh ubuntu@<instance-ip>
bash scripts/setup_lambda_gpu.sh  # ~15-30 min (downloads GPT-OSS-20B)
rsync -avz data/processed/pdt_10k_gpt41/ ubuntu@<instance-ip>:parallel-decoder-transformer/data/processed/pdt_10k_gpt41/

# Launch training
export WANDB_API_KEY=$(cat wandb.txt)
export PDT_DEVICE=cuda
nohup uv run torchrun --nproc_per_node=8 scripts/train_wandb.py \
  --config configs/gpt_oss_transfer_production.yaml > nohup.out 2>&1 &
```

### What Happens During Training

1. **Config loading** — YAML loaded, nested dicts coerced to config dataclasses via `_coerce_model_config()` and `_coerce_training_config()`
2. **Dataset loading** — `KDJsonlDataset` builds a lazy offset index over all JSONL lines for O(1) random access (does not load all data into memory)
3. **Model init** — GPT-OSS-20B trunk loaded with `bfloat16`, distributed via `device_map`, lower 4 layers frozen. All PDT heads initialized.
4. **Teacher branch** — `DatasetTeacherNotesProvider` reads pre-generated notes from JSONL metadata. Wrapped in `CachedTeacherNotesProvider` for disk caching.
5. **Collator** — `TwoBranchKnowledgeDistillationCollator` pads to `max_length`, builds 4D notes bus tensors `[B, max_snapshots, streams, notes_dim]`, generates plan item IDs via content hashing
6. **Training loop** — for each step: determine stage → apply freeze/unfreeze policy → teacher forward → student forward → compute losses → backprop → gradient accumulation → optimizer step → EMA update → log/eval/checkpoint
7. **Checkpoint resume** — finds latest `adapters_step_*.pt` in `telemetry_dir`, loads adapter weights and training state. Re-applies stage policy after resume (because `requires_grad` is ephemeral).

## Data Flow

### JSONL → Batch Tensors

The `KDJsonlDataset` (`dataset.py`) deserializes each JSONL line into a `KDRecord` with fields: `student_ids`, `student_labels`, `planner_ids`, `notes_student`, `notes_teacher`, `stream_id`, `teacher_snapshots`, `student_snapshots`, `plan_items`, `plan_catalog`, `coverage_targets`, `coverage_supervision_mask`, `continuation_sufficiency_labels`, `metadata`.

The collator transforms records into training batch tensors:

```python
batch = {
    "input_ids": [B, 8192],           # Padded input sequences
    "attention_mask": [B, 8192],
    "labels": [B, 8192],              # Shifted for LM loss
    "planner_ids": [B, 16],           # Hashed latent plan slots
    "planner_mask": [B, 16],
    "stream_ids": [B],                # Which stream (0, 1, or 2)
    "notes_student": [B, 3, 2048],    # Speculative notes
    "notes_teacher": [B, 3, 2048],    # Ground-truth notes
    "teacher_notes_bus": [B, 4, 3, 2048],  # Dynamic Notes Bus
    "plan_item_ids": [B, max_items],  # Hashed plan items for coverage
    "coverage_targets": [B, max_items],
    "coverage_mask": [B, max_items],
    "sectional_independence": [B],
    # ... additional bus masks, strides, versions, coverage
}
```

### Notes Bus Extraction

`_extract_notes_from_bus()` applies the lag parameter: drops the last `lag` valid snapshots from visibility (most recent notes are not yet committed). **Exception**: for sectional-independence samples, lag is set to 0 (all snapshots visible — independent streams need no synchronization delay).

## Threshold Calibration

### Agreement Threshold (Auto-Calibrated)

The trainer maintains ROC statistics across 19 thresholds (0.05–0.95) for the agreement head. `_maybe_recalibrate_agreement_threshold()` selects the threshold maximizing `(precision, recall, threshold)` and **auto-updates** `config.agreement_threshold` during training.

Written to `agreement_thresholds.json` at training completion.

### Coverage Threshold (Report-Only)

The coverage head uses a fixed threshold (`coverage_threshold: 0.4` by default). The trainer accumulates per-threshold statistics and writes them to `coverage_thresholds.json`, but the F1-optimal threshold is **not** auto-applied — it is logged to WandB as `coverage_threshold_opt` for monitoring only.

Post-hoc analysis on a saved manifest (CPU-only, no GPU):

```bash
uv run scripts/coverage_threshold_sweep.py \
    --manifest experiments/eval_manifest.json \
    --bootstrap 1000 \
    --output figures/coverage_pr_curve.png \
    --csv results/coverage_threshold_sweep.csv
```

To apply a better threshold for subsequent runs:
```yaml
training:
  coverage_threshold: 0.20  # Updated from sweep analysis
```

## Monitoring

### Key Metrics

| Metric | Target | Meaning |
|--------|--------|---------|
| `kd_ce_ratio` | > 0.8 | Student/teacher alignment |
| `mask_ablation` | > 0.15 | Notes dependency (model uses notes) |
| `stability_kl` | < 0.2 | Pre/post notes-update consistency |
| `coverage_precision` | > 0.6 | Plan item ownership accuracy |
| `agreement_precision` | > 0.7 | Commit readiness accuracy (by Stage 3) |
| `rollback_kl` | < 0.5 | Stability under note perturbations |

### WandB Dashboard

`train_wandb.py` logs all metrics as `{prefix}/{key}` (e.g., `train/kd_ce_ratio`, `eval/coverage_f1`). Also logs `global_step`, `epoch`, `stage`. Final `adapters.pt` is uploaded as a WandB artifact.

### Monitoring Commands

```bash
# Live logs
tail -f nohup.out

# GPU utilization
nvidia-smi -l 1

# Process status
ps -ef | grep train_wandb

# Stop training (kills DDP workers)
pkill -9 -f torchrun && pkill -9 -f train_wandb
```

## Telemetry Outputs

All written to `telemetry_dir` (default: `experiments/gpt_oss/`):

| File | Contents |
|------|----------|
| `adapters_step_{N}.pt` | Checkpoint with adapter weights + training state |
| `adapters.pt` | Final adapter weights only |
| `train_manifest.json` | Config path, dataset, global step, best eval loss, git SHA |
| `train_run_stages.json` | Stage transition history with timestamps and freeze/unfreeze actions |
| `training_report.json` | Aggregated metrics (`last`, `mean`, `min`, `max` for each key metric) |
| `agreement_thresholds.json` | ROC curve with auto-calibrated threshold |
| `coverage_thresholds.json` | ROC curve across 19 thresholds with F1-optimal point |

Checkpoint cleanup retains the 3 most recent checkpoints plus any at stage transition steps.

## Module Reference

### `trainer.py`

Core training loop (~3,500 lines). Key classes and methods:

**Configuration**: `TrainingConfig`, `TrainerState`, `TeacherBranchConfig`, `CurriculumConfig`, `LossWeights`, `StagePolicyConfig`, `NotesNoiseConfig`, `MetricsConfig`, `NegativeSamplingConfig`, `GradNormConfig`

**Stage management**: `_determine_stage()`, `_on_stage_transition()`, `_apply_stage_policy()`, `_update_trainable()`, `_resolve_policy_modules()`

**Training loop**: `fit()`, `_training_step()`, `_prepare_branch_inputs()`, `_encode_trunk()`, `_run_student_pass()`, `_compute_losses()`

**Threshold calibration**: `_update_agreement_stats()`, `_compute_agreement_roc()`, `_maybe_recalibrate_agreement_threshold()`, `_update_coverage_stats()`, `_compute_coverage_roc()`, `_maybe_recalibrate_coverage_threshold()`

**Advanced features**: `_maybe_apply_negative_sampling()` (adversarial plan items), `_interhead_spec_kl()` (coverage-weighted symmetric KL), parallel micro-steps (`parallel_micro_steps > 0`)

### `dataset.py`

Lazy-loading JSONL dataset. `KDJsonlDataset` builds an offset index for O(1) random access. `_decode()` handles all deserialization: tensor conversion, stream ID normalization, snapshot construction, plan catalog extraction, coverage supervision mask computation.

Coverage score mapping: `COVERED → 1.0`, `PARTIAL → 0.5`, `MISSING → 0.0`.

## DDP Safety Notes

The trainer explicitly handles several DDP edge cases:
- Pre-update (stability) pass skipped on stage transition steps to avoid "marked ready twice" errors
- Mask ablation pass similarly skipped on transition steps
- Usage pass unwraps DDP explicitly to avoid double-backward
- `find_unused_parameters=True` in DDP wrapping (frozen params are unused in early stages)
- `plan_item_ids` only passed when `coverage_head` has trainable parameters

## Troubleshooting

**OOM**: Reduce `batch_size` (try 1), reduce `max_length` (try 4096), enable `gradient_checkpointing: true`, use `bfloat16`.

**Notes not used (`mask_ablation` near 0)**: Increase `loss_weights.use`, verify `cross_attention.gating_init` is negative (start with gate closed), check `bus_mix_prob` is not too high, ensure student notes have variance.

**Poor KD alignment (`kd_ce_ratio` < 0.5)**: Increase `loss_weights.kd`, verify `teacher.enabled: true`, check teacher notes are non-zero, lower learning rate, extend Stage 0.

**Loss spikes / NaN**: Check `max_grad_norm: 1.0` is set, reduce learning rate, increase `warmup_steps`, check for malformed examples.

**Resume after failure**: Re-run with the same command. The trainer finds the latest checkpoint in `telemetry_dir` and restores step, optimizer state, and scheduler state. Stage policy is re-applied after resume.
