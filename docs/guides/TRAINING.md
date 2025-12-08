# Training Pipeline

Production training workflow for the Parallel Decoder Transformer (PDT) with frozen GPT-OSS-20B trunk. This document describes parameter-efficient training **after** completing the dataset pipeline (documented in `DATASET_PIPELINE.md`).

## Pre-Trained Checkpoints

Pre-trained adapter checkpoints from the production training run are available:

**https://storage.googleapis.com/parallel-decoder-transformer/checkpoints/gpt-oss-8xH100-50000steps/**

### Download Checkpoints

```bash
# Create checkpoint directory
mkdir -p experiments/gpt_oss

# Final checkpoint (50k steps, 71.6% coverage precision)
wget https://storage.googleapis.com/parallel-decoder-transformer/checkpoints/gpt-oss-8xH100-50000steps/adapters_step_50000.pt \
  -O experiments/gpt_oss/adapters_step_50000.pt

# Training manifests
wget https://storage.googleapis.com/parallel-decoder-transformer/checkpoints/gpt-oss-8xH100-50000steps/train_manifest.json \
  -O experiments/gpt_oss/train_manifest.json

wget https://storage.googleapis.com/parallel-decoder-transformer/checkpoints/gpt-oss-8xH100-50000steps/training_report.json \
  -O experiments/gpt_oss/training_report.json

wget https://storage.googleapis.com/parallel-decoder-transformer/checkpoints/gpt-oss-8xH100-50000steps/agreement_thresholds.json \
  -O experiments/gpt_oss/agreement_thresholds.json
```

**Available checkpoints:**
- `adapters_step_50000.pt` - Final checkpoint (recommended)
- `adapters_step_47500.pt` - Late-stage checkpoint
- `adapters_step_45000.pt` - Late-stage checkpoint
- `adapters_step_25000.pt` - Mid-training (Stage 3 start)
- `adapters_step_22500.pt` - Mid-training (Stage 2 end)

## Overview

PDT training uses **parameter-efficient knowledge distillation** with a frozen 20B trunk:
- **Teacher branch**: Generates supervision from ground-truth notes (pre-generated)
- **Student branch**: Learns correct outputs from noisy/hallucinated notes via **Speculative Invariance**
- **Frozen trunk**: GPT-OSS-20B parameters remain frozen (only adapters/heads trainable)

**Hardware requirements**: 8x NVIDIA B200 GPUs (180GB VRAM each)  
**Training duration**: 50,000 steps (~30 hours)  
**Final results**: 71.6% coverage precision on plan item prediction

Teacher notes are pre-generated during Stage 3 of the dataset pipeline. Training does NOT call any LLM APIs.

## Prerequisites

### 1. Completed Dataset Pipeline

You must have completed all 5 stages of the dataset pipeline OR downloaded pre-generated datasets:

**Option A: Download pre-generated datasets (recommended)**
```bash
# See DATASET_PIPELINE.md "Pre-Generated Datasets" section
wget https://storage.googleapis.com/parallel-decoder-transformer/data/archives/pdt_10k_gpt41_jsonl_train.tar.gz
wget https://storage.googleapis.com/parallel-decoder-transformer/data/archives/pdt_10k_gpt41_jsonl_eval.tar.gz
tar -xzf pdt_10k_gpt41_jsonl_train.tar.gz -C data/processed/
tar -xzf pdt_10k_gpt41_jsonl_eval.tar.gz -C data/processed/
```

**Option B: Generate from scratch**
```bash
# Follow complete pipeline in DATASET_PIPELINE.md
```

**Verify dataset exists:**
```bash
ls data/datasets/pdt_10k_gpt41/
# Should contain: train.parquet, validation.parquet, test.parquet, manifest.json

ls data/processed/pdt_10k_gpt41/
# Should contain: kd_train.jsonl, kd_validation.jsonl, kd_test.jsonl
```

### 2. Environment Setup

```bash
# Ensure virtual environment is active
uv sync

# Verify GPU availability (required for GPT-OSS-20B)
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. Model Weights

Download GPT-OSS-20B weights to `gpt-oss-20b/original/` directory (see project README).

## Training Configuration

Training is configured via `configs/gpt_oss_transfer.yaml`. Key sections:

### Model Configuration

```yaml
model:
  hidden_size: 4096          # GPT-OSS-20B hidden dimension
  vocab_size: 32000          # Vocabulary size
  notes_dim: 2048            # Notes embedding dimension
  num_heads: 32              # Attention heads
  plan_vocab_size: 65536     # Plan hash buckets
  
  trunk:
    base_model: "gpt-oss-20b/original"
    torch_dtype: "bfloat16"
    device_map: "auto"       # Distribute across available GPUs
    freeze_lower_layers: 4   # Freeze bottom 4 transformer layers
    
  notes_bus:
    snapshot_dim: 2048
    max_snapshots: 4         # Dynamic notes bus capacity
    lag: 1                   # Lag between speculation and commitment
    
  cross_attention:
    hidden_size: 4096
    notes_dim: 2048
    num_heads: 32
    gating_init: -4.0        # Start with gate nearly closed
```

### Training Configuration

**Production config** (`configs/gpt_oss_transfer_production.yaml`):

```yaml
training:
  dataset_path: "data/processed/pdt_10k_gpt41/kd_train.jsonl"
  eval_dataset_path: "data/processed/pdt_10k_gpt41/kd_validation.jsonl"
  telemetry_dir: "experiments/gpt_oss"
  
  batch_size: 1              # Reduced for memory constraints
  grad_accumulation: 16      # Maintains effective batch size of 16
  learning_rate: 2.0e-4
  weight_decay: 0.01
  max_steps: 50000           # Full production training
  warmup_steps: 1250
  save_every: 2500           # Checkpoint interval
  
  log_interval: 25
  eval_interval: 10000
  
  teacher:
    enabled: true
    type: "stop_grad"        # Shared model with detached gradients
    ema_decay: 0.995
    
  dataset_teacher:
    cache_dir: "data/teacher_cache"
    max_snapshots: 4
```

### Curriculum Configuration

Training proceeds through **4 stages** (Stage 4 trunk unfreezing is not supported on B200 hardware):

```yaml
training:
  curriculum:
    B: 4                     # Max snapshots in dynamic notes bus
    L: 32                    # Commit horizon (tokens)
    delta: 1                 # Lag parameter
    stage_schedule:          # Stage transitions at these steps
      - 0                    # Stage 0: Planner Pretrain (0-3,750)
      - 3750                 # Stage 1: Adapter Bootstrap (3,750-10,000)
      - 10000                # Stage 2: Notes Bus / SNC Training (10,000-25,000) - EXTENDED
      - 25000                # Stage 3: Rollback / Coverage+Agreement (25,000-50,000) - EXTENDED
```

#### Stage 0: Planner Pretrain (Steps 0-3,749)
- **Duration**: 3,750 steps (~6 epochs)
- **Objective**: Bootstrap planner and notes heads
- **Frozen**: Trunk, stream adapters, cross-attention, speculation, agreement, coverage
- **Trainable**: Planner head, notes head
- **Active Losses**: Planner loss only

#### Stage 1: Adapter Bootstrap (Steps 3,750-9,999)
- **Duration**: 6,250 steps (~11 epochs)
- **Objective**: Learn stream-specific adaptations
- **Frozen**: Trunk, speculation, agreement, coverage
- **Trainable**: Stream adapters, cross-attention, stream classifier
- **Active Losses**: Adapter loss
- **Noise**: `bus_mix_prob: 1.0` (force teacher notes while speculation frozen)

#### Stage 2: Notes Bus / SNC Training - EXTENDED (Steps 10,000-24,999)
- **Duration**: 15,000 steps (2× original allocation)
- **Objective**: Enable dynamic notes bus and speculation mechanism
- **Frozen**: Trunk, agreement, coverage
- **Trainable**: Speculation head (all others remain trainable)
- **Active Losses**: `kd`, `spec_kl`
- **Noise**: `bus_mix_prob: 0.75` (jumpstart usage with teacher notes)

#### Stage 3: Rollback / Coverage+Agreement - EXTENDED (Steps 25,000-50,000)
- **Duration**: 25,000 steps (5× original allocation)
- **Objective**: Train agreement and coverage heads for self-correction
- **Frozen**: Trunk only (parameter-efficient training)
- **Trainable**: Agreement head, coverage head (all others remain trainable)
- **Active Losses**: `kd`, `use`, `coverage`, `agreement`, `stability`
- **Noise**: 
  - `bus_mix_prob: 0.35` - 35% teacher notes
  - `stream_dropout_prob: 0.15` - 15% stream dropout

#### Stage 4: Trunk Unfreezing (NOT SUPPORTED)
**Hardware constraint**: Stage 4 requires >190GB VRAM per GPU. B200 GPUs (180GB) cannot run this stage.

The production curriculum achieves 71.6% coverage precision through parameter-efficient training (Stages 0-3 only) without trunk unfreezing.

### Loss Weights

```yaml
training:
  loss_weights:
    kd: 1.0          # Knowledge distillation (teacher-student alignment)
    stab: 0.1        # Stability loss (pre/post update consistency)
    use: 0.0         # Usage loss (penalize ignoring notes)
    cov: 0.2         # Coverage loss (plan item coverage prediction)
    nli: 0.05        # NLI loss (semantic entailment)
    red: 0.0         # Redundancy loss
    spec_kl: 0.1     # Speculative KL divergence
    role: 0.0        # Stream classification loss
    agree: 0.0       # Agreement prediction loss
```

Losses activate progressively by stage (see `_active_loss_weights()` in trainer).

## Training Execution

### Local Development

```bash
uv run python scripts/train.py --config configs/gpt_oss_transfer.yaml
```

### Production (Remote with WandB)

```bash
# Setup WandB first (one-time)
uv run wandb login

# Run with monitoring
tmux new -s training
uv run python scripts/train_wandb.py --config configs/gpt_oss_transfer.yaml
```

See "Training on Lambda Labs 8x H100" section below for full remote setup.

### What Happens

1. **Configuration Loading**
   - Loads model and training config from YAML
   - Seeds RNG for reproducibility
   - Validates stage schedule and parameters

2. **Dataset Loading**
   - Creates `KDJsonlDataset` from `data/processed/pdt_10k/kd_train.jsonl`
   - Creates eval dataset from `data/processed/pdt_10k/kd_validation.jsonl` (if provided)
   - Each line is a stream-level training example with:
     - `student_ids`: Tokenized input sequence
     - `planner_ids`: Tokenized plan tokens
     - `notes_student`: Speculative (noisy) notes embeddings
     - `notes_teacher`: Ground-truth notes (pre-generated)
     - `metadata`: Contains teacher plan, document text, versioned snapshots

3. **Model Initialization**
   - Loads GPT-OSS-20B trunk with specified dtype (`bfloat16`)
   - Distributes model across GPUs via `device_map: "auto"`
   - Freezes lower 4 layers of trunk
   - Initializes all PDT heads (planner, notes, speculation, agreement, coverage)

4. **Teacher Branch Setup**
   - If `teacher.type == "ema"`: Creates separate EMA teacher model
   - If `teacher.type == "stop_grad"`: Uses same model with `detach()` (default)
   - **Teacher notes provider** reads pre-generated notes from dataset metadata
   - **No LLM calls** - all notes extracted from JSONL records

5. **Collator Configuration**
   - Creates `TwoBranchKnowledgeDistillationCollator`
   - Batches examples and pads to `max_length: 8192`
   - Builds dynamic notes bus tensors:
     - `teacher_notes_bus`: [batch, max_snapshots, streams, notes_dim]
     - `student_notes_bus`: [batch, max_snapshots, streams, notes_dim]
   - Generates plan item IDs via content hashing (`plan_hash_buckets: 65536`)

6. **Training Loop**
   - For each step until `max_steps`:
     - Determine current curriculum stage
     - Apply stage-specific freeze/unfreeze policies
     - Sample batch from DataLoader
     - **Teacher forward pass**:
       - Encode input with trunk: `hidden = trunk(input_ids)`
       - Apply stream adapters: `adapted = stream_adapters(stream, hidden)`
       - Apply cross-attention: `attended = cross_attn(adapted, teacher_notes)`
       - Generate teacher outputs: planner logits, notes logits, etc.
     - **Student forward pass**:
       - Reuse trunk encoding (efficiency)
       - Apply stream adapters with student (noisy) notes
       - Apply cross-attention with student notes
       - Generate student outputs
     - **Compute losses**:
       - `kd_loss`: KL divergence between student/teacher planner logits
       - `stability_loss`: Pre/post update consistency (tests Speculative Invariance)
       - `coverage_loss`: Plan item coverage prediction
       - `spec_kl_loss`: Speculative notes KL divergence
       - Other auxiliary losses
     - Scale losses by `loss_weights` and curriculum stage
     - Backpropagate and accumulate gradients
     - Update optimizer every `grad_accumulation` steps
     - Update EMA teacher if applicable
     - Log metrics every `log_interval` steps
     - Evaluate on `eval_dataset` every `eval_interval` steps

7. **Checkpointing & Telemetry**
   - Writes to `telemetry_dir` (default: `experiments/gpt_oss/`):
     - `train_manifest.json`: Training metadata, final metrics
     - `train_run_stages.json`: Stage transition history
     - `training_report.json`: Aggregated training metrics
     - `agreement_thresholds.json`: ROC curve for agreement threshold
     - `adapters.pt`: Lightweight adapter checkpoint (only PDT parameters)

## Training Data Flow

### JSONL Record Structure

Each line in the split-specific JSONL files (e.g., `kd_train.jsonl`) represents **one stream** from **one document**:

```json
{
  "example_id": "survey_7825850_e13d6bbe_stream_1",
  "sample_id": "survey_7825850_e13d6bbe",
  "stream_id": "stream_1",
  "domain": "survey",
  "split": "train",
  
  "student_ids": [101, 7865, ...],           // Tokenized input (8192 tokens)
  "student_labels": [7865, 2548, ...],       // Labels (shifted for LM loss)
  "planner_ids": [101, 2059, ...],           // Tokenized plan
  
  "notes_student": [[0.1, 0.2, ...], ...],   // Speculative notes (3 x 2048)
  "notes_teacher": [[0.12, 0.19, ...], ...], // Ground-truth notes (3 x 2048)
  
  "metadata": {
    "document_text": "Maienfeld is a municipality...",
    "document_paragraphs": ["Maienfeld is...", "The town..."],
    "teacher_plan": {
      "plan": [
        {
          "stream_id": "stream_1",
          "header": "Early History",
          "summary": "Covers Maienfeld's location...",
          "notes": ["Must identify canton", "Must mention Alps"]
        },
        // ... streams 2 and 3
      ],
      "segments": [
        {"stream": "stream_1", "paragraph_start": 0, "paragraph_end": 2}
      ]
    },
    "teacher_notes": {
      "stream_1": ["Maienfeld", "Graubünden", "municipality"],
      "stream_2": [...],
      "stream_3": [...]
    },
    "notes_versioned": [
      {
        "version": 0,
        "stride": 0,
        "stream_notes": {
          "stream_1": {"ENT": [...], "FACT": [...], "COVERAGE": [...]},
          "stream_2": {...},
          "stream_3": {...}
        },
        "coverage": {"stream_1": 1.0, "stream_2": 0.8, "stream_3": 1.0}
      },
      // ... snapshots 1-3
    ],
    "sectional_independence": true
  },
  
  "true_notes": [
    {
      "stream_id": "stream_1",
      "ENT": [{"id": "E1", "name": "Maienfeld", "type": "municipality", ...}],
      "FACT": [{"subj_id": "E1", "predicate": "located_in", "object": "Graubünden", ...}],
      "COVERAGE": [{"plan_item_id": "item_1", "status": "complete"}]
    }
  ]
}
```

### Collation Process

The `TwoBranchKnowledgeDistillationCollator` transforms JSONL records into training batches:

```python
batch = {
    # Input sequences (padded to max_length)
    "input_ids": torch.LongTensor,          # [B, 8192]
    "attention_mask": torch.LongTensor,     # [B, 8192]
    "labels": torch.LongTensor,             # [B, 8192]
    "planner_ids": torch.LongTensor,        # [B, 8192]
    
    # Stream assignments
    "stream_ids": torch.LongTensor,         # [B] - which stream (0, 1, or 2)
    
    # Notes tensors
    "notes_student": torch.FloatTensor,     # [B, 3, 2048] - speculative notes
    "notes_teacher": torch.FloatTensor,     # [B, 3, 2048] - ground-truth notes
    
    # Dynamic notes bus (teacher)
    "teacher_notes_bus": torch.FloatTensor,     # [B, 4, 3, 2048]
    "teacher_bus_mask": torch.BoolTensor,       # [B, 4]
    "teacher_bus_stride": torch.LongTensor,     # [B, 4]
    "teacher_bus_version": torch.LongTensor,    # [B, 4]
    "teacher_bus_coverage": torch.FloatTensor,  # [B, 4, 3]
    
    # Dynamic notes bus (student)
    "student_notes_bus": torch.FloatTensor,     # [B, 4, 3, 2048]
    "student_bus_mask": torch.BoolTensor,       # [B, 4]
    # ... (similar stride/version/coverage fields)
    
    # Plan item hashing
    "plan_item_ids": torch.LongTensor,      # [B, max_items] - hashed plan items
    "plan_item_mask": torch.BoolTensor,     # [B, max_items]
    "coverage_targets": torch.FloatTensor,  # [B, max_items] - coverage labels
    "coverage_mask": torch.BoolTensor,      # [B, max_items]
    
    # Metadata
    "metadata": List[Dict],                 # Original metadata for each example
    "sectional_independence": torch.BoolTensor,  # [B]
}
```

### Forward Pass Architecture

```
Input IDs → Trunk Encoder (GPT-OSS-20B)
                ↓
         Hidden States [B, L, 4096]
                ↓
         ┌──────┴──────┐
         ↓              ↓
   TEACHER BRANCH   STUDENT BRANCH
         ↓              ↓
   Stream Adapters Stream Adapters
   (teacher notes) (student notes)
         ↓              ↓
   Cross-Attention Cross-Attention
   (teacher notes) (student notes)
         ↓              ↓
   Attended States Attended States
         ↓              ↓
      Heads           Heads
   • Planner      • Planner
   • Notes        • Notes
   • Speculation  • Speculation
   • Agreement    • Agreement
   • Coverage     • Coverage
         ↓              ↓
   Teacher Logits  Student Logits
         └──────┬──────┘
                ↓
         Loss Computation
         • KD Loss (KL divergence)
         • Stability Loss
         • Coverage Loss
         • Speculation KL
                ↓
           Backprop
```

## Loss Functions

### 1. Knowledge Distillation Loss (`kd`)

KL divergence between student and teacher planner distributions:

```python
kd_loss = F.kl_div(
    F.log_softmax(student_logits / T_lm, dim=-1),
    F.softmax(teacher_logits / T_lm, dim=-1),
    reduction='batchmean'
) * (T_lm ** 2)
```

Encourages student to match teacher's output distribution despite noisy notes.

### 2. Stability Loss (`stab`)

Measures pre/post update consistency (Speculative Invariance):

```python
stab_loss = F.kl_div(
    F.log_softmax(post_update_logits, dim=-1),
    F.softmax(pre_update_logits, dim=-1),
    reduction='batchmean'
)
```

Tests whether updating notes bus (delta=1 → delta=0) changes predictions. Low stability = notes are being used correctly.

### 3. Coverage Loss (`cov`)

Binary cross-entropy for plan item coverage prediction:

```python
cov_loss = F.binary_cross_entropy_with_logits(
    coverage_logits,    # [B, max_items]
    coverage_targets,   # [B, max_items]
    weight=coverage_mask
)
```

Trains model to predict which plan items were covered in the generated text.

### 4. Speculative KL Loss (`spec_kl`)

KL divergence between speculative notes and teacher notes:

```python
spec_kl_loss = F.kl_div(
    F.log_softmax(student_spec_notes / T_spec, dim=-1),
    F.softmax(teacher_notes / T_spec, dim=-1),
    reduction='batchmean'
)
```

### 5. Usage Loss (`use`)

Penalizes ignoring notes via mask ablation:

```python
# Compute outputs with and without notes
outputs_with_notes = model(hidden, notes=student_notes)
outputs_no_notes = model(hidden, notes=zeros_like(student_notes))

# Penalize if distributions are identical (notes not used)
use_loss = -F.kl_div(
    F.log_softmax(outputs_no_notes, dim=-1),
    F.softmax(outputs_with_notes, dim=-1),
    reduction='batchmean'
)
```

### 6. NLI Loss (`nli`)

Semantic entailment verification (optional, requires `nli_scorer`):

```python
nli_loss = nli_scorer.score(
    premises=generated_text,
    hypotheses=notes_text,
    margin=nli_margin
)
```

## Monitoring & Debugging

### Key Metrics

During training, monitor these logged metrics (available in stdout or WandB):

```
train | step=10 | loss=2.45 | stage=0
  | planner_loss=2.35         # Planner head cross-entropy
  | kd_ce_ratio=0.92          # KD loss / CE loss ratio (higher = better alignment)
  | mask_ablation=0.15        # Notes usage (higher = more dependent on notes)
  | stability_kl=0.08         # Pre/post update divergence
  | coverage_precision=0.65   # Plan coverage prediction accuracy
  | agreement_precision=0.70  # Agreement head accuracy
```

**Healthy training signals**:
- `kd_ce_ratio` > 0.8: Student matching teacher well
- `mask_ablation` > 0.1: Model using notes (not ignoring them)
- `stability_kl` < 0.2: Consistent behavior under notes updates
- `coverage_precision` > 0.6: Good plan tracking

**WandB users**: These metrics are automatically logged as:
- `train/kd_ce_ratio`
- `train/mask_ablation`
- `train/stability_kl`
- etc.

See `CHECKLIST.md` for comprehensive monitoring guidelines and abort criteria.

### Evaluation

The config includes evaluation on the validation split:

```yaml
training:
  dataset_path: "data/processed/pdt_10k/kd_train.jsonl"
  eval_dataset_path: "data/processed/pdt_10k/kd_validation.jsonl"
  eval_interval: 200
```

Evaluation runs every 200 steps and logs:
```
eval | step=200 | eval_loss=2.15 | best=2.15
```

### Telemetry Files

After training completes, check `experiments/gpt_oss/`:

#### `train_manifest.json`
```json
{
  "config_path": "configs/gpt_oss_transfer.yaml",
  "dataset": "data/processed/pdt_10k/kd_train.jsonl",
  "global_step": 1000,
  "best_eval_loss": 2.08,
  "agreement_threshold": 0.15,
  "git_sha": "a1b2c3d",
  "notes_schema_version": "1.0"
}
```

#### `train_run_stages.json`
```json
[
  {
    "stage_index": 0,
    "stage_name": "planner_pretrain",
    "start_step": 0,
    "end_step": 500,
    "steps": 500,
    "duration": 245.3,
    "actions": {
      "freeze": ["trunk", "role_adapters"],
      "unfreeze": ["planner_head", "notes_head"]
    }
  },
  // ... stages 1-4
]
```

#### `training_report.json`
```json
{
  "generated_at": "2025-11-28T10:30:00Z",
  "global_step": 1000,
  "stage": 4,
  "train_metrics": {
    "loss": {"last": 1.95, "mean": 2.18, "min": 1.92, "max": 2.85},
    "mask_ablation": {"last": 0.22, "mean": 0.18},
    "stability_kl": {"last": 0.12, "mean": 0.15},
    "coverage_f1": {"last": 0.68, "mean": 0.62}
  }
}
```

#### `adapters.pt`

Lightweight checkpoint containing **only trainable parameters** (not full GPT-OSS-20B):
- Planner head
- Notes head
- Speculation head
- Agreement head
- Coverage head
- Cross-attention layers
- Stream adapters
- Notes bus

Load for inference:
```python
model = ParallelDecoderTransformer(config)
model.trunk_adapter.load_model()  # Load GPT-OSS-20B trunk
adapter_state = torch.load("experiments/gpt_oss/adapters.pt")
model.load_adapter_state_dict(adapter_state)
```

## Common Issues

### Out of Memory

**Symptoms**: CUDA OOM errors

**Solutions**:
1. Reduce `batch_size` (try `batch_size: 1`)
2. Reduce `max_length` in collator (try `4096` instead of `8192`)
3. Enable gradient checkpointing in trunk config
4. Use `torch_dtype: "bfloat16"` (not float32)

### Notes Not Being Used (`mask_ablation` near 0)

**Symptoms**: `mask_ablation < 0.05`, student ignoring notes

**Solutions**:
1. Increase `loss_weights.use` (try `0.1`)
2. Verify `cross_attention.gating_init` is negative (start with gate closed)
3. Check `bus_mix_prob` is low in early stages
4. Ensure student notes have variance (not all zeros)

### Poor KD Alignment (`kd_ce_ratio` < 0.5)

**Symptoms**: Student not learning from teacher

**Solutions**:
1. Increase `loss_weights.kd` (try `2.0`)
2. Verify teacher branch is enabled (`teacher.enabled: true`)
3. Check teacher notes are non-zero
4. Lower learning rate temporarily
5. Extend Stage 0 (planner pretrain) duration

### Unstable Training (Loss Spikes)

**Symptoms**: Sudden loss increases, NaN losses

**Solutions**:
1. Add gradient clipping: `max_grad_norm: 1.0`
2. Reduce learning rate (try `5e-5`)
3. Increase `warmup_steps` (try `100`)
4. Check for malformed examples in dataset (filter out)

## Training on 8x B200

For distributed training on **8x B200 (180GB)** GPUs, use the WandB-enabled training script for real-time remote monitoring and persistent metrics.

### Setup WandB (One-Time)

```bash
# On Lambda instance
uv add wandb
uv run wandb login  # Enter your API key from wandb.ai/authorize
```

### WandB-Enabled Training (Recommended)

For production runs on remote servers, use `train_wandb.py`:

```bash
# 1. SSH to B200 instance
ssh ubuntu@<instance-ip>

# 2. Clone repo
git clone https://github.com/<your-org>/parallel-decoder-transformer.git
cd parallel-decoder-transformer

# 3. One-time setup (installs dependencies, downloads GPT-OSS-20B)
bash scripts/setup_lambda_gpu.sh

# 4. Transfer dataset
rsync -avz data/processed/pdt_10k_gpt41/ ubuntu@<instance-ip>:parallel-decoder-transformer/data/processed/pdt_10k_gpt41/

# 5. Set environment variables
export WANDB_API_KEY=$(cat wandb.txt)
export PDT_DEVICE=cuda

# 6. Launch training (background with nohup)
nohup uv run torchrun --nproc_per_node=8 scripts/train_wandb.py \
  --config configs/gpt_oss_transfer_production.yaml > nohup.out 2>&1 &

# 7. Capture PID for monitoring
echo "Process launched. PID: $!"
```

**Monitoring training:**
```bash
# Live logs (main DDP output)
tail -f nohup.out

# Check process status
ps -ef | grep train_wandb

# GPU utilization
nvidia-smi -l 1
```

**Stopping training:**
```bash
# Standard kill often orphans DDP workers - use nuclear option:
pkill -9 -f torchrun
pkill -9 -f train_wandb
```

**Resuming training:**
```bash
# Script automatically finds latest checkpoint in experiments/gpt_oss/
# Relaunch with the same command:
nohup uv run torchrun --nproc_per_node=8 scripts/train_wandb.py \
  --config configs/gpt_oss_transfer_production.yaml > nohup.out 2>&1 &
```

### Why WandB for Remote Training

**Problem**: SSH disconnection loses stdout logs and real-time metrics.

**Solution**: `train_wandb.py` provides:
- **Remote monitoring** via WandB web UI
- **Persistent metrics** - all training curves saved to cloud
- **Automatic checkpointing** - adapters uploaded as artifacts
- **Stage tracking** - visualize curriculum transitions
- **Alert integration** - notifications on errors

### WandB Metrics Dashboard

Monitor these critical signals at `wandb.ai/<your-username>/parallel-decoder-transformer`:

**Core Learning Signals**:
- `train/kd_ce_ratio` - Student/teacher alignment (target: 0.5-2.0)
- `train/planner_loss` - Should decrease steadily
- `train/agreement_precision` - Agreement head accuracy (target: >0.7 by Stage 4)

**Speculative Invariance** (Stage 3):
- `train/rollback_kl` - Stability under note perturbations (target: <0.5)
- `train/repair_error_rate` - Token-level changes after rollback (target: <10%)
- `train/stability_kl` - Pre/post update consistency
- `train/coverage_precision` - Coverage prediction accuracy (achieved: 71.6% at step 50k)

**Usage Detection** (Critical):
- `train/mask_ablation` - Notes dependency (target: >0.15)
- `train/usage_loss` - Should be non-zero if notes are used

See `CHECKLIST.md` for detailed monitoring guidelines and abort criteria.

### Local Development Training

For local development without remote monitoring:

```bash
# Single-GPU (development config)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py \
  --config configs/gpt_oss_transfer.yaml

# Multi-GPU (device_map="auto" handles distribution)
uv run python scripts/train.py --config configs/gpt_oss_transfer.yaml
```

**Note**: Use `train_wandb.py` with production config for remote training. Local `train.py` uses shorter curriculum (1k steps) for development.

### Deployment Setup

Production deployment is a **two-step process**:

#### Step 1: Instance Setup (One-Time)

Run `setup_lambda_gpu.sh` once on a fresh instance:

```bash
# On fresh GPU instance
cd parallel-decoder-transformer
bash scripts/setup_lambda_gpu.sh
```

This script:
- Installs system dependencies
- Installs `uv` package manager
- Configures PyTorch with CUDA support
- Creates Python 3.12 virtual environment
- Installs all project dependencies
- Downloads GPT-OSS-20B weights (~40GB)
- Verifies GPU availability

**Duration**: ~15-30 minutes (mostly model download)

#### Step 2: Training Launch (Each Run)

**Prerequisites**:
- Completed Step 1 (setup_lambda_gpu.sh)
- Completed dataset pipeline (see DATASET_PIPELINE.md)
- Dataset files in `data/processed/pdt_10k_gpt41/`
- WandB API key in `wandb.txt`

**Launch command**:
```bash
# Set environment
export WANDB_API_KEY=$(cat wandb.txt)
export PDT_DEVICE=cuda

# Launch with torchrun (8 GPUs)
nohup uv run torchrun --nproc_per_node=8 scripts/train_wandb.py \
  --config configs/gpt_oss_transfer_production.yaml > nohup.out 2>&1 &
```

## Advanced: Curriculum Tuning

### Extending Stages

For larger datasets, extend stage durations:

```yaml
training:
  curriculum:
    stage_schedule:
      - 0
      - 1000   # Stage 0: 1000 steps
      - 3000   # Stage 1: 2000 steps
      - 6000   # Stage 2: 3000 steps
      - 10000  # Stage 3: 4000 steps
      - 15000  # Stage 4: 5000 steps
  max_steps: 20000
```

### Custom Stage Policies

Override stage behavior:

```yaml
training:
  stage_policies:
    2:
      name: "notes_bus_warmup"
      bus_mix_prob: 0.1        # Start with 10% teacher mix
      freeze:
        - "trunk"
        - "agreement_head"
      unfreeze:
        - "notes_bus"
        - "speculation_head"
      notes_noise:
        drop_p: 0.02
        paraphrase_p: 0.05
```

### Adaptive Agreement Threshold

Training auto-calibrates `agreement_threshold` using ROC curve:

```python
# In trainer, threshold is recalibrated to maximize F1
threshold = trainer._maybe_recalibrate_agreement_threshold()
# Saved to: experiments/gpt_oss/agreement_thresholds.json
```

Manually override:
```yaml
training:
  agreement_threshold: 0.20  # Higher = more conservative rollback
```

## Architecture Notes

### Two-Branch Design

Training maintains two parallel forward passes:
1. **Teacher branch**: Uses ground-truth notes, provides supervision
2. **Student branch**: Uses noisy notes, learns to match teacher output

This design enables **Speculative Invariance**: student learns to produce correct outputs regardless of note quality.

### Dynamic Notes Bus

The notes bus maintains multiple snapshots with different lag values:
- `lag=0`: Most recent notes (not yet committed)
- `lag=1`: Previous snapshot (default for inference)
- `lag=2+`: Historical snapshots

During training, the collator builds a 4D tensor:
```python
notes_bus[batch, snapshot, stream, notes_dim]
# [B, 4, 3, 2048]
```

The `_extract_notes_from_bus()` method selects snapshots based on current lag and masks.

### Sectional Independence

For multi-stream documents where streams are independent:
```python
sectional_independence = True
```

This enables:
- Stream-specific label masking (only supervise on target stream's tokens)
- Cross-stream attention blocking
- Independent rollback decisions per stream

Configured in dataset pipeline via 3-stream plan generation.

## Production Checklist

Before deploying a trained model:

- [ ] Training completed without errors
- [ ] Final `eval_loss` is reasonable (< 2.5 for well-trained model)
- [ ] `mask_ablation` > 0.15 (model uses notes)
- [ ] `coverage_f1` > 0.6 (good plan tracking)
- [ ] `adapters.pt` checkpoint saved successfully
- [ ] `train_manifest.json` contains correct git SHA
- [ ] Tested inference with saved adapters
- [ ] Validated outputs match expected format
- [ ] Agreement threshold calibrated appropriately

## Summary

**Training inputs**: Split-specific JSONL files from dataset pipeline:
- Training: `data/processed/pdt_10k_gpt41/kd_train.jsonl`
- Validation: `data/processed/pdt_10k_gpt41/kd_validation.jsonl`
- Test: `data/processed/pdt_10k_gpt41/kd_test.jsonl` (for final evaluation)

**Training commands**:
```bash
# Local development (1k steps, single GPU)
uv run python scripts/train.py --config configs/gpt_oss_transfer.yaml

# Production (50k steps, 8 GPUs with DDP)
export WANDB_API_KEY=$(cat wandb.txt)
nohup uv run torchrun --nproc_per_node=8 scripts/train_wandb.py \
  --config configs/gpt_oss_transfer_production.yaml > nohup.out 2>&1 &
```

**Key outputs**:
- `experiments/gpt_oss/adapters_step_50000.pt` - Final checkpoint
- `experiments/gpt_oss/train_manifest.json` - Training metadata
- `experiments/gpt_oss/training_report.json` - Aggregated metrics
- WandB dashboard: `wandb.ai/<user>/parallel-decoder-transformer`

**Production training**:
- **Duration**: 50,000 steps (~30 hours on 8x B200)
- **Hardware**: 8x NVIDIA B200 (180GB VRAM each)
- **Final results**: 71.6% coverage precision
- **Curriculum**: 4 stages (Stage 4 trunk unfreezing not supported)

**Critical requirements**:
- Teacher notes pre-generated in dataset pipeline
- GPT-OSS-20B trunk weights downloaded  
- 8x B200 GPUs or equivalent (178GB+ VRAM per GPU)
- WandB account for production training

For questions or issues, refer to:
- `DATASET_PIPELINE.md` - Dataset generation
- `TECHNICAL_IMPLEMENTATION.md` - Architecture details
- `scripts/train.py` - Training entry point
- `src/parallel_decoder_transformer/training/trainer.py` - Training loop

