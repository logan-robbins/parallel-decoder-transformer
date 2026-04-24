# Parallel Decoder Transformer (PDT)

PDT augments a frozen Qwen3 decoder trunk with trainable sidecar modules for
K synchronized streams that coordinate through a narrow delayed latent bus.
The current code is the mechanism-first rebuild from `PLAN.md`: hash-era
planner IDs, supervised teacher notes, `NotesHead`, and the separate
`PlanEmbedding` module have been removed.

## Current System State

The runtime path is:

```text
shared context -> frozen trunk prompt encode -> VQ planner
               -> plan_notes_proj -> snapshot-0 notes on Dynamic Notes Bus
               -> SNC reads visible notes during K stream continuations
               -> SpeculationHead writes block-end notes for later blocks
```

The training path now performs teacher-forced differentiable block rollout.
Receiver LM loss can backpropagate through SNC into visible sibling notes and
the speculation writer that produced them. Loss reporting includes
`lm_ce_dependency` and `lm_ce_nondependency` separately.

## Repository Map

```text
src/pdt/
  config/       Dataclass config schema and YAML loader
  trunk/        Qwen3 adapter and instrumented decoder layer wrapper
  sidecar/      SNC, stream adapters, VQ planner, plan note projection, heads
  runtime/      Dynamic Notes Bus, notes windows, orchestrator, counterfactuals
  training/     Canonical dependency dataset loader, losses, rollout trainer
  datasets/     Dataset generation and retokenization support
  cli/          train / infer / ablate entry points

scripts/
  generate_dependency_dataset.py
  generate_snapshot_routing_dataset.py
  retokenize_corpus.py
  validate_dependency_dataset.py
  train.py / infer.py compatibility wrappers
```

## Local Environment

Use `uv` only:

```bash
uv venv .venv --python 3.12
uv sync
```

Apple Silicon is supported for code and smoke-test validation. Do not run
scale Qwen3 training on this Mac M4 host; no NVIDIA CUDA GPUs are available.

## Data Workflow

Generate local synthetic dependency data:

```bash
uv run scripts/generate_dependency_dataset.py \
  --output data/datasets/ldc/train.jsonl \
  --num-examples 256
```

Retokenize with a tokenizer that already exists locally. The script uses
`local_files_only=True` and will fail fast rather than downloading weights:

```bash
uv run scripts/retokenize_corpus.py \
  --input data/datasets/ldc/train.jsonl \
  --output data/processed/latent_dependency_control/train.jsonl \
  --tokenizer /path/to/local/Qwen3-4B-Base
```

Run CE admission audits only when a local model path is available:

```bash
uv run scripts/validate_dependency_dataset.py \
  --input data/processed/latent_dependency_control/train.jsonl \
  --model /path/to/local/Qwen3-4B-Base \
  --output-report data/processed/latent_dependency_control/audit.json
```

## Training And Inference

Single-process training entry point:

```bash
uv run scripts/train.py --config configs/pdt_qwen3_4b.yaml
```

DDP on a CUDA host:

```bash
uv run torchrun --nproc_per_node=N -m pdt.cli.train --config configs/pdt_qwen3_4b.yaml
```

Inference:

```bash
uv run scripts/infer.py \
  --config configs/pdt_qwen3_4b.yaml \
  --checkpoint experiments/qwen3_4b/checkpoints/step_0050000.pt \
  --prompt "Coordinate three streams over this snapshot." \
  --max-new-tokens 256
```

## Validation

Current local smoke validation:

```bash
uv run pytest tests/smoke/ -v
```

Latest result on this workspace: 19 passed.

The no-hash contract check is:

```bash
rg -n "planner_ids|notes_teacher|notes_student|NotesHead|TeacherCache|plan_hash_salt|notes_head|weights\\.notes|weights\\.spec" src/pdt tests configs scripts
```

Expected remaining matches are explicit negative tests or removed-field
rejection lists, not runtime/training dependencies.
