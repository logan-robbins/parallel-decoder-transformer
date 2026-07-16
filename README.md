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
`lm_ce_dependency` and `lm_ce_nondependency` separately. The functional teacher
is the same revision-pinned frozen trunk with all PDT contexts cleared; it sees
one complete privileged chat prefix per block and supplies dependency-only
forward KL at exactly `T=2`, while hard CE covers every active target token.

The canonical trunk is
[`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
at revision `cdbee75f17c01a7cc42f958dc650907174af0554`. The YAML, schema,
adapter, tokenizer preflight, prompt builders, and checkpoint identity all pin
that exact pair. PDT does not add special tokens or mutate the frozen embedding
matrix.

The implemented synchronization contract is exact:

- `K=3`, `tau=32`, and `Delta=1`.
- Every processed target block has exactly 32 active tokens.
- The addressed SNC window is fixed at `2K`: K prompt anchors followed by the
  latest eligible dynamic write from each producer under last-write-wins
  replacement. For the canonical model its shape is `(B, 6, 256)`.
- Runtime prefills each private stream once, consumes each generated token once,
  freezes all K windows at the start of a synchronous round, and publishes all
  tau-th-token writes only after every stream completes that round.
- Structured runtime calls must provide `stream_block_transition_ids` for every
  stream: exactly one row per generated block, with an empty block-0 row and a
  nonempty row thereafter. After synchronous block-m publication, runtime
  consumes every stream's next transition under the frozen block-(m+1) Delta=1
  window; transition tokens advance KV state but emit no output or bus write.
- Training uses batch size 1 with gradient accumulation: it compacts and
  prefills each stream prompt once. Before each block after block 0, it consumes
  that block's canonical chat/observation transition under the newly frozen SNC
  context; transitions advance the cache but produce no LM loss or bus write.
  It then consumes the complete 32-token target block in one differentiable
  cached forward. The prefix-final logit scores target token 0 and block logits
  `0..30` score target tokens `1..31`; the final target hidden writes the bus
  only after all K block forwards finish.
- The frozen trunk stays in eval mode and gradient checkpointing is disabled.
  Hugging Face may drop `past_key_values` when train mode and gradient
  checkpointing coincide, so training rejects either state and also fails if
  any prefill or block forward returns no cache.
- Planner, plan-note, speculation, and classifier parameters remain FP32;
  instrumented trunk-resident SNC/adapters follow the BF16 trunk. Every mixed
  boundary casts explicitly, and CE/KD reductions accumulate in FP32.
- Runtime exposes no agreement or rollback result fields. Commit control is
  absent until a trained, separately validated controller is implemented.
- Each current note is a dense 256-dimensional BF16 vector: 512 bytes or 4096
  physical bits per producer/write. The default synthetic payload is 18 exact
  bits, so its physical efficiency ceiling is `18/4096 = 0.00439453125`. The
  planner VQ does not quantize dynamic notes.

## Locked Training Recipe

- Frozen student trunk: `Qwen/Qwen3-4B-Instruct-2507` at a recorded revision.
- Functional teacher: the identical frozen trunk with full serialized context
  and all PDT sidecar contexts disabled. Use dependency-masked token KL at
  temperature 2 plus hard CE on all target tokens.
- Natural response teacher: `Qwen/Qwen3-30B-A3B-Instruct-2507`, used offline
  only after the synthetic causal and planner gates pass.
- Mechanism data: randomized exact-entropy cross-stream register relay, not
  generic world knowledge.
- Natural transfer data: validated 40k HotpotQA plus up to 10k English OASST1
  root prompts, transformed into exactly three serializable stream targets.
- Scale host: one H100 SXM 80GB with at least 200GB persistent storage. The
  current configuration contains exactly 401,325,095 trainable parameters
  (approximately 401.3M).

## Repository Map

```text
src/pdt/
  config/       Dataclass config schema and YAML loader
  trunk/        Qwen3 adapter and instrumented decoder layer wrapper
  sidecar/      SNC, stream adapters, VQ planner, plan note projection, heads
  runtime/      Dynamic Notes Bus, notes windows, orchestrator, counterfactuals
  training/     Canonical dependency dataset loader, losses, rollout trainer
  datasets/     Dataset generation and retokenization support
  evaluation/   Strict paired causal-ablation aggregation
  checkpoint.py Versioned atomic save/load/resume contract
  cli/          train / infer / ablate entry points

scripts/
  generate_dependency_dataset.py
  generate_snapshot_routing_dataset.py
  retokenize_corpus.py
  validate_dependency_dataset.py
  smoke_qwen3_pdt.py
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
The cached real-checkpoint forward contract can be rerun without network
access:

```bash
mkdir -p experiments/qwen3_4b/logs
nohup env HF_HUB_OFFLINE=1 uv run scripts/smoke_qwen3_pdt.py --device mps \
  > experiments/qwen3_4b/logs/real_smoke.log 2>&1 &

while pgrep -f "scripts/smoke_qwen3_pdt.py --device mps" >/dev/null; do
  tail -n 80 experiments/qwen3_4b/logs/real_smoke.log
  sleep 15
done
```

## Data Workflow

Retokenization uses prompt schema `qwen3-instruct-temporal-chat-v2` with
`add_generation_prompt=True` and `enable_thinking=False`. Raw streams store one
private observation per block. The initial addressed stream prompt contains
only block 0's observation; `block_transition_ids[m]` reveals observation `m`
between completed target `m-1` and target `m` (`m=0` is the required empty
row). Teacher prompt `m` contains observations only through `m` and completed
targets only through `m-1`. Legacy split prompt fragments and full private
register logs are rejected.

Generate the 32-example overfit set:

```bash
uv run scripts/generate_dependency_dataset.py \
  --output data/datasets/ldc/train_32.jsonl \
  --num-examples 32 --slots 3 --blocks 8 --streams 3 --seed 101

uv run scripts/generate_dependency_dataset.py \
  --output data/datasets/ldc/validation_32.jsonl \
  --num-examples 32 --slots 3 --blocks 8 --streams 3 \
  --split validation --seed 501
```

Generate the 1k scale gate and its rho-zero null twin:

```bash
uv run scripts/generate_dependency_dataset.py \
  --output data/datasets/ldc/gate_1000.jsonl \
  --num-examples 1000 --slots 3 --blocks 8 --streams 3 --seed 201

uv run scripts/generate_dependency_dataset.py \
  --output data/datasets/ldc/gate_null_1000.jsonl \
  --num-examples 1000 --slots 3 --blocks 8 --streams 3 --rho 0 --seed 301
```

Generate the scale corpus only after both gates pass:

```bash
uv run scripts/generate_dependency_dataset.py \
  --output data/datasets/ldc/train_20000.jsonl \
  --num-examples 20000 --slots 3 --blocks 8 --streams 3 --seed 401

uv run scripts/generate_dependency_dataset.py \
  --output data/datasets/ldc/validation_2000.jsonl \
  --num-examples 2000 --slots 3 --blocks 8 --streams 3 --split validation --seed 501

uv run scripts/generate_dependency_dataset.py \
  --output data/datasets/ldc/null_2000.jsonl \
  --num-examples 2000 --slots 3 --blocks 8 --streams 3 --rho 0 --split validation --seed 601
```

Retokenize with a tokenizer that already exists locally. The script uses
`local_files_only=True` and will fail fast rather than downloading weights:

```bash
mkdir -p experiments/qwen3_4b/logs
nohup uv run scripts/retokenize_corpus.py \
  --input data/datasets/ldc/train_20000.jsonl \
  --output data/processed/latent_dependency_control/train.jsonl \
  --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
  > experiments/qwen3_4b/logs/retokenize.log 2>&1 &

while pgrep -f "scripts/retokenize_corpus.py" >/dev/null; do
  tail -n 80 experiments/qwen3_4b/logs/retokenize.log
  sleep 15
done
```

Run CE admission audits only when a local model path is available:

```bash
nohup uv run scripts/validate_dependency_dataset.py \
  --input data/processed/latent_dependency_control/train.jsonl \
  --model /path/to/local/Qwen3-4B-Instruct-2507 \
  --output-report data/processed/latent_dependency_control/audit.json \
  > experiments/qwen3_4b/logs/audit.log 2>&1 &

while pgrep -f "scripts/validate_dependency_dataset.py" >/dev/null; do
  tail -n 80 experiments/qwen3_4b/logs/audit.log
  sleep 15
done
```

## Training And Inference

CUDA preflight:

```bash
uv sync
uv run scripts/check_gpu.py
uv run scripts/check_qwen3_config.py
uv run pytest tests/smoke/ -v
```

The GPU check must report `cuda.is_available: True` and an 80GB H100-class
device. Start the long-running single-GPU job under `nohup` and poll its log at
15-second intervals:

```bash
mkdir -p experiments/qwen3_4b/logs
nohup uv run scripts/train.py --config configs/pdt_qwen3_4b.yaml \
  > experiments/qwen3_4b/logs/train.log 2>&1 &

while pgrep -f "scripts/train.py --config configs/pdt_qwen3_4b.yaml" >/dev/null; do
  tail -n 80 experiments/qwen3_4b/logs/train.log
  sleep 15
done
```

Resume restores phi, optimizer, scheduler, global step, and the matching
curriculum freeze policy from the canonical checkpoint format:

```bash
nohup uv run scripts/train.py \
  --config configs/pdt_qwen3_4b.yaml \
  --resume experiments/qwen3_4b/checkpoints/step_0002500.pt \
  > experiments/qwen3_4b/logs/resume.log 2>&1 &

while pgrep -f "scripts/train.py.*--resume" >/dev/null; do
  tail -n 80 experiments/qwen3_4b/logs/resume.log
  sleep 15
done
```

Legacy ad-hoc checkpoints are intentionally incompatible. Inference and
ablation loading verify the exact frozen trunk, revision, ordered instrumented
layers, parameter keys/shapes/dtypes, and checkpoint version.

Do not launch the configured 50,000 optimizer steps as the first rental job.
Point the config at the 32-example set, pass gradient/mutation/gate-zero checks,
then run the 1k gate and measure peak VRAM plus examples/second. Multi-GPU DDP
is not part of the first causal gate because it replicates the model and does
not solve per-rank graph or memory defects.

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

Latest result on this workspace (2026-07-16): 208 tests passed locally.
The tests cover prompt/data timing, fixed-window lag and LWW semantics, runtime
cache scheduling, functional distillation, strict checkpoints, and token-weighted
paired causal metrics.

This is contract evidence, not a trained-model result. Still unproven are a
real Qwen3-4B optimizer step, nonzero end-to-end phi gradients on CUDA, a
trained checkpoint, the 32-example overfit/causal acceptance gate, source-swap
behavior on 1,000 examples, throughput, and peak VRAM. The parameter-matched
self-only and communication upper-bound baseline runners also remain to be
implemented.

The no-hash contract check is:

```bash
rg -n "planner_ids|notes_teacher|notes_student|NotesHead|TeacherCache|plan_hash_salt|notes_head|weights\\.notes|weights\\.spec" src/pdt tests configs scripts
```

Expected remaining matches are explicit negative tests or removed-field
rejection lists, not runtime/training dependencies.
