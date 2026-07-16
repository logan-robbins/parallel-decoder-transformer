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
- Runtime packs all private prompts into one prefill, keeps their distinct
  logical histories as rows of one frontier-owned KV cache, and advances the K
  live rows with one trunk call per generated token. It freezes all K windows
  at the start of a synchronous round and publishes all tau-th-token writes
  only after every stream completes that round.
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
- Every dynamic note is now four indices into four 256-entry product
  codebooks: exactly 32 capacity bits per producer/write. The bus accepts only
  that integer tuple and reconstructs the 256-dimensional local SNC tensor
  from the shared codebook; callers cannot attach an unrestricted float
  payload. The default synthetic payload is 18 exact bits, so the configured
  source-to-channel rate ratio is `18/32 = 0.5625`.
- The targeted causal mutation cycles one transmitted sub-code modulo 256 and
  then decodes that altered tuple. It therefore guarantees a different valid
  message instead of relying on a float perturbation that might requantize to
  the original code.

## Fresh-Eye Research Boundary

Three independent gates now define the claim:

1. **Information:** after prompt, plan, and receiver-local history, the missing
   sibling state must have a low-rate sufficient statistic.
2. **Work/span:** the answer's dependency DAG must have width; K streams cannot
   shorten a true causal chain.
3. **Hardware:** the K live frontier tokens must enter one packed model call so
   frozen matrices can be fetched once. Causally independent sequential Python
   calls would not create that wall-clock opportunity.

Paired dependency-span CE remains the causal utility metric. It is not labeled
Shannon mutual information: arbitrary model CE differences include unequal
predictor-approximation errors and need not obey a message-bit ceiling. Exact
bit claims require a finite message alphabet and the known-entropy whole-payload
audit in `pdt.diagnostics.information`. The runtime and differentiable rollout
now use a strict 32-bit product-VQ message. A rate--distortion sweep and the
known-entropy decoding audit remain empirical gates; merely configuring 32
bits does not prove that training uses them effectively.

The hardware lower-bound model is executable without CUDA:

```bash
uv run scripts/decode_roofline.py \
  --streams 3 --contexts 1024 4096 16384 65536
```

It uses the pinned 4,022,468,096-parameter trunk, the current recurrent sidecar
split (173,156,388 shared parameters plus 31,494,144 per stream), Qwen3 GQA
geometry, 989 dense BF16 TFLOP/s, and 3.35 TB/s HBM bandwidth. On that idealized
H100 SXM roofline, the packed PDT call has lower-bound latency advantages over
a three-call sequential baseline of 2.858x, 2.615x, 2.060x, and 1.447x at
1k, 4k, 16k, and 64k tokens per stream respectively. Every point is
memory-bound. The decline is expected: weight reuse is nearly constant while
private KV reads grow as `K*L`; a full-KV cross-stream baseline grows as
`K^2*L`.

The packed-weight premise also passes a local short-context MPS sanity check:

```bash
mkdir -p experiments/qwen3_4b/logs
nohup uv run scripts/batch_latency.py \
  --batches 1 3 --steps 8 --trials 3 --warmup 1 \
  --device mps --dtype float32 \
  > experiments/qwen3_4b/logs/batch_latency_mps.log 2>&1 &

while pgrep -f "scripts/batch_latency.py" >/dev/null; do
  tail -n 80 experiments/qwen3_4b/logs/batch_latency_mps.log
  sleep 15
done
```

The 2026-07-16 primitive run measured 71.59 ms/step at batch 1 and 76.37 ms/step at
batch 3: batch 3 cost 1.07x per frontier step while emitting three tokens, a
2.81x aggregate primitive. This is vanilla-trunk MPS evidence, not a PDT or
CUDA latency result.

The actual PDT runtime is now packed. This command crosses the first `tau=32`
boundary and audits every physical trunk call and finite note tuple:

```bash
nohup uv run scripts/smoke_qwen3_pdt.py \
  --device mps --max-new-tokens 33 \
  > experiments/qwen3_4b/logs/packed_strict_32bit_boundary_mps.log 2>&1 &
```

The verified run used one `(1,18)` planner call, one `(3,18)` stream prefill,
and 33 `(3,1)` continuation calls. It emitted one four-index code per stream,
then consumed the delayed block-0 notes at token 33. Total planner + prefill +
decode time was 5.519 seconds, or 17.94 aggregate generated tokens/second.
This proves physical batch shape, cache continuity, addressing, and finite
transport on MPS. It is not a CUDA speedup measurement, and the untrained
zero-initialized stream paths correctly produced identical streams/codes.

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
  current configuration contains exactly 401,409,063 trainable parameters
  (approximately 401.4M).

## Repository Map

```text
src/pdt/
  config/       Dataclass config schema and YAML loader
  trunk/        Qwen3 adapter and instrumented decoder layer wrapper
  sidecar/      Addressed SNC, stream adapters, planner VQ, dynamic product VQ
  runtime/      Dynamic Notes Bus, notes windows, orchestrator, counterfactuals
  training/     Canonical dependency dataset loader, losses, rollout trainer
  datasets/     Dataset generation and retokenization support
  evaluation/   Paired causal metrics and strict bus/self-only comparison
  checkpoint.py Versioned atomic save/load/resume contract
  cli/          train / infer / ablate entry points

scripts/
  generate_dependency_dataset.py
  generate_snapshot_routing_dataset.py
  retokenize_corpus.py
  validate_dependency_dataset.py
  smoke_qwen3_pdt.py
  compare_self_only.py
  train.py / infer.py compatibility wrappers
```

## Local Environment

Use `uv` only:

```bash
uv sync --frozen
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

Bootstrap a fresh single-H100 host from the repository root. The script uses
the committed lockfile, PyTorch SDPA, the pinned Qwen revision, and the full
smoke suite; it does not install an alternate attention path:

```bash
mkdir -p experiments/bootstrap
nohup bash scripts/setup_lambda_gpu.sh \
  > experiments/bootstrap/setup.log 2>&1 &

while pgrep -f "scripts/setup_lambda_gpu.sh" >/dev/null; do
  tail -n 80 experiments/bootstrap/setup.log
  sleep 15
done
```

The GPU check must report `cuda.is_available: True` and an 80GB H100-class
device. The first write is the fresh two-update bus optimizer probe. Update one
opens the zero-initialized output projections; update two requires finite,
nonzero gradients in SNC q/k/v/o, header/gate, stream-adapter, and speculation
groups. The probe writes peak-memory/time telemetry and a format-v3 checkpoint,
then exits without evaluation or a long run:

```bash
mkdir -p experiments/qwen3_4b/probe_bus/logs
nohup uv run scripts/train.py \
  --config configs/pdt_qwen3_4b.yaml \
  --optimizer-probe \
  --telemetry-dir experiments/qwen3_4b/probe_bus \
  > experiments/qwen3_4b/probe_bus/logs/train.log 2>&1 &

while pgrep -f "scripts/train.py.*--optimizer-probe" >/dev/null; do
  tail -n 80 experiments/qwen3_4b/probe_bus/logs/train.log
  sleep 15
done

test -f experiments/qwen3_4b/probe_bus/optimizer_probe.json
test -f experiments/qwen3_4b/probe_bus/checkpoints/step_0000002.pt
```

Verified on one NVIDIA H100 80GB HBM3 on 2026-07-16: both optimizer updates
completed in 14.12 seconds, with 27.40GB peak allocated and 30.44GB peak
reserved. The second backward produced finite nonzero gradients in SNC
q/k/v/o, addressed headers, inner/outer gates, stream adapters, and the
speculation writer. The stage-0-frozen planner and stream classifier correctly
reported zero active parameters. The probe wrote a 2.0GB format-v3 checkpoint
and `optimizer_probe.json`; the locked 512-update bus run was then admitted.

Only after that passes, run the locked 32-example bus condition from fresh
weights. This is a 512-update, batch-one overfit schedule with all four stages
reached by update 256:

```bash
mkdir -p experiments/qwen3_4b/overfit32_bus/logs
nohup uv run scripts/train.py \
  --config configs/pdt_qwen3_4b.yaml \
  --coordination-source bus \
  --telemetry-dir experiments/qwen3_4b/overfit32_bus \
  --dataset-path data/processed/latent_dependency_control/train.jsonl \
  --eval-dataset-path data/processed/latent_dependency_control/validation.jsonl \
  --max-steps 512 --grad-accumulation 1 --warmup-steps 32 \
  --stage-schedule 0 32 128 256 \
  --save-every 128 --eval-interval 512 --log-interval 1 \
  > experiments/qwen3_4b/overfit32_bus/logs/train.log 2>&1 &

while pgrep -f "scripts/train.py.*overfit32_bus" >/dev/null; do
  tail -n 80 experiments/qwen3_4b/overfit32_bus/logs/train.log
  sleep 15
done
```

Then train the independently initialized, parameter-identical self-only
condition with the same data, optimizer, losses, and schedule. Its fixed `2K`
window contains only the receiver's own prompt tail and latest delay-eligible
target-block tail; sibling tensors are structurally absent:

```bash
mkdir -p experiments/qwen3_4b/overfit32_self_only/logs
nohup uv run scripts/train.py \
  --config configs/pdt_qwen3_4b.yaml \
  --coordination-source self_only \
  --telemetry-dir experiments/qwen3_4b/overfit32_self_only \
  --dataset-path data/processed/latent_dependency_control/train.jsonl \
  --eval-dataset-path data/processed/latent_dependency_control/validation.jsonl \
  --max-steps 512 --grad-accumulation 1 --warmup-steps 32 \
  --stage-schedule 0 32 128 256 \
  --save-every 128 --eval-interval 512 --log-interval 1 \
  > experiments/qwen3_4b/overfit32_self_only/logs/train.log 2>&1 &

while pgrep -f "scripts/train.py.*overfit32_self_only" >/dev/null; do
  tail -n 80 experiments/qwen3_4b/overfit32_self_only/logs/train.log
  sleep 15
done

uv run scripts/compare_self_only.py \
  --bus experiments/qwen3_4b/overfit32_bus/eval_0000512.json \
  --self-only experiments/qwen3_4b/overfit32_self_only/eval_0000512.json
```

The comparison exits nonzero when self-only recovers at least `0.5` of the
bus's paired dependency-span gate-zero gain. Checkpoint format v3 also embeds
`coordination_source`, so bus and self-only states cannot cross-load despite
their intentionally identical parameter shapes. Do not launch the configured
50,000 updates before the probe, 32-example bus causal metrics, and self-only
comparison pass. Multi-GPU DDP is outside the first gate.

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

Latest result on this workspace (2026-07-16): 241 tests passed locally.
The tests cover prompt/data timing, fixed-window lag and LWW semantics, runtime
cache scheduling, functional distillation, strict checkpoints, and token-weighted
paired causal metrics, plus exact finite-rate information accounting,
packed/separate/full-KV roofline arithmetic, training-integrated self-only
ownership/leakage checks, and strict control-telemetry comparison.

The H100 optimizer probe is real execution evidence, not yet a trained-model
result. Still unproven are the 32-example overfit/causal acceptance gate,
source-swap behavior on 1,000 examples, trained throughput, and matched-quality
peak VRAM. The independently trainable parameter-matched self-only runner and
its `<0.5` comparison are implemented; blind, sequential-oracle, full-text,
and full-KV quality runners remain.

The no-hash data-contract check is:

```bash
uv run pytest tests/smoke/pdt_tests/test_no_hashing.py -v
```

Expected remaining matches are explicit negative tests or removed-field
rejection lists, not runtime/training dependencies.
