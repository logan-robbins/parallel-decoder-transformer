# Parallel Decoder Transformer (PDT)

PDT is a frozen dense-Qwen3 trunk extended with trainable planner heads,
versioned latent notes, cross-note attention, and per-stream adapters. Three
persistent streams write one coordinated long-form document while exchanging a
finite, delayed message rather than full text or full KV state.

The current system is ready for a new H100 mechanism run. It is not yet a
positive research result: the earlier 512-update experiment used short register
sentences and is retained only as historical proof that gradients and mutation
paths worked. The next result must come from the long-form contract and the
document-paired causal evaluator described below.

## Current Architecture

The canonical path is:

```text
shared document brief -> frozen trunk -> VQ planner -> per-stream anchor notes
private block update  -> cached stream continuation -> 32-token prose block
final block hidden    -> product-VQ writer -> delayed versioned notes history
visible notes history -> SNC in 12 trunk layers -> later stream continuation
```

There is one implementation for both scale rungs:

| Profile | Frozen trunk | Hidden/layers | Instrumented layers | Trainable phi |
|---|---|---:|---|---:|
| `qwen3_4b_instruct_2507` | `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554` | 2560 / 36 | 2,5,8,11,14,17,20,23,26,29,32,35 | 156,484,647 |
| `qwen3_14b` | `Qwen/Qwen3-14B@40c069824f4251a91eefaf281ebe4c544efd3e18` | 5120 / 40 | 2,6,9,12,16,19,22,26,29,32,36,39 | 305,374,247 |

The trunk stays frozen, in eval mode, and uses its real differentiable KV
cache. Gradient checkpointing is rejected because Hugging Face can disable the
cache in that state. Batch size is one and larger effective batches use
gradient accumulation.

The trainable structural extensions are not full-width copies of the trunk:

- SNC projects trunk queries into a fixed 512-wide, eight-head communication
  space, reads 256-dimensional addressed notes, then projects back to trunk
  width.
- The planner and stream classifier use fixed 512-wide bottlenecks.
- Each instrumented layer has an independent SNC module, three independent
  bottleneck adapters, an SNC outer gate, and an adapter outer gate.
- Evaluation records actual parameter norms and inner/outer gate probabilities
  for every instrumented layer, so a run cannot hide behind aggregate loss if
  the structural path never opens.

The synchronization contract is fixed at three streams, 32 target tokens per
block, one-block publication delay, and sixteen blocks of version history. An
SNC read sees three prompt anchors plus sixteen exact write versions for each
producer, giving a `(B, 51, 256)` addressed window. Dynamic slots are ordered by
age and carry producer, anchor/dynamic kind, and lag metadata. The old `2K`
last-write-wins window has been removed because it could not represent a
dependency at lag 4, 8, or 16.

## What “Codebook” Means

There are two distinct learned finite dictionaries:

1. The planner codebook has 8,192 learned 512-dimensional entries. At prompt
   time, each of sixteen planner slots selects one entry; those selections are
   projected into the three initial anchor notes. It represents document-level
   role and plan state.
2. The dynamic writer has four independent 256-entry sub-codebooks. Each block
   write transmits four integer indices, exactly 32 bits, and the receiver
   reconstructs a 256-dimensional note from the shared dictionaries. It
   represents evolving cross-section state.

Codebook utilization is diagnostic, not proof of coordination. Telemetry
reports total observations, the maximum number of unique entries that could
have been observed, effective entries per slot, and exact collapse. There is no
absolute “1,000 entries” gate and no utilization statistic is allowed to
replace causal intervention evidence.

## Long-Form Data Contract

`long-form-private-stream-v1` is continuous expository prose, not QA and not a
set of short answer sentences. Every example contains:

- three section streams: historical evidence, risk analysis, and practical
  recommendations;
- 32 synchronized blocks per stream and exactly 32 pinned-tokenizer target
  tokens per block, giving 1,024 target tokens per stream;
- one private stream-local packet before every block;
- sixteen cross-section constraints per stream, with four first uses at each
  lag 1, 4, 8, and 16;
- sixteen local prose control blocks per stream;
- an immutable source stream/block, payload text, three independent codewords,
  and exact 18-bit entropy annotation for every dependency;
- a surface-matched `rho=0` twin that resolves the same references from the
  receiver's own private history.

The privileged functional teacher is the identical frozen trunk with sidecar
contexts disabled. For each target block it receives the receiver's current
private observation plus only the older sibling observations named by that
block's dependency annotations. It supplies forward token KL at temperature 2;
hard CE remains active on every target token.

Raw JSONL is immutable and the generator refuses to overwrite it. Processed
JSONL is derived and may be regenerated. Processed records store the exact
tokenizer model and revision, so 4B and 14B outputs have separate directories.

Generate the current 32-document train, held-out, and null sets from the repo
root:

```bash
uv run scripts/generate_dependency_dataset.py \
  --output data/datasets/long_form_dependency/train_32.jsonl \
  --num-examples 32 --split train --seed 1729

uv run scripts/generate_dependency_dataset.py \
  --output data/datasets/long_form_dependency/validation_32.jsonl \
  --num-examples 32 --split validation --seed 2718

uv run scripts/generate_dependency_dataset.py \
  --output data/datasets/long_form_dependency/null_validation_32.jsonl \
  --num-examples 32 --split null_validation --rho 0 --seed 2718
```

Retokenize for the selected pinned profile after its tokenizer has been cached:

```bash
PROFILE=qwen3_4b_instruct_2507
mkdir -p "data/processed/long_form_dependency/$PROFILE" logs

nohup uv run scripts/retokenize_corpus.py \
  --input data/datasets/long_form_dependency/train_32.jsonl \
  --output "data/processed/long_form_dependency/$PROFILE/train.jsonl" \
  --trunk-profile "$PROFILE" > "logs/retokenize_${PROFILE}_train.log" 2>&1 &
```

Poll every long-running command at 15-second intervals. Repeat for
`validation_32.jsonl -> validation.jsonl` and
`null_validation_32.jsonl -> null_validation.jsonl`. Use `--force` only for a
derived processed file; never delete or alter the raw JSONL.

The local workspace currently has all three 4B processed sets, each with 32
documents. Generated data is intentionally gitignored, so a fresh GPU host
must reproduce it with the commands above.

## Causal Evidence Contract

Evaluation runs four aligned teacher-forced rollouts on each complete document:

- baseline;
- SNC gate zero;
- sibling dynamic-note norm scramble;
- a guaranteed one-subcode mutation of one configured producer/block write.

Mutation is measured only at annotated future tokens whose dependency names
that exact source write. Dependency and nondependency effects are retained per
document. The primary selectivity statistic is a difference-in-differences:

```text
(CE_gate_zero - CE_baseline)_dependency
  - (CE_gate_zero - CE_baseline)_nondependency
```

No effect ratio is computed. Telemetry contains deterministic document
bootstrap intervals for the dependency effect, nondependency effect,
selectivity difference, targeted mutation KL, and dependency effect at lags
1/4/8/16. The bus evidence gate requires at least 32 documents and positive
lower 95% bounds for dependency effect, selectivity, and targeted mutation KL.

The self-only control is separately initialized and parameter-matched. Its SNC
replacement sees the same number of receiver-owned prompt/block states and the
same 16-block horizon, but no sibling tensor. `scripts/compare_self_only.py`
aligns document IDs and bootstraps the paired bus-minus-self-only dependency
effect. It passes only when that advantage's lower bound is positive; the old
“less than 50% recovery” threshold has been removed.

## Local Verification

Use Python 3.12 and `uv` exclusively:

```bash
uv sync --frozen
uv run ruff check src tests scripts
uv run pytest tests/smoke/ -v
```

Current local verification on 2026-07-16: Ruff passes, mypy reports no issues
across 51 source files, and all 248 smoke tests pass.

Apple Silicon is for code, schema, tokenizer, and small real-trunk checks, not
scale training. A cached 4B packed-frontier smoke can be run with:

```bash
mkdir -p experiments/qwen3_4b_instruct_2507/smoke/logs
nohup env HF_HUB_OFFLINE=1 uv run scripts/smoke_qwen3_pdt.py \
  --trunk-profile qwen3_4b_instruct_2507 --device mps --max-new-tokens 33 \
  > experiments/qwen3_4b_instruct_2507/smoke/logs/run.log 2>&1 &
```

The executable hardware accounting model remains available through
`uv run scripts/decode_roofline.py --streams 3 --contexts 1024 4096 16384`.
It is a roofline bound, not a trained PDT latency result.

## H100 Bootstrap

On a fresh H100 clone, bootstrap one profile at a time:

```bash
mkdir -p experiments/bootstrap
nohup bash scripts/setup_lambda_gpu.sh \
  --trunk-profile qwen3_4b_instruct_2507 \
  > experiments/bootstrap/setup_4b.log 2>&1 &
```

The setup uses the committed lockfile, checks at least 75 GB of visible GPU
memory, validates the pinned tokenizer/config, and runs the complete smoke
suite. It does not use pip, FlashAttention, gradient checkpointing, or an
alternate model path. Poll `experiments/bootstrap/setup_4b.log` every 15
seconds and stop on an error. Then regenerate and retokenize the long-form data
on that host.

The first model write is always the two-update optimizer/gradient probe:

```bash
PROFILE=qwen3_4b_instruct_2507
RUN="experiments/$PROFILE/probe_bus"
mkdir -p "$RUN/logs"
nohup uv run scripts/train.py \
  --config configs/pdt_qwen3_4b.yaml \
  --trunk-profile "$PROFILE" \
  --optimizer-probe --telemetry-dir "$RUN" \
  > "$RUN/logs/train.log" 2>&1 &
```

Do not start training unless `optimizer_probe.json` and
`checkpoints/step_0000002.pt` both exist and the report shows finite nonzero
gradients in every active stage-0 group. The previous full-width 4B probe used
30.44 GB reserved on an H100; that number does not predict the new long-horizon
4B or 14B profile, so both require fresh measurements.

## 4B Mechanism Run

The true overfit evaluation uses the train set for both optimization and the
end-of-run causal evaluation. With two H100s, run bus and self-only concurrently
on separate hosts; do not shard either 4B condition.

Bus:

```bash
PROFILE=qwen3_4b_instruct_2507
DATA="data/processed/long_form_dependency/$PROFILE/train.jsonl"
RUN="experiments/$PROFILE/overfit32_bus"
mkdir -p "$RUN/logs"
nohup uv run scripts/train.py \
  --config configs/pdt_qwen3_4b.yaml --trunk-profile "$PROFILE" \
  --coordination-source bus --telemetry-dir "$RUN" \
  --dataset-path "$DATA" --eval-dataset-path "$DATA" \
  --max-steps 512 --grad-accumulation 1 --warmup-steps 32 \
  --stage-schedule 0 32 128 256 \
  --save-every 128 --eval-interval 512 --log-interval 1 \
  > "$RUN/logs/train.log" 2>&1 &
```

Self-only uses the identical command with
`--coordination-source self_only` and
`RUN=experiments/$PROFILE/overfit32_self_only`.

After both finish:

```bash
uv run scripts/compare_self_only.py \
  --bus experiments/qwen3_4b_instruct_2507/overfit32_bus/eval_0000512.json \
  --self-only experiments/qwen3_4b_instruct_2507/overfit32_self_only/eval_0000512.json \
  --minimum-documents 32
```

Evaluate either checkpoint on held-out or null documents without taking an
optimizer step. The schedule arguments must match the checkpoint:

```bash
PROFILE=qwen3_4b_instruct_2507
SOURCE="experiments/$PROFILE/overfit32_bus"
EVAL="experiments/$PROFILE/overfit32_bus_heldout"
mkdir -p "$EVAL/logs"
nohup uv run scripts/train.py \
  --config configs/pdt_qwen3_4b.yaml --trunk-profile "$PROFILE" \
  --resume "$SOURCE/checkpoints/step_0000512.pt" --eval-only \
  --telemetry-dir "$EVAL" \
  --eval-dataset-path "data/processed/long_form_dependency/$PROFILE/validation.jsonl" \
  --max-steps 512 --warmup-steps 32 --stage-schedule 0 32 128 256 \
  > "$EVAL/logs/eval.log" 2>&1 &
```

## Dense 14B Rung

Run dense 14B only after the 4B bus path overfits and beats the matched control.
Bootstrap with `--trunk-profile qwen3_14b`, retokenize the same immutable raw
documents into `data/processed/long_form_dependency/qwen3_14b/`, and run the
two-update probe with `--trunk-profile qwen3_14b`. A single H100 is the intended
first attempt; the second H100 should run the matched condition concurrently.
Only introduce model sharding if the measured 14B optimizer probe cannot fit.

The 30.5B/3.3B-active Qwen3 MoE is not the default “bigger model.” Its expert
topology and low-precision path introduce a different architectural variable.
Test it only if a trained dense 14B result identifies active trunk capacity as
the limitation.

## Repository Map

```text
configs/                 canonical base YAML; profile overrides stay on one path
src/pdt/config/          pinned profiles and cross-component validation
src/pdt/datasets/        long-form contract and revision-pinned retokenization
src/pdt/sidecar/         planner, projection, product-VQ writer, adapters, SNC
src/pdt/runtime/         packed generation, versioned notes history, interventions
src/pdt/training/        cached differentiable rollout, losses, trainer
src/pdt/evaluation/      document bootstrap and bus/self-only comparison
src/pdt/diagnostics/     architecture, codebook, causal, hardware, information audits
scripts/                 generation, validation, bootstrap, train/infer/ablate tools
PLAN.md                  living research and acceptance plan
```

## Known Unproven Work

- The new long-form 4B bus and self-only runs have not yet been trained.
- Dense 14B memory, optimizer health, and causal effects have not yet been
  measured.
- Planner semantics and dynamic-code utilization remain descriptive until
  causal gates pass.
- Blind, full-text, full-KV, sequential-oracle, single-stream, and full-finetune
  quality baselines remain to be implemented.
- No QA, short-answer, Wikipedia, or other natural-transfer corpus has yet been
  admitted. A future natural corpus must preserve long-form document structure.
