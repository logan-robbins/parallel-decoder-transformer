# Model-Intrinsic Parallel Generation

Working paper title:

**Model-Intrinsic Parallel Generation: Planner-Conditioned Latent Coordination
Across Synchronous Decoder Frontiers**

This repository tests whether one frozen causal transformer can be extended
with a learned planner and a delayed latent communication bus so that three
physical decoder frontiers generate three complementary long-form sections at
the same time. The current repository contains the architecture, strict
source-grounded data pipeline, training curriculum, packed runtime, and
falsifiable evaluators. It does not yet contain a positive result from the new
real-data experiment.

The older synthetic short-sentence and QA-style datasets are historical
diagnostics only. They are not admissible evidence for this experiment.

## Canonical system

The single production path is:

```text
source + expository prompt
        |
frozen Qwen prompt encoder
        |
continuous unordered planner [B, 3, 8, 512]
        |
three persistent read-only plan memories
        |
one frozen shared trunk with three physical KV/frontier rows
        |
one packed [3, 32] teacher-forced call per synchronized block
        |
four-index product-VQ write from each completed lane block
        |
one-block-delayed addressed dynamic-note reads at 12 trunk layers
        |
three multi-paragraph sections in learned presentation order
```

D1, D2, and D3 are physical cache and bus addresses, not permanent semantic
experts. Every example randomly binds the three unordered teacher plans to the
three physical rows. A shared plan-conditioned adapter is used at every
instrumented layer; there is no independently parameterized adapter bank that
can memorize `D1 = history`, `D2 = biography`, or any other fixed role.

The planner runs once. Generated tokens never re-enter it. Static outline
memory remains visible at every instrumented layer, while the dynamic bus
carries only delayed fragment state. There is no fourth synthesis decoder and
no serial autoregressive pass that combines the three sections.

Natural inference stops each lane at its own EOS while retaining all three
physical rows in the packed KV frontier. Completed rows are logically masked.
Structured mechanism evaluation intentionally uses fixed synchronized block
counts.

## Frozen trunks and scale policy

The first rung is the revision-pinned dense 4B trunk:

- `Qwen/Qwen3-4B-Instruct-2507`
- revision `cdbee75f17c01a7cc42f958dc650907174af0554`
- BF16 frozen weights
- 12 instrumented layers: `2,5,8,11,14,17,20,23,26,29,32,35`

Dense Qwen3-14B is the capacity-control rung. It is run only if the shared 4B
oracle executor fails and the failure could reasonably be model capacity
rather than data, routing, optimization, or plumbing. Lane-specific decoder
stacks are considered only after the identical shared-trunk oracle architecture
fails with both dense 4B and dense 14B.

Use one H100 for the 4B rung. Do not provision two H100s speculatively. A
second H100 is justified only if a measured 14B memory probe proves that the
canonical model must be sharded to fit.

## Real-data contract

Each immutable source packet contains 2,000–8,000 `o200k_base` tokens from a
revision-pinned English Wikipedia snapshot. Article-hash splits make train,
validation, and test source-disjoint.

The OpenAI Batch pipeline has two strict `/v1/responses` stages using the
pinned teacher `gpt-5.4-mini-2026-03-17`:

1. Extract 12–64 atomic source-grounded facts, exact paragraph provenance, and
   one plausible contradicted hard negative per fact.
2. In one joint request, produce one natural expository prompt, exactly three
   unordered plans, and all three long-form teacher sections.

Each teacher section must contain 700–1,000 Qwen tokens, four to twelve
connected paragraphs, developed sentences, and no QA, bullet list, short
answer, or fourth synthesis. Every source fact has exactly one `OWNER` lane.
The other lanes label it `REFERENCE` or `ABSENT`. Every reference is attached
to one exact delayed cross-plan dependency with exact source and target
evidence quotes. Dependencies must come from both siblings and must point from
an earlier sibling paragraph to a later receiver paragraph.

Retokenization stores:

- the exact Qwen tokenizer identity and revision;
- paired positive and hard-negative BGE fact embeddings;
- eight continuous 512-wide semantic plan targets per lane;
- exact fact-to-outline routing;
- exact owner/reference/absent block labels;
- exact dependency token masks, not whole-block approximations;
- source-block to target-token dependency edges that prove the one-block bus
  delay is satisfied after Qwen tokenization;
- outline progress and presentation-order targets.

The tokenized schema is `pdt-real-plan-tokenized-v2`. Older processed rows fail
validation and must be regenerated from immutable raw data. Every prose target
is followed by the pinned Qwen EOS token in training so per-lane stopping is
learned rather than bolted onto inference.

## Build the data

Python 3.12 and `uv` are mandatory. Every long-running command is launched
with `nohup`, logs verbosely, and is polled every 15 seconds.

Start with the 128-example schema pilot. The next paid gate is 2,048 training
examples. Do not request the 20,000-example corpus until the architecture gate
has passed.

For each of `train`, `validation`, and `test`, choose the required count and
create an immutable source file:

```bash
SPLIT=train
COUNT=128
ROOT=data/raw/real_plan/pilot
mkdir -p "$ROOT/sources" logs

nohup uv run scripts/prepare_wikipedia_sources.py \
  --split "$SPLIT" --count "$COUNT" \
  --output "$ROOT/sources/$SPLIT.jsonl" \
  > "logs/wikipedia_${SPLIT}.log" 2>&1 &
```

Build and submit fact extraction:

```bash
mkdir -p "$ROOT/requests" "$ROOT/batch" "$ROOT/facts"

nohup uv run scripts/prepare_real_plan_data.py build-fact-requests \
  --sources "$ROOT/sources/$SPLIT.jsonl" \
  --output "$ROOT/requests/${SPLIT}_facts.jsonl" \
  > "logs/build_${SPLIT}_facts.log" 2>&1 &

uv run scripts/prepare_real_plan_data.py submit \
  --requests "$ROOT/requests/${SPLIT}_facts.jsonl"
```

Record the returned Batch ID. Check it without inventing a polling loop:

```bash
uv run scripts/prepare_real_plan_data.py status --batch-id BATCH_ID
```

After completion, download and validate the immutable result:

```bash
uv run scripts/prepare_real_plan_data.py download \
  --batch-id BATCH_ID \
  --output "$ROOT/batch/${SPLIT}_facts_results.jsonl"

nohup uv run scripts/prepare_real_plan_data.py parse-fact-results \
  --sources "$ROOT/sources/$SPLIT.jsonl" \
  --results "$ROOT/batch/${SPLIT}_facts_results.jsonl" \
  --output "$ROOT/facts/$SPLIT.jsonl" \
  > "logs/parse_${SPLIT}_facts.log" 2>&1 &
```

Build the joint three-lane request, submit it, download it, and parse it:

```bash
mkdir -p "$ROOT/examples"

nohup uv run scripts/prepare_real_plan_data.py build-joint-requests \
  --sources "$ROOT/sources/$SPLIT.jsonl" \
  --facts "$ROOT/facts/$SPLIT.jsonl" \
  --output "$ROOT/requests/${SPLIT}_joint.jsonl" \
  > "logs/build_${SPLIT}_joint.log" 2>&1 &

uv run scripts/prepare_real_plan_data.py submit \
  --requests "$ROOT/requests/${SPLIT}_joint.jsonl"

uv run scripts/prepare_real_plan_data.py download \
  --batch-id JOINT_BATCH_ID \
  --output "$ROOT/batch/${SPLIT}_joint_results.jsonl"

nohup uv run scripts/prepare_real_plan_data.py parse-joint-results \
  --sources "$ROOT/sources/$SPLIT.jsonl" \
  --facts "$ROOT/facts/$SPLIT.jsonl" \
  --results "$ROOT/batch/${SPLIT}_joint_results.jsonl" \
  --output "$ROOT/examples/$SPLIT.jsonl" \
  > "logs/parse_${SPLIT}_joint.log" 2>&1 &
```

Batch submission rejects empty files, more than 50,000 requests, files larger
than 200 MB, failed requests, and all overwrite attempts. Parsed outputs are
published atomically and never modify raw artifacts.

Retokenize each validated split for the 4B trunk:

```bash
PROFILE=qwen3_4b_instruct_2507
PROCESSED="data/processed/real_plan/$PROFILE"
mkdir -p "$PROCESSED"

nohup uv run scripts/retokenize_real_plan.py \
  --input "$ROOT/examples/$SPLIT.jsonl" \
  --output "$PROCESSED/$SPLIT.jsonl" \
  --trunk-profile "$PROFILE" \
  --embedding-device cpu \
  > "logs/retokenize_${SPLIT}_${PROFILE}.log" 2>&1 &
```

Validation is the fact-similarity threshold calibration split. Test is the
held-out generation split. Their article IDs must be disjoint or evaluation
fails.

No real-plan examples have been generated in this workspace yet. Existing
`long_form_dependency` and `pdt_10k` files are older experiments and are not
inputs to this training path.

## Objective and curriculum

After exact six-way matching of the unordered predicted plans to teacher
plans, the configured objective is:

```text
L = 1.00 token CE
  + 1.00 plan semantic
  + 1.00 balanced fact route
  + 0.50 outline progress
  + 1.00 class-balanced fact write
  + 0.25 dynamic-note alignment
  + 0.10 presentation order
  + 0.25 dynamic VQ commitment
  + 1.00 dynamic VQ codebook
  + 0.10 dynamic codebook usage
```

Fact route balances positive and negative queries. Fact write balances
`OWNER`, `REFERENCE`, and `ABSENT`, preventing the many absent labels from
creating a trivial classifier. A fact is correct only when its assigned
physical lane writes it. A correct fact emitted by a different lane does not
rescue owner recall.

The four stages are:

1. `oracle_outline_executor`: inject teacher outlines; freeze planner and
   trunk; train the executor, shared adapters, semantic heads, SNC, and writer.
2. `planner_distillation`: freeze the complete executor; train only the
   continuous unordered planner through semantic, route, and order losses.
3. `joint_packed_rollout`: keep the trunk frozen and train all phi components.
4. `late_joint_training`: continue the same architecture without introducing
   another path.

Checkpoint format 4 stores only phi, optimizer, scheduler, step/stage, frozen
trunk identity, instrumentation topology, and bus/self-only condition. A
checkpoint cannot cross-load into a different trunk revision or scientific
condition.

## H100 bootstrap and first write

On the persistent volume containing the repository:

```bash
mkdir -p experiments/bootstrap
nohup bash scripts/setup_lambda_gpu.sh \
  --trunk-profile qwen3_4b_instruct_2507 \
  > experiments/bootstrap/setup_4b.log 2>&1 &
```

Poll the log every 15 seconds and terminate on an error. The bootstrap uses
the committed lockfile, validates at least 75 GB of visible GPU memory, checks
the pinned Qwen contract, and runs the complete smoke suite.

The first model write is always the two-update optimizer probe:

```bash
RUN=experiments/qwen3_4b_instruct_2507/real_plan_bus
mkdir -p "$RUN/logs"

nohup uv run scripts/train.py \
  --config configs/pdt_qwen3_4b.yaml \
  --trunk-profile qwen3_4b_instruct_2507 \
  --optimizer-probe \
  --telemetry-dir "$RUN" \
  > "$RUN/logs/optimizer_probe.log" 2>&1 &
```

Do not start the paid run unless `optimizer_probe.json` reports finite nonzero
gradients and nonzero parameter movement on the second update, and
`checkpoints/step_00000002.pt` exists.

Launch training with the canonical config:

```bash
nohup uv run scripts/train.py \
  --config configs/pdt_qwen3_4b.yaml \
  --trunk-profile qwen3_4b_instruct_2507 \
  --resume "$RUN/checkpoints/step_00000002.pt" \
  --telemetry-dir "$RUN" \
  > "$RUN/logs/train.log" 2>&1 &
```

The default stage boundaries are `0, 3750, 10000, 25000`. Save and evaluate
stage-boundary checkpoints before treating the next stage as evidence. The
canonical sidecar initialization and data-lane permutation seed is `1729`. The
trainer’s teacher-forced evaluator reruns the exact targets with dynamic notes
removed while preserving static plans, and computes per-token CE changes on
exact dependency tokens versus exact nondependency tokens. It reports
document-level bootstrap intervals; passing plumbing tests never sets these
results.

## Free-generation evidence

One checkpoint-stage-aware command runs the appropriate evidence surface.
Stage 0 evaluates only the oracle-plan executor. Stage 1 and later run:

- oracle plan;
- learned plan;
- learned plan with all plan nodes zeroed;
- learned plan with the physical D1 and D2 plans exchanged.

The fact threshold is frozen on teacher prose from validation, then used
unchanged on held-out test generation. Every result includes the full text for
all three physical lanes.

```bash
CHECKPOINT="$RUN/checkpoints/step_00010000.pt"
EVAL="$RUN/free_generation_step_00010000.json"
ROOT=data/raw/real_plan/pilot
PROCESSED=data/processed/real_plan/qwen3_4b_instruct_2507

nohup uv run scripts/evaluate_real_plan_generation.py \
  --config configs/pdt_qwen3_4b.yaml \
  --checkpoint "$CHECKPOINT" \
  --calibration-raw "$ROOT/examples/validation.jsonl" \
  --calibration-tokenized "$PROCESSED/validation.jsonl" \
  --evaluation-raw "$ROOT/examples/test.jsonl" \
  --evaluation-tokenized "$PROCESSED/test.jsonl" \
  --max-new-tokens 1000 \
  --device cuda \
  --embedding-device cuda \
  --output "$EVAL" \
  > "$RUN/logs/free_generation_step_00010000.log" 2>&1 &
```

Configured checks are descriptive evidence gates, never assertions inserted by
unit tests:

- oracle owner-fact recall lower confidence bound at least 90%;
- oracle unauthorized leakage upper confidence bound at most 10%;
- learned owner recall retains at least 80% of oracle;
- zeroing the plan drops owner recall by at least 30 percentage points;
- the lane-swap output retains at least 80% of learned-plan owner recall when
  scored at the moved addresses;
- moved-address scoring beats the deliberately wrong unmoved-address scoring
  by at least 30 percentage points;
- every lane remains long-form: at least 700 tokens, at least four paragraphs,
  and at most 10% sentences shorter than eight words.

The teacher-forced dynamic-note ablation is a separate causal gate. Plan
dependence does not prove dynamic communication, and dynamic-note sensitivity
does not prove correct decomposition.

## Self-only and larger-model controls

The self-only condition uses the same persistent plan path, parameter count,
instrumented layers, query path, and history horizon. Its SNC replacement can
read only delayed states from the receiver’s own prior blocks. It is a real
trained condition, not a disabled bus or stub. Train it in a separate
telemetry directory with:

```bash
SELF=experiments/qwen3_4b_instruct_2507/real_plan_self_only
mkdir -p "$SELF/logs"
nohup uv run scripts/train.py \
  --config configs/pdt_qwen3_4b.yaml \
  --trunk-profile qwen3_4b_instruct_2507 \
  --coordination-source self_only \
  --telemetry-dir "$SELF" \
  > "$SELF/logs/train.log" 2>&1 &
```

This is run after the bus architecture gate, not concurrently on a
speculatively rented second H100.

If shared 4B fails its oracle executor gate after data and optimization audits,
run the same immutable examples through the pinned dense 14B tokenizer and the
same architecture. Start with one H100 memory/optimizer probe. Provision a
second H100 only if measured memory proves sharding is required.

## Local verification

Local verification checks implementation contracts only:

```bash
uv sync --frozen
uv run ruff check src scripts tests
uv run mypy src scripts
nohup uv run pytest tests -q > nohup.out 2>&1 &
```

Poll `nohup.out` every 15 seconds. The M4 is suitable for schema, unit,
tokenizer, and small CPU/MPS diagnostics. It is not used to infer whether the
4B or 14B scientific architecture works. On 2026-07-16, Ruff passed, mypy
passed across 84 source files, and all 198 tests passed on the local CPU.

The current source of truth for the design is [07_16.md](07_16.md). The living
implementation status is [PLAN.md](PLAN.md).
