# Model-Intrinsic Parallel Generation

Working paper title:

**Model-Intrinsic Parallel Generation: Planner-Conditioned Latent Coordination
Across Synchronous Decoder Frontiers**

This repository tests whether one frozen causal transformer can be extended
with a learned planner and a delayed latent communication bus so that three
physical decoder frontiers generate three complementary long-form sections at
the same time. The current repository contains the architecture, strict
teacher-output schema, training curriculum, packed runtime, and falsifiable
evaluators. Its existing Hugging Face Wikipedia sampler is not admissible for
real training and must be replaced by the historical-source ingress specified
below. The repository does not yet contain a positive result from the new
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

The first corpus contains reliable historical events and processes only. It
does not admit arbitrary Wikipedia pages, biographies as primary topics,
lists, timelines, chronologies, indexes, year pages, outlines, current events,
or catalog-like pages.

Source ingress must retain immutable pinned-revision wikitext, rendered
semantic HTML, PageAssessments metadata, and parsed reference records. The
student-visible renderer is a strict allowlist containing only the article
title, hierarchical content headings, and ordinary prose paragraphs. Inline
links contribute only their display text. Tables, lists, infoboxes, templates,
figures, captions, galleries, maps, citation markers, footnotes, references,
bibliographies, URLs, navigation, categories, authority-control blocks,
hatnotes, pronunciation, math, code, and edit artifacts never enter
model-visible text.

An accepted article must be English main-namespace, non-redirect,
non-disambiguation, at least 25 years removed from the end of the event, tied
to a history-oriented WikiProject, and rated `FA`, `GA`, `A`, or `B` by a
relevant project. Its complete cleaned body must be 3,000–7,000 pinned-Qwen
tokens with at least six substantive sections and twelve substantive
paragraphs. Articles are accepted or rejected whole; they are never truncated
or stitched.

Each article requires at least 30 inline reference occurrences, 15 distinct
cited works, eight identifiable scholarly or institutional sources, citations
on at least 70% of substantive paragraphs, and no source contributing more
than 25% of occurrences. Citation, neutrality, disputed-accuracy,
original-research, cleanup, and hoax maintenance templates cause rejection.
Paragraph-to-reference relations remain separate audit metadata and are not
rendered to the student.

The OpenAI Batch pipeline has two strict `/v1/responses` stages using the
pinned teacher `gpt-5.4-mini-2026-03-17`:

1. Extract 18–48 cited atomic facts, exact paragraph/reference provenance, and
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

The existing `pdt-real-plan-v1` and `pdt-real-plan-tokenized-v2` contracts
belong to the rejected source-ingress plumbing and are inadmissible for paid
data. The replacement must publish `pdt-historical-source-v1`,
`pdt-real-plan-v2`, and `pdt-real-plan-tokenized-v3`. Every prose target is
followed by the pinned Qwen EOS token in training so per-lane stopping is
learned rather than bolted onto inference.

## Data-build stop gate and handoff

Do not run `scripts/prepare_wikipedia_sources.py`. It reads cleaned Hugging
Face rows that have lost heading/citation relationships and it truncates
oversized articles to a prefix. Do not submit any fact or joint Batch request
from its output. The existing Batch, schema, and retokenization code is
plumbing awaiting admissible source ingress.

Python 3.12 and `uv` remain mandatory. Every future long-running command must
use `nohup`, verbose logging, and 15-second polling. No paid data or GPU command
is currently authorized.

The next operator must complete these steps in order:

1. Replace the sampler with pinned-revision Wikimedia wikitext plus rendered
   semantic-DOM acquisition while preserving immutable raw inputs.
2. Implement the heading/prose allowlist and prove that no forbidden node or
   excluded section reaches model-visible source text.
3. Extend source provenance with heading paths, paragraph-local reference IDs,
   parsed reference metadata, revision identity, assessment metadata, and
   deterministic acceptance/rejection reasons. Bump the source, raw-example,
   and tokenized schema identities to the versions specified above.
4. Enforce the historical-domain, relevant `FA`/`GA`/`A`/`B` assessment,
   complete-body 3,000–7,000-token, section, paragraph, citation-distribution,
   scholarly-source, source-dominance, and maintenance-template gates.
5. Cluster near-duplicate and related event families before assigning splits.
   An entire family must remain in exactly one split.
6. Add golden and adversarial parser tests. Make Batch request construction
   fail unless the source file matches the exact SHA-256 of its accepted
   manifest.
7. Balance the 128 accepted pilot examples across wars/battles,
   revolutions/transitions, treaties/crises, social movements/reforms,
   exploration/migration, scientific/industrial developments,
   disasters/reconstruction, and cultural/institutional transformations.
8. Run the cheap fact stage first and require 18–48 cited facts distributed
   across twelve paragraphs and four sections plus a feasible three-way
   decomposition.
9. Submit joint three-lane requests only for candidates that pass the fact
   gate. Manually inspect all 128 accepted records.
10. Freeze the renderer, schema, prompts, manifests, and rejection report
    before retokenization and the single-H100 optimizer/oracle probe. Expand to
    2,048 only after those gates pass. Do not request 20,000 until the
    architecture shows value.

No real-plan examples have been generated in this workspace. Existing
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
