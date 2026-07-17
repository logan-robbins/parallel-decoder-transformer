# Model-Intrinsic Parallel Generation

Working paper title:

**Model-Intrinsic Parallel Generation: Planner-Conditioned Latent Coordination
Across Synchronous Decoder Frontiers**

This repository tests whether one frozen causal transformer can be extended
with a learned planner and a delayed latent communication bus so that three
physical decoder frontiers generate three complementary long-form sections at
the same time. The current repository contains the architecture, strict
teacher-output schema, training curriculum, packed runtime, and falsifiable
evaluators. The old Hugging Face Wikipedia sampler and lexical proxy
experiments have been removed. The implemented source path acquires exact
Wikimedia revisions, parses Parsoid HTML with a prose-only allowlist, publishes
immutable eligibility manifests, and cryptographically binds teacher requests
to the accepted manifest. A separately namespaced, manually curated
single-article inspection example now exercises the proposed source, fact,
planner, ownership, dependency, and long-form target contract. It is not a
training corpus and establishes no positive scientific result.

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

The checked-in architecture screen is a deterministic one-factor scientific
screen, not an asserted optimum. It varies trunk scale, instrumented depth,
notes width, SNC width, adapter bottleneck, planner depth and feed-forward
width, product-VQ shape, both residual-gate initializations, learning rate,
weight decay, and the matched bus/self-only condition. It compiles 27 variants
at three independent seeds, for 81 exact run configs:

```bash
uv run scripts/compile_architecture_sweep.py \
  --spec configs/pdt_architecture_screen.yaml \
  --output-dir experiments/architecture_screen/configs
```

Compilation validates and hashes every config without loading or downloading a
model. It is plumbing only. Execute the screen only after admissible pilot data
and the oracle-executor gate exist; selected interactions require a subsequent
registered experiment rather than post-hoc cherry-picking.

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
The other lanes label it `REFERENCE` or `ABSENT`. The three semantic lanes
have an explicit reader-facing presentation order, while all six
semantic-to-`D1`/`D2`/`D3` assignments remain eligible so a physical decoder
cannot acquire a permanent historical role.

Cross-lane references are optional. The three physical decoder frontiers are
the required supervision target; note-bus use is a separate, sparse signal.
When a reference exists, one exact dependency must bind its owner and receiver
quotes, its owner lane must precede its receiver in presentation order, and
its owner paragraph must be available in an earlier generation block. The
compiler permits zero dependencies and rejects forward or same-block
dependencies.

Every manually taught record also carries a teacher audit covering the
featured-article quality basis, representative citations used by selected
fact paragraphs, complete historical-claim coverage, and decomposition
naturalness. Representative citation IDs are validated against the extracted
source and the exact fact-evidence paragraphs.

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

## Manually taught real-data contract inspection

The inspection path under `src/model_intrinsic_parallel` imports no existing
project dataset or dataset code. Its inherited inputs are immutable pinned
Wikimedia revisions only. The current inspection set contains four manually
taught featured articles with distinct narrative structures:

| Example | Cited facts | Fact paragraphs | Top sections | Notes | Qwen lane tokens |
| --- | ---: | ---: | ---: | ---: | --- |
| Baltimore railroad strike of 1877 | 24 | 22 | 9 | 3 | 765 / 777 / 771 |
| Blackwater Fire of 1937 | 18 | 12 | 4 | 0 | 761 / 731 / 705 |
| Battle of Sluys | 18 | 17 | 4 | 1 | 812 / 751 / 733 |
| Great Stink | 18 | 16 | 5 | 2 | 771 / 731 / 775 |

Rebuild the content-only source directly from the raw revision:

```bash
uv run scripts/extract_wikipedia_inspection_source.py \
  --raw-jsonl data/raw/historical/pilot_8_midscale/pinned_revisions.jsonl \
  --title "Baltimore railroad strike of 1877" \
  --revision-id 1361100762 \
  --output data/model_intrinsic_parallel/source/baltimore_railroad_strike_1877.json
```

Compile the manually curated inspection record with the actual locally cached
Qwen tokenizer:

```bash
uv run scripts/compile_wikipedia_inspection_example.py \
  --source data/model_intrinsic_parallel/source/baltimore_railroad_strike_1877.json \
  --curation data/model_intrinsic_parallel/curation/baltimore_railroad_strike_1877.json \
  --output data/model_intrinsic_parallel/examples/baltimore_railroad_strike_1877.json \
  --tokenizer-model Qwen/Qwen3-4B-Base \
  --tokenizer-revision 906bfd4b4dc7f14ee4320094d8b41684abff8539
```

Acquire and screen the additional pinned revisions with:

```bash
uv run scripts/acquire_wikipedia_audit_candidates.py \
  --titles data/model_intrinsic_parallel/candidates/audit_round_2_titles.txt \
  --output data/raw/model_intrinsic_parallel/audit_round_2/pinned_revisions.jsonl \
  --user-agent "https://github.com/logan-robbins/parallel-decoder-transformer"

uv run scripts/screen_wikipedia_audit_candidates.py \
  --raw-jsonl data/raw/model_intrinsic_parallel/audit_round_2/pinned_revisions.jsonl \
  --output-dir data/model_intrinsic_parallel/source/audit_round_2 \
  --screen-output data/model_intrinsic_parallel/source/audit_round_2_screen.json \
  --tokenizer-model Qwen/Qwen3-4B-Base \
  --tokenizer-revision 906bfd4b4dc7f14ee4320094d8b41684abff8539
```

Raw acquisition is immutable and fails if its output path already exists.
Screening removes lists, tables, templates, and complete reference-like
subtrees, including combined headings such as `Notes, citations and sources`.
It writes only derived content-heading and ordinary-prose documents.

Every compiled target paragraph declares the complete set of facts allowed to
support its historical claims, and compilation requires that set to equal its
`OWNER`/`REFERENCE` realizations exactly. The records identify themselves as
`manual-source-grounded-inspection` and `data-contract-inspection-only`; they
validate the data and architecture contract but are not empirical results.

Run its extraction, integrity, ownership, binding, and token-mutation tests:

```bash
uv run pytest tests/model_intrinsic_parallel -q
```

## Data build and stop gate

`scripts/prepare_wikipedia_sources.py` no longer exists. The only accepted
source input is a reviewed `pdt-historical-candidate-v1` JSONL catalog. Each row
pins `page_id`, `revision_id`, exact title, `EVENT` or `PROCESS`, one of the
eight historical categories, reviewed event end year, family keys, and related
page IDs. This review is a real input requirement; the code does not ask a
classifier to invent dates, categories, or family relationships.

Acquisition cross-checks the candidate against both the official Action API
and the REST revision-with-HTML response. It verifies the revision SHA-1,
retains the raw Action response, raw wikitext, semantic HTML, assessments,
reference records, licenses, and exact hashes, and parses maintenance
transclusions from Parsoid metadata. Any failed candidate prevents publication
of a partial raw file.

For a 128-record pilot, review more than 16 candidates per category because
eligibility rejection and family deduplication are expected. Then run:

```bash
RAW=data/raw/historical/pilot
SOURCE_ROOT=data/accepted/historical/pilot
mkdir -p "$RAW/logs"

nohup uv run scripts/acquire_historical_revisions.py \
  --candidates "$RAW/candidates.jsonl" \
  --output "$RAW/pinned_revisions.jsonl" \
  --user-agent "PDT-research/1.0 contact@example.com" \
  --acquisition-date 2026-07-17 \
  > "$RAW/logs/acquire.log" 2>&1 &

nohup uv run scripts/prepare_historical_sources.py \
  --input "$RAW/pinned_revisions.jsonl" \
  --output-dir "$SOURCE_ROOT" \
  --examples-per-category 16 \
  --trunk-profile qwen3_4b_instruct_2507 \
  > "$RAW/logs/filter.log" 2>&1 &
```

Poll each log every 15 seconds and terminate on error. The filter is local-only
and requires the pinned Qwen fast tokenizer already present in cache. It
atomically publishes `accepted_sources.jsonl`, `accepted_manifest.json`,
`rejections.json`, and `selection.json`; it never weakens a threshold to fill a
quota. If eligibility or balanced selection fails, the same output directory
instead contains immutable `failure.json` and `rejections.json` evidence and
the command exits nonzero. A failed balanced selection also retains every
fully validated row in integrity-bound `eligible_sources.jsonl`, but never
publishes an accepted manifest. The teacher-data command cannot consume that
partial pool. Verify the exact accepted manifest bytes before constructing any
teacher request:

```bash
MANIFEST="$SOURCE_ROOT/accepted_manifest.json"
MANIFEST_SHA=$(shasum -a 256 "$MANIFEST" | awk '{print $1}')

uv run scripts/prepare_real_plan_data.py build-fact-requests \
  --sources "$SOURCE_ROOT/accepted_sources.jsonl" \
  --accepted-manifest "$MANIFEST" \
  --accepted-manifest-sha256 "$MANIFEST_SHA" \
  --output data/teacher/pilot/fact_requests.jsonl
```

The fact-stage output must validate all 18–48 facts, exact paragraph quotes,
local reference IDs, twelve-paragraph/four-section distribution, hard
negatives, and complete source identity before joint requests can be built.
Joint results are validated again against the same source and manifest hashes.
After manual inspection, publish the source-family split bundle:

```bash
uv run scripts/prepare_real_plan_data.py split-examples \
  --input data/raw/real_plan/pilot/all_examples.jsonl \
  --output-dir data/raw/real_plan/pilot/examples
```

Retokenization is also local-only and atomically publishes
`pdt-real-plan-tokenized-v3`:

```bash
uv run scripts/retokenize_real_plan.py \
  --input data/raw/real_plan/pilot/examples/train.jsonl \
  --output data/processed/real_plan/qwen3_4b_instruct_2507/train.jsonl \
  --trunk-profile qwen3_4b_instruct_2507 \
  --embedding-device cpu
```

Python 3.12 and `uv` remain mandatory. Every future long-running command must
use `nohup`, verbose logging, and 15-second polling. No paid data or GPU command
is currently authorized.

The next external work is therefore concrete: create and review the candidate
catalog, acquire and filter it, inspect all acceptance/rejection records, and
only then authorize the cheap fact Batch. Manually inspect all 128 final pilot
records. Freeze renderer, schemas, prompts, manifests, and rejection report
before retokenization and the one-H100 optimizer/oracle probe. Expand to 2,048
only after the pilot contract passes. Do not request 20,000 until the
architecture shows value.

The 2026-07-17 live acquisition screens downloaded two eight-row candidate
batches. The broad batch was entirely oversized. The mid-scale batch produced
one strict eligible source, `Baltimore railroad strike of 1877`: 6,474 pinned
Qwen tokens, 14 sections, 61 substantive paragraphs, and 55 reference records.
The integrity-bound real source record is retained in
`eligible_sources.jsonl`, but the batch could not fill all eight categories, so
it published failure evidence rather than an accepted manifest. No teacher
request is authorized from either screen.

No teacher-generated pilot corpus has been produced in this workspace. The one
manually curated inspection record is sufficient to inspect the proposed data
contract, but it is not distributional training data and cannot satisfy an
empirical gate. Planner trainability tests establish tensor and gradient
plumbing only.

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
  --entailment-device cuda \
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

The pinned NLI scores are an automatic screen, not final factual evidence.
Export every condition × lane × fact decision into a blinded queue, collect two
complete independent annotations, and send every disagreement or `UNCERTAIN`
item to a distinct third adjudicator:

```bash
uv run scripts/manual_fact_audit.py export \
  --generation-evaluation "$EVAL" \
  --raw-examples "$ROOT/examples/test.jsonl" \
  --queue "$RUN/manual_fact_queue.jsonl" \
  --key "$RUN/manual_fact_key.json" \
  --randomization-seed 1729

uv run scripts/manual_fact_audit.py adjudicate \
  --queue "$RUN/manual_fact_queue.jsonl" \
  --key "$RUN/manual_fact_key.json" \
  --annotator-a "$RUN/annotations_a.jsonl" \
  --annotator-b "$RUN/annotations_b.jsonl" \
  --adjudicator "$RUN/adjudications.jsonl" \
  --output "$RUN/manual_fact_result.json"
```

The hidden key scores owner/reference/absent roles at exact physical lanes and
derives the lane-swap unmoved-address control from the same judgments. No
automatic fact score can set the final human evidence gate.

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

The self-only checkpoint uses the same free-generation evaluator and human
audit, with the condition identity overridden before strict checkpoint load:

```bash
nohup uv run scripts/evaluate_real_plan_generation.py \
  --config configs/pdt_qwen3_4b.yaml \
  --coordination-source self_only \
  --checkpoint "$SELF/checkpoints/step_00010000.pt" \
  --calibration-raw "$ROOT/examples/validation.jsonl" \
  --calibration-tokenized "$PROCESSED/validation.jsonl" \
  --evaluation-raw "$ROOT/examples/test.jsonl" \
  --evaluation-tokenized "$PROCESSED/test.jsonl" \
  --max-new-tokens 1000 \
  --device cuda \
  --entailment-device cuda \
  --output "$SELF/free_generation_step_00010000.json" \
  > "$SELF/logs/free_generation_step_00010000.log" 2>&1 &
```

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
4B or 14B scientific architecture works. The local suite also exercises the
packed self-only generation path with explicit contract doubles; that proves
reachability and causality checks only, not model quality. On 2026-07-17, Ruff
and mypy passed across 85 source files and all 248 tests passed locally.

The current source of truth for the design is [07_16.md](07_16.md). The living
implementation status is [PLAN.md](PLAN.md).
