# Dataset Generation Pipeline

5-stage pipeline that converts raw text sources into training-ready JSONL files for the Parallel Decoder Transformer. Each output record provides the supervision signals required by the training objectives defined in [PAPER.md](../../../docs/arxiv_submission/PAPER.md) (Appendix A): planner slot IDs, true/speculative notes tensors, coverage targets, and continuation-sufficiency (agreement) labels.

## Pre-Generated Datasets

Pre-generated datasets from the production pipeline are available for download:

**https://storage.googleapis.com/parallel-decoder-transformer/data/archives/**

```bash
# Training split (2.7GB compressed, ~47GB uncompressed)
mkdir -p data/processed/pdt_10k_gpt41
wget https://storage.googleapis.com/parallel-decoder-transformer/data/archives/pdt_10k_gpt41_jsonl_train.tar.gz
tar -xzf pdt_10k_gpt41_jsonl_train.tar.gz -C data/processed/

# Evaluation splits (647MB compressed, ~5GB uncompressed)
wget https://storage.googleapis.com/parallel-decoder-transformer/data/archives/pdt_10k_gpt41_jsonl_eval.tar.gz
tar -xzf pdt_10k_gpt41_jsonl_eval.tar.gz -C data/processed/

# Verify
ls data/processed/pdt_10k_gpt41/
# kd_train.jsonl  kd_validation.jsonl  kd_test.jsonl
```

## Pipeline Overview

| Stage | Name | Input | Output | LLM Cost |
|-------|------|-------|--------|----------|
| 1 | Preflight | Wikipedia manifest | `preflight_accepted.jsonl` | ~$0.001/article |
| 2 | Plan Generation | Accepted articles | `{domain}/{sample_id}.json` plan files | ~$0.02/plan |
| 3 | Notes Generation | Plan files | `{domain}/{sample_id}.json` notes files | ~$0.18/plan |
| 4 | Collation | Notes files | `train/validation/test.parquet` | $0 |
| 5 | KD Export | Parquet splits | `kd_{split}.jsonl` | $0 |

Cost and timing figures in this document are planning estimates derived from the current configs and API settings.

## Environment Setup

```bash
uv venv .venv --python 3.12
uv sync

# API keys (create .env in repository root)
OPENAI_API_KEY=your_key_here
OPENAI_ORG_ID=your_org_id  # Optional
```

## Configuration

Production configs are in `configs/dataset/`. The production config uses a **split-model** architecture:

- **Plan generation**: `gpt-5.1` (reasoning model, `reasoning_effort: low`, `service_tier: flex`)
- **Notes generation**: `gpt-4.1` (non-reasoning, `service_tier: default`)

This is controlled by the `notes_llm` field in the YAML config. When set, Stage 2 uses the primary `llm` config and Stage 3 uses `notes_llm`. The resolution logic is in `config.py:resolve_notes_llm_config()`.

```yaml
# configs/dataset/notes_gpt41_production.yaml
llm:
  backend: openai
  openai:
    model: gpt-5.1
    service_tier: flex
    client_timeout: 900.0
    reasoning_effort: low

notes_llm:
  backend: openai
  openai:
    model: gpt-4.1
    service_tier: default
    client_timeout: 600.0
```

### Generation Hyperparameters

Separate `GenerationConfig` instances control each stage (defined in `config.py:DatasetBuildConfig`):

| Stage | Temperature | Max Tokens | Purpose |
|-------|-------------|------------|---------|
| Plan generation | 0.2 | 100,000 | Low temperature for deterministic decomposition |
| True notes | 0.2 | 16,384 | Low temperature for faithful extraction |
| Speculative notes | 0.2 | 16,384 | Controlled hallucination via prompt, not temperature |

## Stage 1: Preflight Validation

Filters source articles for structural suitability using deterministic checks followed by LLM-based classification. Implemented in `preflight.py`.

### Two-Stage Filtering Strategy

**Character pre-filter (fast)**: Eliminates ~95% of rejects without tokenization.
- Range: 10,000–30,000 characters (configurable)

**Token filter (precise)**: Validates survivors against exact token budgets.
- Tokenizer: `tiktoken cl100k_base` (100,277 vocab) — used for fast approximate filtering only
- Range: 200–25,000 tokens, 100,000 total budget

**LLM classifier** (`WikiArticleClassifier`): Sends the first 8,000 characters plus up to 15 headings to the LLM. Returns a structured JSON with:
- `keep` (bool), `tier` (tier1–tier4 or avoid), `article_type`, `rationale`, `rejection_reason`

### Command

```bash
uv run scripts/preflight_plans.py \
  --survey 1500 \
  --output-dir data/prep/preflight/pdt_10k_gpt41 \
  --wiki-classifier-model gpt-4.1 \
  --wiki-classifier-service-tier default \
  --wiki-classifier-batch-size 100 \
  --wiki-classifier-concurrency 8 \
  --wiki-min-article-chars 10000 \
  --wiki-max-article-chars 30000
```

### Outputs

- `preflight_accepted.jsonl` — validated candidates with `sample_id`, `domain`, `messages`, `token_counts`, `source_metadata`, `dedupe_key`
- `preflight_rejected.jsonl` — rejected candidates with `reason`
- `preflight_report.json` — summary counts by domain and rejection reason

### Resume

Manual via `--wiki-offset`. Check last processed index:
```bash
jq -r '.source_metadata.dataset_index' data/prep/preflight/pdt_10k_gpt41/preflight_accepted.jsonl | sort -n | tail -1
```

## Stage 2: Plan Generation

Decomposes each source article into a 3-stream plan using OpenAI Structured Outputs with strict JSON schema validation. Implemented in `plan_generation.py`.

### Plan Schema

Each plan contains exactly 3 streams, each with:
- `header` — section title
- `summary` — 1-sentence descriptor for note seeding
- `entities` — 2–3 ENT templates (`"ENT:name=X,type=Y"`)
- `constraints` — 1–2 FACT/COVERAGE templates
- `section_contract` — exact character slice (`{type, start_idx, end_idx}`) of the source this stream owns
- `notes_contract` — ≥2 mandatory semantic requirements
- `reasoning` — 3–5 step justification citing character counts for load balance

Schema enforces `additionalProperties: false` everywhere and uses `json_schema` strict mode at the API level, which is intended to keep plan outputs schema-conformant.

### Domain-Specific Prompts

Three prompts in `plan_generation.py` adapt the decomposition strategy by domain:

| Domain | Strategy | Prompt |
|--------|----------|--------|
| Survey (Wikipedia) | Topic-based splitting | `WIKI_PROMPT` — enforces "Blind Start" rule (Stream 2 must not reference Stream 1), biography guard (`BIO_SKIP_TOKEN`), load balancing |
| QA (SQuAD) | Triangulation | `SQUAD_PROMPT` — Stream 1: grounding, Stream 2: reasoning, Stream 3: synthesis |
| Math (Reasoning Gym) | State hand-off | `RG_PROMPT` — Stream 1: setup, Stream 2: execution, Stream 3: verification |

### Biography Guard

If the LLM determines an article is a biography, it returns `"SKIP_BIOGRAPHY"` instead of a plan. The generator logs this as a soft skip (not a failure). Biographies are poor candidates for topical parallel decomposition.

### Command

```bash
uv run scripts/run_dataset_pipeline.py \
  --config configs/dataset/notes_gpt41_production.yaml \
  --plan-dir data/prep/plans/pdt_10k_gpt41 \
  --notes-dir data/prep/notes/pdt_10k_gpt41 \
  --dataset-dir data/datasets/pdt_10k_gpt41 \
  --survey 1000 \
  --preflight-manifest data/prep/preflight/pdt_10k_gpt41/preflight_accepted.jsonl \
  --skip-notes --skip-collate --skip-kd-export \
  --plan-batch-size 18 --plan-concurrency 12
```

### Resume

Automatic — skips plans where `{plan_dir}/{domain}/{sample_id}.json` already exists. Disable with `--no-plan-resume`.

## Stage 3: Notes Generation

Generates true notes (teacher) and speculative notes (student) with controlled hallucinations. Implemented in `notes_generation.py`. This is the most complex and expensive stage.

### Notes Schema (ENT/FACT/COVERAGE)

The canonical Python data model is in `data/extraction/schema.py` (version 2.0):

- **ENT** (EntityCard): `id`, `name`, `type`, `canonical` (bool), `aliases`
- **FACT** (FactStatement): `subj_id`, `predicate`, `object`, `evidence_span` (start, end, text), `certainty` [0,1]
- **COVERAGE** (CoverageSignal): `plan_item_id`, `status` (COVERED/PARTIAL/MISSING)

### Compact Wire Format

LLM responses use a compact array format to reduce token footprint relative to verbose JSON:

```json
{
  "notes": [
    ["ENT", "E1", "Maienfeld", "municipality", true, []],
    ["FACT", "E1", "located_in", "Graubünden", 1.0, [0, 50, "text span"]],
    ["COVERAGE", "stream_1_item_1", "complete"]
  ]
}
```

### Per-Stream Parallel Architecture

Each LLM request sees only a single stream's text slice (from `section_contract`). This enforces the "Blind Start" rule — streams don't see each other's content at generation time, mirroring the inference protocol.

**Pass 1 — Teacher (true notes):** 3 parallel requests per plan (one per stream). Each receives the stream's text slice with 0-based local indexing. After response, `_remap_evidence_indices()` adds `section_contract.start_idx` to convert evidence spans to document-absolute coordinates.

**Pass 2 — Student (speculative notes):** 3 requests per stream per variant = 9 requests per plan (with default 3 variants). Each receives the stream's true notes and degrades them at 15–20% noise rate through:
- Subtle corruption (change a detail in a FACT's object while keeping the original evidence span — creating an explicit contradiction the Agreement Head is trained to detect)
- Hallucination (add non-existent aliases or entities)
- Omission (drop non-critical facts)

**Total API calls per plan: 12** (3 true + 9 speculative).

### Post-Processing

After both passes, for each plan (`_postprocess_batch`):
1. Derives initial seed notes from the plan contract via `derive_initial_notes_from_plan()` (`plan_contract_notes.py`)
2. Merges seed notes with true notes via `merge_seed_notes()`
3. Builds versioned snapshot chain via `build_versioned_note_snapshots()` — snapshot 0 = `plan_contract` (mirrors the inference-time planner initialization), snapshot 1 = `teacher_true`
4. Generates procedural snapshots from evidence span positions via `generate_procedural_snapshots()` (`procedural_snapshots.py`) — facts appear in the bus in reading order
5. Assigns rollback flags (random, probability 0.15–0.25)

### Snapshot 0 Contract

The pipeline explicitly materializes snapshot 0 as a `plan_contract` versioned note. This mirrors the inference-time behavior described in the paper where the planner initializes the Dynamic Notes Bus before any tokens are emitted. The derivation logic in `plan_contract_notes.py` parses `ENT:`, `FACT:`, and `COVERAGE:` templates from the plan's `entities`, `constraints`, and `notes_contract` fields.

### Command

```bash
nohup uv run scripts/run_dataset_pipeline.py \
  --config configs/dataset/notes_gpt41_production.yaml \
  --plan-dir data/prep/plans/pdt_10k_gpt41 \
  --notes-dir data/prep/notes/pdt_10k_gpt41 \
  --dataset-dir data/datasets/pdt_10k_gpt41 \
  --skip-plans --skip-collate --skip-kd-export \
  --notes-batch-size 18 --notes-concurrency 12 \
  --log-level INFO > logs/notes_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Resume

Automatic — skips notes where `{notes_dir}/{domain}/{sample_id}.json` already exists. Disable with `--no-notes-resume`.

## Stage 4: Collation

Tokenizes notes and exports to Arrow/Parquet with train/validation/test splits. Implemented in `collation.py`.

### Tokenizer

Collation uses the **GPT-OSS-20B native tokenizer** (199,998 vocab). This is the correct tokenizer for downstream training. Falls back to `tiktoken cl100k_base` if the tokenizer path is unavailable.

### Augmentation

Creates `--augment N` additional copies per sample (default 2) with perturbed timing parameters:
- `lag_delta` ±1 (minimum 1)
- `note_cadence_M` ±2 (minimum 2)

This multiplies effective dataset size by `(1 + N)` without additional LLM cost. With `--augment 2`, 1,000 notes become 3,000 records.

### Command

```bash
uv run scripts/run_dataset_pipeline.py \
  --notes-dir data/prep/notes/pdt_10k_gpt41 \
  --dataset-dir data/datasets/pdt_10k_gpt41 \
  --skip-plans --skip-notes --skip-kd-export \
  --augment 2 --max-len 8192
```

### Outputs

Parquet files in `{dataset-dir}/`:
- `train.parquet` (80%), `validation.parquet` (10%), `test.parquet` (10%)
- `manifest.json` (counts and config)

Columns include: `sample_id`, `domain`, `x_text`, `plan_text`, `z_n`, `z_hat`, `notes_true`, `notes_speculative`, `notes_versioned`, `x_tokens`, `z_n_tokens`, `z_hat_tokens`, `plan_tokens`, `lag_delta`, `note_cadence_M`.

## Stage 5: KD JSONL Export

Transforms Parquet splits into JSONL records consumed by `KDJsonlDataset` during training. Implemented in `kd_export.py`. One JSONL record per stream per example.

### Planner ID Generation

Plan text items are hashed into the 65,536-entry latent planner vocabulary via `utils/plan_catalog.py`:

1. `canonical_plan_catalog_entries()` — builds a deterministic ordered list from `notes_contract`, `summary`, `header`, and `section_contract` fields
2. `hash_plan_text()` — normalizes to lowercase, prepends salt (`"parallel-decoder-v1::"`), SHA-256 hashes, takes modulo 65,536 (maps 0 → 1 to avoid the pad token)
3. Output `planner_ids` are padded to 16 slots (the paper's S = 16 fixed plan slots)

### Notes Embedding

Teacher and student notes are embedded via `_HashingStreamEmbedder` (wraps `_HashingEmbedder` from `data/teacher_provider.py`). This produces deterministic dense vectors from stringified notes using SHA-256 hashing — no local embedding model required.

### Continuation Sufficiency Labels

Implements the agreement/readiness supervision target. All slots are 1 (safe to continue) unless `rollback_flags.triggered=True`, in which case the last `event_count` slots are set to 0.

### Command

```bash
uv run scripts/run_dataset_pipeline.py \
  --dataset-dir data/datasets/pdt_10k_gpt41 \
  --processed-dir data/processed/pdt_10k_gpt41 \
  --notes-dim 2048 \
  --kd-splits train validation test \
  --skip-plans --skip-notes --skip-collate
```

### Output Record Format

Each JSONL line represents one stream from one document:

```json
{
  "example_id": "survey_....:stream_1",
  "sample_id": "survey_....",
  "stream_id": "stream_1",
  "domain": "survey",
  "split": "train",
  "student_ids": [101, 7865, ...],
  "student_labels": [7865, 2548, ...],
  "planner_ids": [412, 9801, ...],
  "notes_student": [[...], [...], [...]],
  "notes_teacher": [[...], [...], [...]],
  "teacher_snapshots": [...],
  "student_snapshots": [...],
  "continuation_sufficiency_labels": [1, 1, ...],
  "metadata": {
    "document_text": "...",
    "teacher_plan": {...},
    "teacher_notes": {...},
    "notes_versioned": [...],
    "sectional_independence": true,
    "coverage_provenance": {...}
  },
  "true_notes": [...],
  "plan_tokens": ["Must identify canton", ...]
}
```

### Resume

Append-mode resume via `example_id` deduplication — reads existing IDs from the JSONL file on startup and skips records that already exist.

## Full Production Workflow

```bash
# Stage 1: Preflight (survey 1500, expect ~1000 accepted)
uv run scripts/preflight_plans.py \
  --survey 1500 \
  --output-dir data/prep/preflight/pdt_10k_gpt41 \
  --wiki-classifier-model gpt-4.1 \
  --wiki-min-article-chars 10000 --wiki-max-article-chars 30000

# Stage 2: Plans
uv run scripts/run_dataset_pipeline.py \
  --config configs/dataset/notes_gpt41_production.yaml \
  --plan-dir data/prep/plans/pdt_10k_gpt41 \
  --notes-dir data/prep/notes/pdt_10k_gpt41 \
  --dataset-dir data/datasets/pdt_10k_gpt41 \
  --survey 1000 \
  --preflight-manifest data/prep/preflight/pdt_10k_gpt41/preflight_accepted.jsonl \
  --skip-notes --skip-collate --skip-kd-export \
  --plan-batch-size 18 --plan-concurrency 12

# Stage 3: Notes (background, longest stage)
nohup uv run scripts/run_dataset_pipeline.py \
  --config configs/dataset/notes_gpt41_production.yaml \
  --plan-dir data/prep/plans/pdt_10k_gpt41 \
  --notes-dir data/prep/notes/pdt_10k_gpt41 \
  --dataset-dir data/datasets/pdt_10k_gpt41 \
  --skip-plans --skip-collate --skip-kd-export \
  --notes-batch-size 18 --notes-concurrency 12 \
  --log-level INFO > logs/notes_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Stage 4: Collation
uv run scripts/run_dataset_pipeline.py \
  --notes-dir data/prep/notes/pdt_10k_gpt41 \
  --dataset-dir data/datasets/pdt_10k_gpt41 \
  --skip-plans --skip-notes --skip-kd-export \
  --augment 2 --max-len 8192

# Stage 5: KD Export
uv run scripts/run_dataset_pipeline.py \
  --dataset-dir data/datasets/pdt_10k_gpt41 \
  --processed-dir data/processed/pdt_10k_gpt41 \
  --skip-plans --skip-notes --skip-collate
```

**Final outputs** (for 1,000 articles with `--augment 2`):
- `kd_train.jsonl`: 2,400 records × 3 streams = 7,200 lines
- `kd_validation.jsonl`: 300 × 3 = 900 lines
- `kd_test.jsonl`: 300 × 3 = 900 lines

## Module Reference

| Module | Stage | Purpose |
|--------|-------|---------|
| `config.py` | All | Configuration dataclasses (`DatasetBuildConfig`, `LLMConfig`, `GenerationConfig`, etc.) |
| `preflight.py` | 1 | Two-stage article filtering + LLM classification |
| `plan_generation.py` | 2 | 3-stream plan decomposition with strict JSON schema |
| `notes_generation.py` | 3 | Per-stream true/speculative notes with controlled hallucinations |
| `plan_contract_notes.py` | 3 | Snapshot 0 derivation from plan contract templates |
| `procedural_snapshots.py` | 3 | Rule-based snapshot generation from evidence span positions |
| `collation.py` | 4 | Tokenization and Parquet export with augmentation |
| `kd_export.py` | 5 | JSONL export with planner ID hashing and notes embedding |
| `async_llm.py` | 2–3 | Async HTTP client for OpenAI Responses API |
| `example.py` | — | Core domain objects (`DatasetExample`, `PlanPayload`, etc.) |
| `processors.py` | — | Utility methods for extracting plan/notes data |
| `qc.py` | — | Post-generation quality control (NLI/embedding checks, disabled by default) |
| `sourcing.py` | — | Wikipedia article streaming from HuggingFace |

Supporting modules outside `datasets/`:
- `data/extraction/schema.py` — canonical ENT/FACT/COVERAGE Python data model
- `data/teacher_provider.py` — `_HashingEmbedder` for deterministic notes vectors
- `utils/plan_catalog.py` — SHA-256 hashing pipeline for latent planner vocabulary

## Monitoring

```bash
# Check preflight results
jq '.rejected_by_reason' data/prep/preflight/pdt_10k_gpt41/preflight_report.json

# Count plans/notes
find data/prep/plans/pdt_10k_gpt41 -name "*.json" | wc -l
find data/prep/notes/pdt_10k_gpt41 -name "*.json" | wc -l

# Monitor notes generation
tail -f logs/notes_*.log | grep -E "(succeeded|Pass|Completed)"

# Check for errors
grep -i "error\|failed" logs/notes_*.log | grep -v "error-prone" | tail -20

# Count final JSONL records
wc -l data/processed/pdt_10k_gpt41/kd_*.jsonl
```

## Troubleshooting

**Timeouts / HTTP 502**: Expected with long-running jobs. The async client (`async_llm.py`) retries on 408, 429, and 5xx errors with exponential backoff (max 30s, ±20% jitter).

**Incomplete responses** (reasoning models): If a reasoning model exhausts its token budget, the client raises a non-retryable error recommending 50k+ token budgets. The retry logic expands the token budget (minimum 50k, up to 200k, 2x multiplier).

**Out of memory**: Reduce `--{plan,notes}-batch-size` and `--{plan,notes}-concurrency`.

**Empty speculative notes**: Fixed in current version. Type prefix stripping handles LLM output variations in compact array format.

## Design Rationale

**Per-stream parallelism**: Each stream's LLM call sees only its own text slice, enforcing the "Blind Start" rule and reducing per-call token count to ~1/3. This mirrors the inference protocol where streams decode independently.

**Snapshot 0 materialization**: Training on the materialized plan-contract snapshot teaches the model what snapshot 0 should look like, completing the planner pretraining objective (Stage 0 of the curriculum).

**Two tokenizers**: Preflight uses `cl100k_base` for fast approximate filtering. Collation uses the exact `gpt-oss-20b` tokenizer for training fidelity.

**Compact arrays**: The protobuf-style wire format materially reduces token count relative to verbose JSON and keeps notes-generation payloads smaller.

**Augmentation over generation**: Perturbing `lag_delta` and `note_cadence_M` creates bus timing diversity without additional LLM cost. The model learns robustness to different Dynamic Notes Bus configurations.
