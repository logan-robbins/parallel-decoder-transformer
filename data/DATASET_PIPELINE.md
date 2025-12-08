# Dataset Generation Pipeline

Production pipeline for generating training-ready datasets for the Parallel Decoder Transformer (PDT), including structured 3-stream plans and true/speculative notes with the ENT/FACT/COVERAGE schema.

## Pre-Generated Datasets

Pre-generated datasets from the production pipeline are available for download:

**https://storage.googleapis.com/parallel-decoder-transformer/data/archives/**

### Quick Download

Skip the full pipeline and download ready-to-use training data:

```bash
# Create directories
mkdir -p data/processed/pdt_10k_gpt41

# Training split (2.7GB compressed, ~47GB uncompressed)
wget https://storage.googleapis.com/parallel-decoder-transformer/data/archives/pdt_10k_gpt41_jsonl_train.tar.gz
tar -xzf pdt_10k_gpt41_jsonl_train.tar.gz -C data/processed/

# Evaluation splits (647MB compressed, ~5GB uncompressed)
wget https://storage.googleapis.com/parallel-decoder-transformer/data/archives/pdt_10k_gpt41_jsonl_eval.tar.gz
tar -xzf pdt_10k_gpt41_jsonl_eval.tar.gz -C data/processed/

# Verify datasets
ls data/processed/pdt_10k_gpt41/
# Should show: kd_train.jsonl, kd_validation.jsonl, kd_test.jsonl
```

**Parquet format (for inspection/analysis):**
```bash
wget https://storage.googleapis.com/parallel-decoder-transformer/data/archives/pdt_10k_gpt41_parquet.tar.gz
tar -xzf pdt_10k_gpt41_parquet.tar.gz -C data/datasets/
```

**Plans only (for understanding structure):**
```bash
wget https://storage.googleapis.com/parallel-decoder-transformer/data/archives/pdt_10k_plans.tar.gz
tar -xzf pdt_10k_plans.tar.gz -C data/prep/
```

## Pipeline Stages

1. **Preflight** — Filter and validate source documents (Wikipedia articles)
2. **Plan Generation** — Create 3-stream decomposition plans using OpenAI Structured Outputs
3. **Notes Generation** — Generate true notes (N) and speculative notes (N̂) with controlled hallucinations
4. **Collation** — Export to Arrow/Parquet format with train/validation/test splits
5. **KD Export** — Transform to split-specific JSONL files for training

## Final Outputs

After completing all 5 stages, you will have:

**Parquet splits** (Stage 4) in `data/datasets/pdt_10k_gpt41/`:
- `train.parquet` - 80% of augmented samples
- `validation.parquet` - 10% of augmented samples
- `test.parquet` - 10% of augmented samples
- `manifest.json` - Dataset metadata

**Training-ready JSONL files** (Stage 5) in `data/processed/pdt_10k_gpt41/`:
- `kd_train.jsonl` - Training split (one record per stream)
- `kd_validation.jsonl` - Validation split (one record per stream)
- `kd_test.jsonl` - Test split (one record per stream)

Each JSONL file contains only its designated split to prevent data leakage during training.

## Environment Setup

### Python Environment

```bash
# Create venv and install dependencies
uv venv .venv --python 3.12
uv sync
```

### API Keys

Create `.env` in repository root:

```bash
OPENAI_API_KEY=your_key_here
OPENAI_ORG_ID=your_org_id  # Optional
```

### Data Sources

- **Math**: Reasoning Gym package (installed automatically)
- **QA**: SQuAD via `datasets` package (installed automatically)
- **Survey**: Wikipedia manifest at `data/manifests/wikipedia_20231101_en_train.json`

## Configuration

### LLM Config Files

Production configs are located in `configs/dataset/`. Use these for Stages 2-3 (plan/notes generation):

**Production config** (GPT-4.1):
```yaml
# configs/dataset/notes_gpt41_production.yaml
llm:
  backend: openai
  openai:
    model: gpt-4.1
    service_tier: default
    client_timeout: 900.0
```

**Test config** (smaller runs):
```yaml
# configs/dataset/notes_gpt41_test.yaml
llm:
  backend: openai
  openai:
    model: gpt-4o-mini
    service_tier: flex
    client_timeout: 600.0
```

**Stage 1 (Preflight)** uses CLI flags `--wiki-classifier-model` and `--wiki-classifier-service-tier` instead of config files.

### Split Models (Plan vs Notes)

To use different models for plan and notes generation, specify `notes_llm` in your config:

```yaml
# Example: GPT-4.1 for plans, GPT-4o-mini for notes
llm:
  backend: openai
  openai:
    model: gpt-4.1
    service_tier: default
    client_timeout: 900.0

notes_llm:
  backend: openai
  openai:
    model: gpt-4o-mini
    service_tier: flex
    client_timeout: 600.0
```

This routes Stage 2 (plan generation) through the primary `llm` model and Stage 3 (notes generation) through `notes_llm`.

## Stage 1: Preflight Validation

Filters Wikipedia articles for structural suitability using deterministic checks + LLM classification.

### Command

```bash
uv run python scripts/preflight_plans.py \
  --survey 1000 \
  --output-dir data/prep/preflight/pdt_10k_gpt41 \
  --wiki-classifier-model gpt-4.1 \
  --wiki-classifier-service-tier default \
  --wiki-classifier-batch-size 100 \
  --wiki-classifier-concurrency 8 \
  --wiki-min-article-chars 10000 \
  --wiki-max-article-chars 30000
```

### Key Parameters

- `--survey N`: Number of Wikipedia articles to validate
- `--wiki-min-article-chars`: Minimum article length in characters (default: 10,000)
- `--wiki-max-article-chars`: Maximum article length in characters (default: 30,000)
- `--wiki-classifier-model`: LLM model for classification (default: gpt-5.1)
- `--wiki-classifier-service-tier`: OpenAI service tier (default: flex)
- `--wiki-classifier-reasoning-effort`: Reasoning effort level - `low` (fast, cheap), `medium` (balanced), `high` (thorough) (default: low)
- `--wiki-classifier-batch-size`: Articles per batch (default: 100)
- `--wiki-classifier-concurrency`: Max concurrent requests (default: 8)
- `--wiki-offset`: Skip first N articles in manifest (default: 0) - use for resuming from previous runs
- `--output-dir`: Where to write accepted/rejected manifests

### Length Filters

Articles are filtered using a **two-stage strategy** for performance:

**Stage 1: Character filters (fast pre-screen):**
- **Character range:** 10,000 - 30,000 chars (configurable via `--wiki-min-article-chars` / `--wiki-max-article-chars`)
- ⚡ **Purpose**: Quickly eliminate obviously unsuitable articles without tokenization cost
- 🎯 **~95% of rejections** happen here, avoiding expensive tokenization

**Stage 2: Token filters (precise validation):**
- **Token range:** 200 - 25,000 tokens (configurable via `--wiki-min-article-tokens` / `--wiki-max-article-tokens`)
- **Token ceiling:** 100,000 tokens total (configurable via `--wiki-max-total-tokens`)
- **Tokenizer:** `cl100k_base` (OpenAI GPT-4 tokenizer, ~4 chars/token)
- 🐌 **Purpose**: Accurate filtering for articles that passed character check

**Default values ensure articles are:**
- Long enough for 3-stream decomposition (10k chars ≈ 2,500 tokens)
- Short enough to avoid truncation during collation (30k chars ≈ 7,500 tokens)
- Suitable for rich, multi-faceted content

**Why both?** Character filtering is ~1000x faster than tokenization. By checking characters first, we avoid tokenizing 95%+ of rejected articles, dramatically speeding up preflight.

**Adjusting filters:**

*Most users should only adjust character filters* - the token filters are set conservatively and rarely need changes.

Example - to accept shorter articles:
```bash
--wiki-min-article-chars 5000 \
--wiki-max-article-chars 40000
```

**All available length filters:**
- `--wiki-min-article-chars` (default: 10,000) - Adjust this to change minimum article length
- `--wiki-max-article-chars` (default: 30,000) - Adjust this to change maximum article length  
- `--wiki-min-article-tokens` (default: 200) - Rarely needs adjustment
- `--wiki-max-article-tokens` (default: 25,000) - Rarely needs adjustment
- `--wiki-max-total-tokens` (default: 100,000) - Budget for plan generation prompt + article

**Tip:** Keep character limits ≈4x the token limits (rough chars-per-token ratio). Token filters catch edge cases where character filters were too permissive.

### Outputs

- `preflight_accepted.jsonl`: Validated candidates (feed to plan generation)
- `preflight_rejected.jsonl`: Filtered candidates with rejection reasons
- `preflight_report.json`: Summary metrics

**Tip:** Check `preflight_report.json` to see rejection reasons:
```bash
jq '.rejected_by_reason' data/prep/preflight/pdt_10k_gpt41/preflight_report.json
```

Common rejection reasons:
- `article_chars_below_min` / `article_chars_above_max` - Character filter caught them (adjust `--wiki-min/max-article-chars`)
- `article_too_short` / `article_tokens>N` - Token filter caught them (usually means character filter needs tightening)
- Other reasons (`disambiguation_filter`, `list_like`, `stub`, etc.) - LLM classifier or heuristic rejections

If 95%+ of rejections are `article_chars_below_min`, your character thresholds are working correctly as a fast pre-filter.

### Performance

**With GPT-4.1:**
- ~100 articles/minute with batch_size=100, concurrency=8
- Cost: ~$0.001 per article classification
- 1000 articles → ~10-15 minutes

## Stage 2: Plan Generation

Generates 3-stream decomposition plans using OpenAI Structured Outputs with strict schema validation.

### Command

```bash
uv run python scripts/run_dataset_pipeline.py \
  --config configs/dataset/notes_gpt41_production.yaml \
  --plan-dir data/prep/plans/pdt_10k_gpt41 \
  --notes-dir data/prep/notes/pdt_10k_gpt41 \
  --dataset-dir data/datasets/pdt_10k_gpt41 \
  --survey 1000 \
  --preflight-manifest data/prep/preflight/pdt_10k_gpt41/preflight_accepted.jsonl \
  --skip-notes \
  --skip-collate \
  --skip-kd-export \
  --plan-batch-size 18 \
  --plan-concurrency 12 \
  --log-level INFO
```

### Key Parameters

- `--plan-batch-size`: Plans per async batch (default: 24, recommend: 18)
- `--plan-concurrency`: Max concurrent requests (default: 8, recommend: 12)
- `--preflight-manifest`: Path to validated candidates from Stage 1
- `--qa/--math/--survey`: Counts per domain

### Schema Enforcement

Plans use `json_schema` mode with strict validation:
- Exactly 3 streams required
- Each stream has ≥2 `notes_contract` bullets
- Non-empty `section_contract` with source text slicing
- Malformed responses rejected by OpenAI before returning

### Performance

**With GPT-4.1:**
- ~60 plans/minute with batch_size=18, concurrency=12
- ~6-10 seconds per plan
- 100% schema compliance (validated at API level)
- 1000 plans → ~17 minutes

### Outputs

Plans written to `{plan-dir}/{domain}/{sample_id}.json`:

```json
{
  "sample_id": "survey_7825850_e13d6bbe",
  "domain": "survey",
  "streams": [
    {
      "stream_id": "stream_1",
      "header": "Early History",
      "summary": "Covers Maienfeld's location...",
      "entities": ["ENT:name=Maienfeld,type=municipality", ...],
      "constraints": ["FACT:subj=Maienfeld,pred=located_in,obj=Graubünden"],
      "section_contract": {
        "type": "source_slice",
        "start_idx": 0,
        "end_idx": 500
      },
      "notes_contract": [
        "Must identify Maienfeld's canton (Graubünden)",
        "Must mention strategic Alpine location"
      ]
    }
    // ... 2 more streams
  ],
  "reasoning": ["Balanced text distribution across 3 streams..."]
}
```

## Stage 3: Notes Generation

Generates true notes (Teacher) and speculative notes (Student) with controlled hallucinations.

### Command

```bash
nohup uv run python scripts/run_dataset_pipeline.py \
  --config configs/dataset/notes_gpt41_production.yaml \
  --plan-dir data/prep/plans/pdt_10k_gpt41 \
  --notes-dir data/prep/notes/pdt_10k_gpt41 \
  --dataset-dir data/datasets/pdt_10k_gpt41 \
  --skip-plans \
  --skip-collate \
  --skip-kd-export \
  --notes-batch-size 18 \
  --notes-concurrency 12 \
  --log-level INFO > logs/notes_generation_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Key Parameters

- `--notes-batch-size`: Plans per batch (default: 24, recommend: 18)
- `--notes-concurrency`: Max concurrent requests (default: 10, recommend: 12)
- `--notes-limit N`: Cap number of plans to process (for testing)
- `--spec-variants`: Speculative variants per plan (default: 3)
- `--no-notes-resume`: Force regeneration (disable resume)

### Two-Pass Process

**Pass 1 (Teacher):** Generate true notes from source text
- 1 request per plan
- `json_object` mode (allows compact arrays)
- Latency: 20-50 seconds with reasoning_effort:low

**Pass 2 (Student):** Generate speculative notes with hallucinations
- 3 requests per plan (variants)
- Noise: paraphrase, drop facts, hallucinate names/dates
- Latency: 10-25 seconds per variant

### Compact Array Format

LLM generates notes as compact tuples for 80% token savings:

```json
{
  "notes": [
    ["ENT", "E1", "Maienfeld", "municipality", true, []],
    ["FACT", "E1", "located_in_canton", "Graubünden", 1.0, [0, 50, "text span"]],
    ["COVERAGE", "stream_1_item_1", "complete"]
  ]
}
```

**Critical Implementation Detail:** The pipeline strips type prefixes (`["ENT", ...]` → `[...]`) during normalization to handle LLM output variations.

### Hallucination Examples

Speculative notes contain **confident lies** to train Speculative Invariance:

- Geographic: "canton of **Zurich**" (truth: Graubünden)
- Names: "Ahmad Shah **Dostum**" (truth: Massoud)
- Dates: "October **30**, 1832" (truth: October 31)
- Titles: "Ministry of Defense **and Space**" (truth: no "and Space")

These hallucinations are captured in **both** text (`z_hat`) **and** structured ENT/FACT entries.

### Performance

**With GPT-4.1:**
- ~6 notes/minute (including 3 speculative variants)
- ~10 seconds per plan total
- 1000 plans → ~2.75 hours

### Outputs

Notes written to `{notes-dir}/{domain}/{sample_id}.json`:

```json
{
  "sample_id": "survey_7825850_e13d6bbe",
  "domain": "survey",
  "true_notes": [
    {
      "stream_id": "stream_1",
      "ENT": [{"id": "E1", "name": "Maienfeld", "type": "municipality", ...}],
      "FACT": [{"subj_id": "E1", "predicate": "located_in", "object": "Graubünden", ...}],
      "COVERAGE": [{"plan_item_id": "item_1", "status": "complete"}]
    }
    // ... 2 more streams
  ],
  "speculative_notes": [
    {
      "variant_id": "variant_0",
      "z_hat": "Maienfeld is a municipality in... canton of Zurich...",  // Hallucination
      "notes": [
        {
          "stream_id": "stream_1",
          "ENT": [{"id": "E3", "name": "Zurich", "type": "canton", ...}],  // Captured!
          "FACT": [...],
          "COVERAGE": [...]
        }
      ],
      "noise_config": {"paraphrase_ratio": 0.15, "hallucination_ratio": 0.05}
    }
    // ... 2 more variants
  ],
  "z_n": "Maienfeld is a municipality in... Graubünden...",  // Truth
  "versioned_notes": [...],  // Snapshots for dynamic notes bus
  "rollback": {"triggered": false}
}
```

## Stage 4: Collation

Tokenizes notes and exports to Arrow/Parquet format with train/validation/test splits.

### Command

```bash
uv run python scripts/run_dataset_pipeline.py \
  --notes-dir data/prep/notes/pdt_10k_gpt41 \
  --dataset-dir data/datasets/pdt_10k_gpt41 \
  --skip-plans \
  --skip-notes \
  --skip-kd-export \
  --augment 2 \
  --max-len 8192
```

### Key Parameters

- `--augment N`: Augmented copies per sample (default: 2)
  - Creates N additional variants with perturbed timing parameters
  - Each augmented copy varies `lag_delta` by ±1 and `note_cadence_M` by ±2
  - Multiplies dataset size by (1 + N) without additional LLM costs
  - Improves model robustness to different dynamic notes bus configurations

- `--max-len`: Tokenizer truncation/padding length (default: 2048, **recommended: 8192**)
  - Must match your model's training context window
  - **GPT-OSS-20B:** Supports up to 128k tokens, but 8192 is optimal for:
    - Preflight-filtered Wikipedia articles (10k-30k chars ≈ 2k-7k tokens)
    - Memory efficiency with batch training on 8x H100 (80 GB SXM5) GPUs
    - 90%+ context preservation for long-form articles
  - Lower values (2048, 4096) truncate content aggressively
  - All sequences are padded/truncated to exactly `max_len` tokens

### Tokenizer

**Critical:** Collation uses the **gpt-oss-20b tokenizer** (199,998 vocab), which differs from preflight's `cl100k_base` (100,277 vocab):
- GPT-OSS tokenizer is ~17% more efficient (fewer tokens for same text)
- This is **correct behavior** — collation must use the model's native tokenizer
- Preflight uses `cl100k_base` for fast approximate filtering only

### Outputs

Parquet files in `{dataset-dir}/`:
- `train.parquet` (80% of data)
- `validation.parquet` (10%)
- `test.parquet` (10%)
- `manifest.json` (counts and config)

Columns include:
- `sample_id`, `domain`
- `x_text`, `plan_text`, `z_n`, `z_hat`
- `notes_true`, `notes_speculative`, `notes_versioned`
- `x_tokens`, `plan_tokens`, `z_n_tokens`, `z_hat_tokens` (tokenized)
- `lag_delta`, `note_cadence_M` (timing parameters)

### Example Output

For 1,000 notes with `--augment 2`:
- **Total records:** 3,000 (1,000 original + 2,000 augmented)
- **Train split:** 2,400 records (80%) → saved to `train.parquet`
- **Validation split:** 300 records (10%) → saved to `validation.parquet`
- **Test split:** 300 records (10%) → saved to `test.parquet`

All token sequences are exactly `max_len` tokens (padded or truncated).

**After KD Export (Stage 5)**, these become:
- **Train:** 2,400 records × 3 streams = 7,200 JSONL records in `kd_train.jsonl`
- **Validation:** 300 records × 3 streams = 900 JSONL records in `kd_validation.jsonl`
- **Test:** 300 records × 3 streams = 900 JSONL records in `kd_test.jsonl`

## Stage 5: KD JSONL Export

Transforms Parquet to JSONL format consumed by `KDJsonlDataset` during training.

### Command

```bash
uv run python scripts/run_dataset_pipeline.py \
  --dataset-dir data/datasets/pdt_10k_gpt41 \
  --processed-dir data/processed/pdt_10k_gpt41 \
  --notes-dim 2048 \
  --kd-splits train validation test \
  --skip-plans \
  --skip-notes \
  --skip-collate
```

### Key Parameters

- `--processed-dir`: Output directory for JSONL files
- `--notes-dim`: Embedding dimension for notes vectors (default: 2048)
- `--kd-splits`: Which splits to export (default: all)

### Outputs

**Split-specific JSONL files** in `{processed-dir}/`:
- `kd_train.jsonl` - Training split
- `kd_validation.jsonl` - Validation split
- `kd_test.jsonl` - Test split

Each file contains one record per stream:

```json
{
  "example_id": "survey_7825850_e13d6bbe_stream_1",
  "sample_id": "survey_7825850_e13d6bbe",
  "stream_id": "stream_1",
  "domain": "survey",
  "split": "train",
  "student_ids": [101, 7865, ...],
  "student_labels": [7865, 2548, ...],
  "planner_ids": [101, 2059, ...],
  "notes_student": [[0.1, 0.2, ...], [0.3, 0.1, ...], [0.2, 0.4, ...]],
  "notes_teacher": [[0.12, 0.19, ...], [0.31, 0.09, ...], [0.21, 0.39, ...]],
  "metadata": {
    "document_text": "Maienfeld is a municipality...",
    "teacher_plan": {...},
    "teacher_notes": {...},
    "notes_versioned": [...],
    "sectional_independence": true
  },
  "true_notes": [...]
}
```

## Production Workflow

### Full 1000-Example Run

```bash
# Stage 1: Preflight
uv run python scripts/preflight_plans.py \
  --survey 1500 \
  --output-dir data/prep/preflight/pdt_10k_gpt41 \
  --wiki-classifier-model gpt-4.1 \
  --wiki-min-article-chars 10000 \
  --wiki-max-article-chars 30000

# Stage 2: Plans
uv run python scripts/run_dataset_pipeline.py \
  --config configs/dataset/notes_gpt41_production.yaml \
  --plan-dir data/prep/plans/pdt_10k_gpt41 \
  --notes-dir data/prep/notes/pdt_10k_gpt41 \
  --dataset-dir data/datasets/pdt_10k_gpt41 \
  --survey 1000 \
  --preflight-manifest data/prep/preflight/pdt_10k_gpt41/preflight_accepted.jsonl \
  --skip-notes --skip-collate --skip-kd-export \
  --plan-batch-size 18 --plan-concurrency 12

# Stage 3: Notes (background job)
nohup uv run python scripts/run_dataset_pipeline.py \
  --config configs/dataset/notes_gpt41_production.yaml \
  --plan-dir data/prep/plans/pdt_10k_gpt41 \
  --notes-dir data/prep/notes/pdt_10k_gpt41 \
  --dataset-dir data/datasets/pdt_10k_gpt41 \
  --skip-plans --skip-collate --skip-kd-export \
  --notes-batch-size 18 --notes-concurrency 12 \
  --log-level INFO > logs/notes_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Monitor: tail -f logs/notes_*.log | grep -E "(succeeded|ERROR)"

# Stage 4: Collation
uv run python scripts/run_dataset_pipeline.py \
  --notes-dir data/prep/notes/pdt_10k_gpt41 \
  --dataset-dir data/datasets/pdt_10k_gpt41 \
  --skip-plans --skip-notes --skip-kd-export \
  --augment 2 --max-len 8192

# Stage 5: KD Export
uv run python scripts/run_dataset_pipeline.py \
  --dataset-dir data/datasets/pdt_10k_gpt41 \
  --processed-dir data/processed/pdt_10k_gpt41 \
  --skip-plans --skip-notes --skip-collate
```

**Outputs**:
- `data/processed/pdt_10k_gpt41/kd_train.jsonl` (2,400 records = 800 samples × 3 streams)
- `data/processed/pdt_10k_gpt41/kd_validation.jsonl` (300 records = 100 samples × 3 streams)
- `data/processed/pdt_10k_gpt41/kd_test.jsonl` (300 records = 100 samples × 3 streams)

**Note**: Record counts = (Parquet rows in split) × (3 streams per document). Each stream gets its own JSONL record.

### Cost & Time Estimates

**1000 Wikipedia articles with GPT-4.1**:

| Stage | Time | Cost (approx) |
|-------|------|---------------|
| Preflight (1500 → 1000) | ~15 min | $2 |
| Plans | ~17 min | $20 |
| Notes (1000 × 4 calls) | ~2.75 hrs | $180 |
| Collation | ~2 min | $0 |
| KD Export | ~1 min | $0 |
| **Total** | **~3 hours** | **~$200** |

## Resume Semantics

### Stage 1: Preflight

**Manual resume** using `--wiki-offset`:
- Preflight always overwrites output files (no automatic resume)
- To continue from a previous run, find your last processed article index:
  ```bash
  jq -r '.source_metadata.dataset_index' data/prep/preflight/pdt_10k_gpt41/preflight_accepted.jsonl | sort -n | tail -1
  # Example output: 199957
  ```
- Resume by setting `--wiki-offset` to start from the next article:
  ```bash
  --wiki-offset 200000
  ```
- Use a different `--output-dir` to avoid overwriting previous results, or merge manually afterward

### Stages 2-5: Automatic Resume

- **Plans:** Skip if `{plan-dir}/{domain}/{sample_id}.json` exists
- **Notes:** Skip if `{notes-dir}/{domain}/{sample_id}.json` exists
- **Collation:** Always regenerates from current notes (idempotent)
- **KD Export:** Resumes by skipping existing `example_id` entries in split files (append mode)

**Force regeneration:**
- Plans: `--no-plan-resume`
- Notes: `--no-notes-resume`
- KD Export: Delete existing split files or use `--force` flag (if implemented)

## Monitoring & Troubleshooting

### Check Progress

```bash
# Stage 1: Preflight progress
wc -l data/prep/preflight/pdt_10k_gpt41/preflight_accepted.jsonl
jq -r '.source_metadata.dataset_index' data/prep/preflight/pdt_10k_gpt41/preflight_accepted.jsonl | sort -n | tail -1

# Count generated artifacts by stage (already updated above)
wc -l data/processed/pdt_10k/kd_*.jsonl                    # Stage 5: Record counts

# Monitor notes generation log
tail -f logs/notes_*.log | grep -E "(succeeded|Pass|Completed)"

# Check for errors in any stage
grep -i "error\|failed" logs/notes_*.log | grep -v "error-prone\|\"z_n\"" | tail -20
```

### Resuming Preflight Runs

If your preflight run is interrupted or you need more articles:

```bash
# 1. Check your last processed article index
LAST_INDEX=$(jq -r '.source_metadata.dataset_index' \
  data/prep/preflight/pdt_10k_gpt41/preflight_accepted.jsonl | sort -n | tail -1)
echo "Last processed index: $LAST_INDEX"

# 2. Resume from next article (e.g., if last was 199957, start at 200000)
uv run python scripts/preflight_plans.py \
  --survey 3000 \
  --output-dir data/prep/preflight/pdt_10k_gpt41_continuation \
  --wiki-classifier-model gpt-4.1 \
  --wiki-classifier-service-tier default \
  --wiki-classifier-batch-size 100 \
  --wiki-classifier-concurrency 8 \
  --wiki-offset 200000

# 3. Merge results if needed
cat data/prep/preflight/pdt_10k_gpt41/preflight_accepted.jsonl \
    data/prep/preflight/pdt_10k_gpt41_continuation/preflight_accepted.jsonl \
    > data/prep/preflight/pdt_10k_gpt41_merged/preflight_accepted.jsonl
```

### Failure Logs

Transient failures (timeouts, HTTP errors) are logged but automatically retried:
- `data/prep/plans/{run_id}/plan_generation_failures_*.jsonl`
- `data/prep/notes/{run_id}/notes_generation_failures_*.jsonl`

Permanent failures (empty): 0 expected with current implementation.

### Common Issues

**Empty notes in speculative variants:**
- **Fixed in current version.** Type prefix stripping handles LLM variations.

**Timeouts/HTTP 502:**
- Expected with long-running jobs. Retry logic handles automatically.

**Out of memory:**
- Reduce `--{plan,notes}-batch-size` and `--{plan,notes}-concurrency`

## Validation

### Semantic Validation Script

**Validate Notes Files:**
```bash
python3 << 'EOF'
import json, glob, random

files = sorted(glob.glob("data/prep/notes/pdt_10k_gpt41/survey/*.json"))
sample = random.choice(files)

with open(sample) as f:
    data = json.load(f)

# Check speculative notes have structured hallucinations
for variant in data["speculative_notes"][:2]:
    notes = variant["notes"][0] if variant["notes"] else {}
    ent_count = len(notes.get("ENT", []))
    fact_count = len(notes.get("FACT", []))
    print(f"{variant['variant_id']}: {ent_count} ENT, {fact_count} FACT")
    if ent_count == 0:
        print("  ❌ SIGNAL DROPOUT!")
    else:
        print("  ✅ OK")
EOF
```

**Validate KD JSONL Splits:**
```bash
# Count records per split
wc -l data/processed/pdt_10k_gpt41/kd_*.jsonl

# Expected output (for 1000 samples with 3 streams each, 2x augmentation):
#   2400 data/processed/pdt_10k_gpt41/kd_train.jsonl
#    300 data/processed/pdt_10k_gpt41/kd_validation.jsonl
#    300 data/processed/pdt_10k_gpt41/kd_test.jsonl
#   3000 total

# Verify no overlap between splits
python3 << 'EOF'
import json

def load_ids(path):
    ids = set()
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            ids.add(data["sample_id"])
    return ids

train = load_ids("data/processed/pdt_10k_gpt41/kd_train.jsonl")
val = load_ids("data/processed/pdt_10k_gpt41/kd_validation.jsonl")
test = load_ids("data/processed/pdt_10k_gpt41/kd_test.jsonl")

print(f"Train samples: {len(train)}")
print(f"Validation samples: {len(val)}")
print(f"Test samples: {len(test)}")
print(f"Overlap train/val: {len(train & val)} (should be 0)")
print(f"Overlap train/test: {len(train & test)} (should be 0)")
print(f"Overlap val/test: {len(val & test)} (should be 0)")
EOF
```

Expected output: All variants show non-zero ENT/FACT counts, and zero overlap between splits.

## Architecture Notes

### Speculative Invariance Training

The pipeline generates **adversarial hallucinations** to train the Student model:

1. **Teacher** generates true notes ($N$) from source text → produces $Z_N$
2. **Student** receives noisy notes ($\widehat{N}$) → must produce $Z \approx Z_N$
3. **Training objective:** $p(Z \mid \widehat{N}) \approx p(Z \mid N)$

**Critical requirement:** Hallucinations in text (`z_hat`) must be captured in structured notes (`ENT`/`FACT`). Otherwise, the Dynamic Notes Bus receives zero vectors and the Student learns to ignore it.

**Current implementation:** ✅ Hallucinations are captured in both text and structure.

### Compact Format Benefits

- **Token savings:** ~80% reduction vs verbose object format
- **Cost impact:** $200 → $40 for 1000 notes
- **Compatibility:** Automatic rehydration to verbose format for storage

### Concurrency Tuning

Recommended settings (Apple Silicon M4, 128GB RAM):
- **Plan generation:** batch_size=18, concurrency=12
- **Notes generation:** batch_size=18, concurrency=12

Higher concurrency = faster but risks rate limiting. Adjust based on OpenAI tier limits.

### Tokenizer Architecture

The pipeline uses **two different tokenizers** at different stages:

**Stage 1 (Preflight):**
- **Tokenizer:** `cl100k_base` (OpenAI GPT-4 tokenizer)
- **Vocab size:** 100,277 tokens
- **Purpose:** Fast approximate filtering without loading 27MB GPT-OSS tokenizer
- **Efficiency:** ~4 characters per token

**Stages 4-5 (Collation & Training):**
- **Tokenizer:** `gpt-oss-20b` native tokenizer
- **Vocab size:** 199,998 tokens
- **Purpose:** Exact tokenization matching model's training data
- **Efficiency:** ~17% more efficient than cl100k_base (fewer tokens for same text)

This is **correct architecture** — preflight provides fast filtering estimates, while collation uses the exact tokenizer required for training.
