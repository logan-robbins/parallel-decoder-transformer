# Parallel Decoder Transformer — Parallel Inference + Training

This repository provides a paper‑aligned Parallel Decoder Transformer runtime (Dynamic Notes Bus + Shared Notes Cross‑Attention) over a fine-tuned GPT‑OSS-20B trunk. Includes a production dataset pipeline and full training infrastructure.

Most people parallelize LLMs by wrapping them in Python scripts (Skeleton of Thought). I moved the parallelization inside the Transformer architecture. I built a 'Dynamic Notes Bus' that lets attention heads in different streams talk to each other via compressed embeddings. It required hacking the KV cache management and deriving a new loss function to ensure the parallel streams didn't hallucinate contradictions. It's effectively System 2 thinking—planning and verifying—baked into the forward pass.

```
                                      ^
                                      |
                         [ Layer N+1 (Frozen Trunk) ]
                                      |
      +-------------------------------+-------------------------------+
      |                                                               |
      |   (3) Final Output of Layer N                                 |
      |                                                               |
      +<--------------------------(+)                                 |
      |                            ^                                  |
      |                            |                                  |
      |                  [ Stream Adapter (MLP) ]                     |
      |                            ^                                  |
      |                            |                                  |
      |   (2) Intermediate         +--------(Gate_Stream)-------------+
      |                                                               |
      +<--------------------------(+)                                 |
      |                            ^                                  |
      |                            |                                  |
      |                  [ SNC (Shared Notes Cross-Attn) ]            |
      |                  (Queries Bus Snapshots)                      |
      |                            ^                                  |
      |                            |                                  |
      |   (1) Original Output      +--------(Gate_SNC)----------------+
      |                                                               |
      |                  [ Layer N (Frozen Trunk) ]                   |
      |                  (Standard Attn + MLP Block)                  |
      |                                                               |
      +-------------------------------+-------------------------------+
                                      |
                                      ^
                         [ Layer N-1 (Frozen Trunk) ]
```

## Documentation

- Dataset pipeline, data generation, and collation: see `docs/guides/DATASET_PIPELINE.md`.
- Training pipeline, configuration, and execution: see `docs/guides/TRAINING.md`.

Highlights

- Stream contracts with independence: plans include per‑stream `section_contract` (machine‑readable slice ownership) and `notes_contract` (required ENT/FACT/COVERAGE bullets). A plan‑level `sectional_independence` flag enables strictly independent section decoding from the initial notes snapshot.
- Latent planner slots are canonical: `planner_head.vocab_size == plan_vocab_size`, fixed `planner_head.num_slots` ids index the shared `plan_embedding`, and the coverage/catalog path hashes into that same latent vocabulary.
- Plan‑derived snapshot 0 at t=0: the notes pipeline writes `versioned_notes` with snapshot 0 = `plan_contract`, and training seeds the Dynamic Notes Bus from that snapshot at the first decoding step (t=0). The trainer preserves this snapshot through the first stride (B) to keep the contract stable.
- Sectional LM supervision: the KD collator materializes `labels_mask` spans per stream when `sectional_independence` is set, and the trainer applies that mask before LM CE so each stream only optimizes its own section.

## Training Approach

- **Student Model**: GPT-OSS-20B (fine-tuned on 8x B200 180GB)
- **Teacher Model**: API-based (GPT-4.1 for production datasets)
- **Dataset**: 10K Wikipedia articles processed through 5-stage pipeline
- **Training Duration**: 50,000 steps (~30 hours on 8x B200)
- **Hardware**: NVIDIA B200 GPUs with 180GB VRAM each
- **Curriculum**: 4-stage parameter-efficient training (trunk frozen)
- **Evaluation Status**: Prior quantitative claims were removed after the coordination rewrite and must be rerun on the current stack

Training uses a parameter-efficient approach where the 20B trunk remains frozen throughout. Only lightweight adapters and auxiliary heads are trained (<5% of total parameters). Stage 4 (trunk unfreezing) requires >190GB VRAM per GPU and is not supported on B200 hardware.

**See `docs/guides/TRAINING.md` for complete training setup, curriculum stages, loss functions, and production deployment instructions.**

## System Architecture

- **GPT‑OSS-20B trunk** with per‑stream adapters and Shared Notes Cross-Attention (SNC)
- **Prompt-time latent planner** with fixed planner slots over the shared plan vocabulary
- **Dynamic Notes Bus** with lagged, versioned snapshots (Δ, K) and all-to-all broadcast topology
- **Agreement‑gated synchronized continuation**: per-stream readiness is checked at stride boundaries and low-readiness streams are rolled back within horizon L
- **Plan coverage diagnostics** with model-predicted or explicitly supplied latent plan ids
- **Dataset builder** that materializes Parquet splits with prompt/plan/sections/notes for KD training
  - Includes `notes_versioned` and `sectional_independence` for Dynamic Notes Bus seeding
- **End-to-end coverage signals** with normalized ENT/FACT/COVERAGE payloads, stable plan catalogs, and calibrated targets
- **Optional safeguards**: Lipschitz monitoring, flicker diagnostics, spectral‑norm hooks, and stage‑gated regularizers (disabled by default)

## Implementation Details

- **Planner bootstrap**: the prompt bootstrap pass mean-pools prompt hidden states, predicts fixed latent planner slots, embeds those ids through `plan_embedding`, projects them with `plan_notes_proj`, L2-normalizes the pooled result, and pushes that shared plan seed to every stream at t=0. When `--plan-contract` is supplied, the contract also materializes snapshot 0 directly in `versioned_notes`. Contract-derived seeds are ownership priors for stream/plan-item responsibility, not hard token constraints.
- **Coordination is stack-level**: the planner initializes snapshot 0, but ongoing coordination is distributed across SNC bus reads, `notes_head` updates, and agreement/coverage control. The repo does not claim that `planner_head` alone carries the full coordination policy.
- **Bootstrap**: each lane still pushes an initial speculation snapshot: adapted hidden → SpeculationHead → DNB (stride=1) after planner snapshot 0 is published. `--seed-text*` and `--seed-notes-file` remain available as debug-only controls for counterfactual experiments.
- **Gates and blending**: the active instrumented SNC path uses a single learned residual gate in each instrumented layer. Repo-level CLI/config controls such as logit-blend `alpha` and other inference overrides are for diagnostics and ablations, not the core trained mechanism claim.
- **Cadence**: Deterministic by default; stochastic/adaptive modes and max‑interval available to force timely emissions. Gate annealing reduces influence after volatile steps and recovers on stability.
- **Topology/lag**: All-to-all broadcast—each stream immediately sees the plan-derived snapshot and then consumes lagged snapshots from every other stream as decoding progresses. Reads still respect lag Δ/versioning.
- **Telemetry**: `--stream-jsonl` prints per‑step JSON; the manifest records timings, cadence events, rollbacks, stride commit events, committed/provisional blocks, `planner_slots`, `slot_ids`, plan masks, coverage logits, and integration metadata.

Bus semantics (embeddings-only)

- The DNB carries fixed-dim embedding vectors (`notes_dim`) and SNC consumes them as K/V for cross-attention. We do not pass text across lanes.
- Any “plan” or “notes” tokens we surface are for observability/coverage only; when present they are embedded inside the model and the embeddings are what enter the bus.

## Notes Schema Contract

- All teacher/speculative notes follow `notes_schema_version = 2.0` with three payloads per stream:
  - `ENT = {id, name, aliases[], type, canonical}` with deduped aliases and stable `id` references for FACT tuples.
  - `FACT = {subj_id, predicate, object, certainty ∈ [0,1], evidence_span:{start,end,text}}` where `subj_id` must point at an ENT id for the same document.
  - `COVERAGE = {plan_item_id, status}` with `status ∈ {covered, partial, missing}` and `plan_item_id` matching the exact catalog entry surfaced to the planner.
- Extraction/builder metadata includes `coverage_provenance = {method, schema_version, strength ∈ {confirmed,hint}, confirmed_plan_items}` so downstream ingest can differentiate confirmed supervision from text-derived hints. Entries marked as `hint` are carried through the KD dataset but masked out of coverage loss/metrics.
- Dataset QC rejects any sample where coverage signals omit a plan item or reference a missing catalog entry.

How we demonstrate divergence (no training)

- Supply a short `--plan-contract` or `--plan-text-file` payload so the orchestrator populates the fixed planner slots before decoding. SNC conditions each lane on those latent plan ids immediately, showing parallelism and causal influence without fine‑tuning.

## Artifacts and Datasets

Pre-trained checkpoints, datasets, and training artifacts are available at:

**https://storage.googleapis.com/parallel-decoder-transformer/**

### Quick Downloads

**Pre-trained adapters (final checkpoint, 50k steps):**
```bash
wget https://storage.googleapis.com/parallel-decoder-transformer/checkpoints/gpt-oss-8xH100-50000steps/adapters_step_50000.pt \
  -O experiments/gpt_oss/adapters_step_50000.pt
```

**Pre-generated datasets:**
```bash
# Training dataset (2.7GB compressed)
wget https://storage.googleapis.com/parallel-decoder-transformer/data/archives/pdt_10k_gpt41_jsonl_train.tar.gz
tar -xzf pdt_10k_gpt41_jsonl_train.tar.gz -C data/processed/

# Evaluation dataset (647MB compressed)
wget https://storage.googleapis.com/parallel-decoder-transformer/data/archives/pdt_10k_gpt41_jsonl_eval.tar.gz
tar -xzf pdt_10k_gpt41_jsonl_eval.tar.gz -C data/processed/
```

**See full artifact listing:** https://storage.googleapis.com/parallel-decoder-transformer/UPLOAD_MANIFEST.md

## Inference Quickstart

This quickstart demonstrates parallel inference with a trained model.

**Prerequisites:**
```bash
# Setup environment with uv
uv venv .venv --python 3.12
uv sync

# Download GPT-OSS-20B weights (see "Local Weights Layout" section below)

# Download pre-trained adapters (optional, for using trained model)
wget https://storage.googleapis.com/parallel-decoder-transformer/checkpoints/gpt-oss-8xH100-50000steps/adapters_step_50000.pt \
  -O experiments/gpt_oss/adapters_step_50000.pt
```

**Prepare streams and a latent plan catalog:**

Create `stream_prefixes.json`:
```json
{
  "stream_1": "You are Part 1. Focus only on part 1: ",
  "stream_2": "You are Part 2. Focus only on part 2: ",
  "stream_3": "You are Part 3. Focus only on part 3: "
}
```

Create `plan_texts.json`:
```json
{
  "stream_1": ["early US history", "colonization", "independence", "constitution"],
  "stream_2": ["modern demographics", "economy", "federal government structure"],
  "stream_3": ["culture", "science", "global alliances of the United States"]
}
```

**Run parallel inference:**
```bash
uv run scripts/infer.py --config configs/gpt_oss_transfer.yaml \
  --prompt "Tell me some facts about the US." \
  --stream stream_1 --stream stream_2 --stream stream_3 \
  --stream-prefix-file stream_prefixes.json \
  --plan-text-file plan_texts.json \
  --read-lag-delta 0 --alpha 1 --gate-g 1 --max-new-tokens 512 --verbose \
  --output experiments/infer/manifest.json
```

Flags that matter

- Default behavior uses the model planner to populate the fixed latent plan slots from the prompt.
- `--plan-contract` overrides the model planner with a dataset-style contract and also seeds snapshot 0 directly.
- `--plan-text-file` supplies per-stream plan catalog strings for controlled latent-plan evaluation and coverage alignment.
- `--seed-text-file`, `--seed-text`, and `--seed-notes-file` are debug-only overrides for manual bus perturbations.
- `--read-lag-delta` (Δ) controls snapshot reveal; use 0 for immediate visibility.
- `--alpha` blends attended vs base logits (1 = attended only).
- `--gate-g` scales SNC residual (0 disables cross‑lane influence; 1 maximizes it).
- `--max-new-tokens` (default 512 if not set) limits tokens per stream.
- `--stream-jsonl` emits per‑token JSON; a manifest is written to the `--output` path.
- `--log-margins` records per-token top-2 logit margins for safety diagnostics.
- `--notes-text-file` attaches human/teacher notes for later scoring when analyzing manifests.
- Counterfactual controls: `--cf-swap stream_a:stream_b`, `--cf-shuffle stream`, `--cf-freeze stream`, `--cf-stale stream:delta`, and `--cf-tag label` annotate notes-window interventions directly in telemetry.
- `--cf-ablate stream` zeros the notes window for a stream (or `--cf-ablate all` for every lane) and records an explicit counterfactual tag in the manifest.
- `--memory-report` records per-step device/host memory plus a PDT memory estimate inside the manifest.
- `--sync-profile` synchronises the device each stride to capture explicit sync overhead timings.
- `--cadence-M stream=M` pins deterministic emission cadence per stream (repeatable, use `all=M`) so manifests log the exact cadence used without editing configs.
- `--baseline sequential|medusa|lookahead|eagle` switches to the sequential or token-level baseline wrappers; manifests stay schema-compatible so analyzers can compare PDT vs. Medusa/Lookahead/EAGLE runs directly.
- `--replay-artifact-dir DIR` captures the tensors required for CPU logit replay under `DIR` (combine with `--replay-chunk-size N` to bound per-chunk rows).

**Telemetry (manifest):**

- `timings`: Overall timings, plus `per_token` list with per-step wall-clock seconds
- `streams`: Per-stream text, token ids, latest snapshot version, gate value, and coverage snapshot
- `gate_trace`: Per-step gate samples
- `cadence_events`, `rollbacks`, `plan`: Auxiliary diagnostics as available
- `git_sha` / `git_dirty`: Commit fingerprint of the checkout that produced the manifest
- Planner metadata: `plan_vocab_size`, `plan_hash_buckets`, `plan_hash_salt`, and `planner_slots` so downstream tools can refuse to mix incompatible latent-plan configurations

## CPU Logit Replay

When GPU access is unavailable, the replay harness exercises the full orchestrator using cached tensors captured during a normal decode.

**Capture artifacts during inference:**
```bash
uv run scripts/infer.py --config configs/gpt_oss_transfer.yaml \
  --replay-artifact-dir experiments/replay/demo_run \
  --replay-chunk-size 2048 \
  [... other inference args ...]
```

This emits:
- `manifest.json` - Prompt, config snapshot, tokenizer fingerprint, plan hashing metadata
- Chunked tensors - `planner_logits*.pt`, `agreement-<stream>-*.pt`, `notes-<stream>-*.pt`, etc.
- `plan_catalog.json` - Resolved plan items aligned to coverage logits

**Replay on CPU:**
```bash
uv run scripts/logit_replay.py \
  --artifact experiments/replay/demo_run \
  --stream-jsonl \
  --output experiments/replay/demo_manifest.json
```

The replay CLI instantiates a lightweight `ReplayModel` that feeds tensors into `MultiStreamOrchestrator`, reproducing cadence, gate traces, rollbacks, and coverage telemetry deterministically on CPU. Pass `--include-events` for per-token event logs.

**Quick demo:** A ready-to-run artifact (`sim/plan_decode_demo/`) is checked into the repo for immediate testing.

## Reproduce Figures

Use `scripts/run_benchmark.sh` to rebuild sequential vs. parallel comparison tables:

```bash
bash scripts/run_benchmark.sh
# Edit the script first to set your prompt/config paths
```

This loads PDT once, runs parallel decode, reruns sequential baseline, and invokes `scripts/compare_seq_parallel.py` to emit fidelity/timing deltas. Manifests are written to `experiments/benchmark/`.

**Analysis tools:**
- **Per-manifest metrics:** `uv run scripts/analyze_infer_manifest.py <manifest.json>`
- **Notes scoring:** `uv run scripts/score_infer_notes.py <manifest.json>`
- **Multi-manifest aggregation:** `uv run scripts/summarize_infer_manifests.py experiments/benchmark/*.json`
- **Coverage threshold sweep:** `uv run scripts/coverage_threshold_sweep.py --manifest <manifest.json> --output figures/coverage_pr_curve.png --csv results/coverage_sweep.csv`

All analyzers enforce matching planner hash metadata before proceeding, preventing accidental comparisons between incompatible vocab/salt settings.

Training telemetry

- `experiments/<run>/train_run_stages.json`: stage transitions with durations.
- `experiments/<run>/training_report.json`: summaries of key metrics (e.g., KD/CE ratios, stability).
- `experiments/<run>/agreement_thresholds.json`: ROC points and the calibrated agreement τ used by the trainer.
- `experiments/<run>/coverage_thresholds.json`: ROC points for the coverage head across 19 thresholds (0.05–0.95) plus the F1-optimal operating point. Use `scripts/coverage_threshold_sweep.py` for a full PR-curve analysis on any saved inference manifest.
- `experiments/<run>/train_manifest.json`: includes the selected τ and paths to other artifacts; `adapters.pt` stores adapter/head weights.

Notes scoring

- `scripts/score_infer_notes.py` scores manifests with attached notes (via `--notes-text-file`) for NLI contradiction, redundancy, and the new attribute consistency metric (cross-stream + per-stream/time). Pass `--write-manifest-metrics` to persist the attribute summaries back into the manifest under `metrics.attribute_consistency`.
- When running inference, supply `--notes-text-file` so downstream evaluation has text to compare against per-stream plan items.

## Known Limitations

- **Manifests:** Per-step `||attended − base||` probes recorded only in replay artifacts, not main manifest
- **Memory:** `commit_L`, `stride_B`, `max_snapshots_K` impact memory. Use `--memory-report` to inspect `manifest.memory.max_tracked_bytes`
- **Dataset QC:** See `docs/guides/DATASET_PIPELINE.md` for quality thresholds and validation

## Documentation Index

- **Training pipeline:** `docs/guides/TRAINING.md` - Model configuration, curriculum stages, loss functions, remote deployment
- **Dataset pipeline:** `docs/guides/DATASET_PIPELINE.md` - 5-stage pipeline, LLM configuration, performance tuning
- **Architecture paper:** `docs/arxiv_submission/main.tex` - planner-first latent coordination paper and mathematical foundations
- **Paper concepts summary:** `CONCEPTS.md` - concise summary of the corrected planner-plus-latent-bus concepts and claim boundary
- **Current paper scope:** mechanism-focused planner/bus architecture pass with stack-level attribution; prior quantitative claims were removed after the coordination rewrite and require reevaluation
- **Paper revision checklist:** `UPDATE_LIST.md` - implementation-aligned audit list for revising the arXiv paper
- **Paper compile command:** from `docs/arxiv_submission/`, run `tectonic --keep-logs --keep-intermediates main.tex`
- **Paper upload bundle:** `tar -czf /tmp/pdt-arxiv-replacement.tar.gz -C docs/arxiv_submission main.tex arxiv.sty references.bib sections figures`
- **Canonical paper location:** `docs/arxiv_submission/main.tex` is the source of truth for the paper
- **Markdown paper mirror:** `docs/arxiv_submission/PAPER.md` is a generated mirror of the current LaTeX paper for easier reading and review

## Dataset Generation + Training

### Prerequisites

1. **Python Environment** (using `uv` package manager):
   ```bash
   uv venv .venv --python 3.12
   uv sync
   ```

2. **API Keys**
   Create `.env` in repository root:
   ```bash
   cp env.example .env
   # Edit .env and add: OPENAI_API_KEY=sk-...
   ```

3. **GPT-OSS-20B Weights**
   See "Local Weights Layout" section below for download and placement instructions.

### Dataset Generation

The dataset pipeline generates training-ready JSONL files through 5 stages:
1. **Preflight** - Filter and validate source documents
2. **Plan Generation** - Create 3-stream decomposition plans
3. **Notes Generation** - Generate true/speculative notes with hallucinations
4. **Collation** - Export to Parquet with train/validation/test splits
5. **KD Export** - Transform to split-specific JSONL for training

**Complete pipeline documentation:** See `docs/guides/DATASET_PIPELINE.md` for:
- Stage-by-stage commands and configuration
- LLM model selection (GPT-4.1 production setup)
- Performance tuning (batch sizes, concurrency, time estimates)
- Cost estimates (~$200 for 1000 Wikipedia articles)
- Resume semantics and troubleshooting
- Validation and quality checks

### Training

Training uses pre-generated datasets with a 4-stage parameter-efficient curriculum where the 20B trunk remains frozen.

**Complete training documentation:** See `docs/guides/TRAINING.md` for:
- Prerequisites and environment setup
- Hardware requirements (8x B200 GPUs, 180GB VRAM each)
- Configuration files (production vs development)
- Curriculum stages (4 stages: planner pretrain → stream bootstrap → notes bus enable → rollback training)
- Teacher-note bus mixing begins in Stage 3; Stage 1 learns stream-specific conditioning under teacher supervision rather than direct teacher-bus forcing
- Extended training schedule (50,000 steps, ~30 hours)
- Loss functions and monitoring metrics
- WandB setup and remote deployment
- Stage 4 hardware constraints (>190GB VRAM required)
- Quantitative results for the rewritten coordination stack must be rerun before reporting

## Local Weights Layout (GPT-OSS-20B)

The production config (`configs/gpt_oss_transfer_production.yaml`) references `model.trunk.base_model: "gpt-oss-20b/original"`. Place the model and tokenizer directories under the repository root:

```
gpt-oss-20b/
  original/
    config.json
    generation_config.json
    model.safetensors.index.json
    model-00001-of-00003.safetensors
    model-00002-of-00003.safetensors
    model-00003-of-00003.safetensors
  tokenizer/
    tokenizer.json
    tokenizer.model
    added_tokens.json
    special_tokens_map.json
    tokenizer_config.json
    chat_template.jinja
```

**Alternative paths**: Edit `model.trunk.base_model` in your config YAML to use an absolute path if storing weights elsewhere.

**Download script**: Use `bash scripts/download_gpt_oss_20b.sh` to automatically download and place weights in the correct location.

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

```
MIT License

Copyright (c) 2025 Logan Robbins

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Citation

If you use this work in your research, please cite:

```bibtex
@misc{robbins2025pdt,
  title={Parallel Decoder Transformer: Model-Internal Parallel Decoding with Speculative Invariance via Note Conditioning},
  author={Robbins, Logan},
  year={2025},
  howpublished={\url{https://github.com/logan-robbins/parallel-decoder-transformer}},
  note={Open-source implementation and training framework}
}
```
 
