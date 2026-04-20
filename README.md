# Parallel Decoder Transformer (PDT) -- Qwen3-4B Base Rebuild

A frozen-trunk Qwen3-4B Base decoder augmented with a trainable sidecar
stack that exposes **K coordinated output streams** sharing an internal
latent workspace. The claim this repository defends is *concept-space
co-referencing*: K streams trained to read from and write to a shared
latent workspace so that each stream's trajectory is shaped by awareness
of where siblings are operating in concept-space -- not by text-mediated
orchestration.

**Status:** pre-training rebuild. Infrastructure built and unit-tested;
training runs require GPU time.

**Paper (markdown):** [docs/arxiv_submission/PAPER.md](docs/arxiv_submission/PAPER.md)

## What this repository contains

```
src/pdt/
  config/          # Dataclass schemas + YAML loader
  trunk/           # Frozen Qwen3 adapter + instrumented decoder layer
  sidecar/         # All trainable phi modules (SNC, adapters, heads,
                     plan_embedding, plan_notes_proj)
  runtime/         # Dynamic Notes Bus, window, state, orchestrator,
                     counterfactuals
  training/        # Curriculum controller, loss assembly, trainer
  diagnostics/     # Codebook-utilization diagnostics (Stage 0 gate)
  datasets/        # Retokenize + re-hash pipeline (Parquet -> Qwen3 JSONL)
  cli/             # train / infer / ablate entry points

configs/
  pdt_qwen3_4b.yaml    # Canonical Qwen3-4B + K=3 + V_p=8192 + d_notes=256

scripts/
  download_corpus.sh       # Pull the preserved teacher-output tarball
  retokenize_corpus.py     # Parquet -> Qwen3 JSONL (zero LLM cost)

tests/smoke/pdt_tests/     # Diagnostic build-order smoke tests
```

## Target shape

| Component                      | Value                                         |
| ------------------------------ | --------------------------------------------- |
| Trunk                          | Qwen3-4B Base, frozen (36 layers, d=2560, GQA 32Q/8KV) |
| Streams K                      | 3                                             |
| SNC instrumentation layers     | every 3rd: [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35] |
| Planner slots S                | 16                                            |
| Planner vocabulary V_p         | 8,192 (down from 65,536 in the v1 paper)      |
| Notes dim d_notes              | 256 (down from 2,048 in v1)                   |
| Block size tau                 | 32 tokens                                     |
| Reveal delay Delta             | 1 round                                       |
| Trainable phi parameter budget | ~350M (planner head dominates at ~336M)       |

## Diagnostic build order

Each step has smoke tests under `tests/smoke/pdt_tests/` that gate the
next. All 11 tests currently pass.

1. **Step 1 -- Frozen trunk + stream adapters (K=2, no SNC, no planner).**
   `test_diag_build_step1.py`. Verifies instrumented layers execute in the
   forward graph and that two streams diverge via adapters alone.

2. **Step 2 -- SNC with zero-init gates, no training.**
   `test_diag_build_step2.py`. Verifies SNC runs, closed gate is a no-op,
   notes content actually reaches attention, and the force-gate override
   (Intervention A) collapses SNC to zero.

3. **Step 3 -- Planner + Stage 0 training (gated on codebook utilization).**
   Run `uv run python -m pdt.cli.train --config configs/pdt_qwen3_4b.yaml`.
   Transition to Stage 1 is gated on
   `unique_entries >= 1000` AND `max(slot_entropy_bits) >= 2`.

4. **Step 4 -- Full curriculum (Stages 1 -> 2 -> 3).**
   Automatic via `CurriculumController` once Stage 0 passes the gate.

5. **Step 5 -- Pre-registered ablations (Interventions A, B, C).**
   Run `uv run python -m pdt.cli.ablate --config ... --checkpoint ...`.
   Reports cross-stream cosine distance, ROUGE-L, and the delta vs.
   baseline for gate-zero and norm-scramble.

## Quick start

### Environment

```bash
uv venv .venv --python 3.12
uv sync
```

### Dataset (zero LLM spend)

```bash
# Download the preserved teacher-output Parquet (2.1 GB compressed).
bash scripts/download_corpus.sh

# Retokenize for Qwen3-4B and re-hash planner IDs at V_p=8192 / d_notes=256.
uv run python scripts/retokenize_corpus.py \
  --input-dir data/datasets/pdt_10k_gpt41 \
  --output-dir data/processed/pdt_10k_qwen3_4b \
  --tokenizer Qwen/Qwen3-4B-Base \
  --plan-hash-buckets 8192 \
  --notes-dim 256
```

The download preserves every plan + notes + rollback payload produced by
the original ~\$2k of GPT-5.1 + GPT-4.1 calls. Retokenization is a local
CPU job that takes ~15-30 minutes for the full 10k corpus.

### Training (NVIDIA GPU)

Training is designed for NVIDIA GPUs; the paper's target configuration is
**4-8 x A100-80GB** (comfortable) or 4 x 48GB cards (tight on the planner
head). Do **not** run full training on Apple Silicon / MPS -- MPS bfloat16
matmul accumulators are unstable on M-series for the attention ops used by
Qwen3, and the planner head's ~336M projector is slow under the MPS
backend. The smoke-test pipeline works on MPS under fp32 for development
iteration only.

On a fresh CUDA host:

```bash
# One-time setup
bash scripts/setup_lambda_gpu.sh

# Single-GPU
uv run python -m pdt.cli.train --config configs/pdt_qwen3_4b.yaml

# N-GPU DDP via torchrun
uv run torchrun --nproc_per_node=N -m pdt.cli.train --config configs/pdt_qwen3_4b.yaml
```

Telemetry (codebook stats, per-stage freeze snapshot, checkpoints) lands
in `experiments/qwen3_4b/` by default.

#### Stage 0 gate

Stage 0 (planner pretrain) only advances to Stage 1 once codebook
utilization clears the pre-registered gate:
`unique_entries >= 1000` AND `max(per_slot_entropy_bits) >= 2`. If either
fails after one epoch, add commitment-loss or EMA regularization before
scaling V_p up.

#### Compute budget estimate

Per the evolution log's submission calendar, the full Stage 0 -> Stage 3
curriculum on 4 x A100-80GB at ~50k training steps fits comfortably
within one week of wall-clock time. Ablations (Interventions A/B/C on a
trained checkpoint) are inference-only and take ~1-2 hours on a single
GPU once the checkpoint is saved.

### Inference

```bash
uv run python -m pdt.cli.infer \
  --config configs/pdt_qwen3_4b.yaml \
  --checkpoint experiments/qwen3_4b/checkpoints/step_0050000.pt \
  --prompt "Tell me three facts about orcas." \
  --max-new-tokens 256
```

Pass `--cf gate_zero | norm_scramble | anchor_swap` to run any of the
three paper-level counterfactuals on the live checkpoint.

### Pre-registered ablations

```bash
uv run python -m pdt.cli.ablate \
  --config configs/pdt_qwen3_4b.yaml \
  --checkpoint experiments/qwen3_4b/checkpoints/step_0050000.pt \
  --prompts-file evaluation/prompts.jsonl \
  --output experiments/qwen3_4b/ablations/manifest.json
```

## What changed vs. the v1 paper and codebase

| Area                          | v1 (arXiv 2512.10054)           | This rebuild                                  |
| ----------------------------- | ------------------------------- | --------------------------------------------- |
| Trunk                         | GPT-OSS-20B                     | Qwen3-4B Base                                 |
| Thesis                        | "Synchronized parallel decoding"| Concept-space co-referencing (existence proof)|
| V_p                           | 65,536                          | 8,192 (planner head: ~3B -> ~336M params)     |
| d_notes                       | 2,048                           | 256                                           |
| Instrumentation               | Silently broken (ModuleList write-back bug) | Identity-asserted; 12 layers wired |
| plan_notes_proj               | Documented but absent from code | Implemented; per-stream snapshot-0 seeder     |
| AgreementHead                 | Hidden-only 1-liner             | Consumes (hidden, W_v, c_v, n_v) per paper    |
| Curriculum name resolver      | Silently fails for SNC / adapters | Reaches every phi parameter                 |
| Counterfactuals A / B / C     | Partial A, no B / no C          | All three implemented properly                |
| Codebook diagnostics          | Absent                          | unique / slot entropy / anchor cosine / histogram |

Detailed findings are in the plan at
`~/.cursor/plans/pdt_qwen3-4b_clean_rebuild_*.plan.md`.

## Testing

```bash
uv run pytest tests/smoke/pdt_tests/ -v
# 11 tests passing
```

## License

MIT. See [LICENSE](LICENSE).
