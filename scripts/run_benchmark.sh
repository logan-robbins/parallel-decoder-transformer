#!/usr/bin/env bash
set -euo pipefail

# Wrapper script to run a benchmark comparison between the Parallel Decoder Transformer parallel
# model and the sequential baseline.

# Ensure the script is run from the project root
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")
cd "$PROJECT_ROOT"

# Activate the virtual environment
source ./.venv/bin/activate

OUTPUT_DIR="experiments/benchmark"
PAR_MANIFEST="$OUTPUT_DIR/parallel_manifest.json"
SEQ_MANIFEST="$OUTPUT_DIR/sequential_manifest.json"
PROMPT="Write a short, three-paragraph summary of the history of the internet."

mkdir -p "$OUTPUT_DIR"

# Precompute shared CLI snippets
PARALLEL_ROLES=(--stream stream_1 --stream stream_2 --stream stream_3)
SHARED_FLAGS=(
  --config configs/gpt_oss_transfer.yaml
  --prompt "$PROMPT"
  --stream-prefix-file stream_prefixes.json
  --seed-text-file seed_texts.json
  --max-new-tokens 512
)

echo "--- Running Benchmark --- "

# 1. Run inference in parallel (Parallel Decoder Transformer) mode
echo "--> Step 1: Running Parallel Decoder Transformer (parallel) inference..."
python scripts/infer.py \
  "${SHARED_FLAGS[@]}" \
  "${PARALLEL_ROLES[@]}" \
  --read-lag-delta 0 --alpha 1 --gate-g 1 \
  --output "$PAR_MANIFEST"

echo "Parallel run manifest saved to $PAR_MANIFEST"

# 2. Run inference in sequential (baseline) mode (no --stream overrides allowed)
echo "--> Step 2: Running Baseline (sequential) inference..."
python scripts/infer.py \
  "${SHARED_FLAGS[@]}" \
  --baseline sequential \
  --output "$SEQ_MANIFEST"

echo "Sequential run manifest saved to $SEQ_MANIFEST"

# 3. Compare the results
echo "--> Step 3: Comparing outputs and generating report..."
python scripts/compare_seq_parallel.py \
  --par-manifest "$PAR_MANIFEST" \
  --seq-manifest "$SEQ_MANIFEST"

# 4. Summarize rollback/gate/coverage metrics for figure repros
echo "--> Step 4: Summarizing rollback/gate/coverage metrics..."
python scripts/summarize_infer_manifests.py \
  "$PAR_MANIFEST" \
  "$SEQ_MANIFEST" \
  --output "$OUTPUT_DIR/metrics_summary.json"

echo "--- Benchmark Complete ---"
