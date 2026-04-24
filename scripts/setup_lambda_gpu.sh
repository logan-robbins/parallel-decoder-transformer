#!/usr/bin/env bash
# Bootstrap an NVIDIA GPU host (Lambda Cloud, Paperspace, or any CUDA box)
# for a PDT training run. Assumes Ubuntu 22.04+ with CUDA 12.x drivers.
#
# Target hardware per the evolution-log plan: 4-8x A100-80GB (comfortable) or
# 4x 48GB cards (tight on planner-head size). The frozen 4B trunk in bf16
# takes ~14GB; sidecar phi (~470M params) in bf16 + fp32 optim state takes
# ~3GB per GPU; per-stream KV caches at K=3, 4K context add ~3-4GB per GPU.
#
# Usage:
#   bash scripts/setup_lambda_gpu.sh
#   # Then:
#   uv run scripts/generate_dependency_dataset.py --output data/datasets/ldc/train.jsonl
#   uv run scripts/retokenize_corpus.py --input data/datasets/ldc/train.jsonl \
#     --output data/processed/latent_dependency_control/train.jsonl \
#     --tokenizer /path/to/local/Qwen3-4B-Base
#   uv run torchrun --nproc_per_node=N -m pdt.cli.train --config configs/pdt_qwen3_4b.yaml

set -euo pipefail

# -----------------------------------------------------------------------------
echo "==> Installing uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
fi

# -----------------------------------------------------------------------------
echo "==> Syncing project dependencies (uv sync)"
uv venv .venv --python 3.12
uv sync

# -----------------------------------------------------------------------------
echo "==> Installing flash-attn (CUDA only; harmless to fail on non-CUDA hosts)"
uv pip install flash-attn --no-build-isolation || \
  echo "    flash-attn install failed -- use attn_implementation='sdpa' in YAML."

# -----------------------------------------------------------------------------
echo "==> Pre-caching Qwen3-4B-Base weights"
uv run scripts/check_qwen3_config.py

# -----------------------------------------------------------------------------
echo "==> GPU visibility check"
uv run scripts/check_gpu.py

echo
echo "Bootstrap complete. Next steps:"
echo "  1) uv run scripts/generate_dependency_dataset.py --output data/datasets/ldc/train.jsonl"
echo "  2) uv run scripts/retokenize_corpus.py \\"
echo "         --input data/datasets/ldc/train.jsonl \\"
echo "         --output data/processed/latent_dependency_control/train.jsonl \\"
echo "         --tokenizer /path/to/local/Qwen3-4B-Base"
echo "  3) For single-GPU: uv run scripts/train.py --config configs/pdt_qwen3_4b.yaml"
echo "     For N-GPU DDP:  uv run torchrun --nproc_per_node=N -m pdt.cli.train --config configs/pdt_qwen3_4b.yaml"
