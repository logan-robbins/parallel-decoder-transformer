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
#   bash scripts/download_corpus.sh
#   uv run python scripts/retokenize_corpus.py --input-dir data/datasets/pdt_10k_gpt41 \
#     --output-dir data/processed/pdt_10k_qwen3_4b \
#     --tokenizer Qwen/Qwen3-4B-Base --plan-hash-buckets 8192 --notes-dim 256
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
uv run python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Pre-fetching tokenizer...')
AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Base', use_fast=True)
print('Pre-fetching model config (not weights, for spot-check)...')
from transformers import AutoConfig
c = AutoConfig.from_pretrained('Qwen/Qwen3-4B-Base')
print(f'  hidden_size = {c.hidden_size}, num_hidden_layers = {c.num_hidden_layers}')
print(f'  num_attention_heads = {c.num_attention_heads}, num_key_value_heads = {c.num_key_value_heads}')
"

# -----------------------------------------------------------------------------
echo "==> GPU visibility check"
uv run python -c "
import torch
print('torch:', torch.__version__)
print('cuda.is_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU[{i}]:', torch.cuda.get_device_name(i),
              f'{torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
"

echo
echo "Bootstrap complete. Next steps:"
echo "  1) bash scripts/download_corpus.sh"
echo "  2) uv run python scripts/retokenize_corpus.py \\"
echo "         --input-dir data/datasets/pdt_10k_gpt41 \\"
echo "         --output-dir data/processed/pdt_10k_qwen3_4b \\"
echo "         --tokenizer Qwen/Qwen3-4B-Base \\"
echo "         --plan-hash-buckets 8192 --notes-dim 256"
echo "  3) For single-GPU: uv run python -m pdt.cli.train --config configs/pdt_qwen3_4b.yaml"
echo "     For N-GPU DDP:  uv run torchrun --nproc_per_node=N -m pdt.cli.train --config configs/pdt_qwen3_4b.yaml"
