#!/usr/bin/env bash
# Bootstrap the canonical single-H100 CUDA host for PDT's optimizer/32/1k gates.
# Assumes Ubuntu 22.04+ with an NVIDIA driver and at least 200GB persistent disk.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p experiments/bootstrap/logs

run_logged() {
  local label="$1"
  shift
  local log="experiments/bootstrap/logs/${label}.log"
  echo "==> ${label}: $*"
  nohup "$@" >"$log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    tail -n 80 "$log" || true
    sleep 15
  done
  wait "$pid"
  tail -n 80 "$log"
}

if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

# uv owns the project-root .venv and consumes the committed lockfile. The
# canonical path uses PyTorch SDPA; flash-attn is not a prerequisite or fallback.
run_logged uv_sync uv sync --frozen
run_logged cuda_preflight uv run scripts/check_gpu.py --min-memory-gb 75
run_logged qwen_contract uv run scripts/check_qwen3_config.py
run_logged smoke_contracts uv run pytest tests/smoke/ -v

echo "Bootstrap complete. First scientific write:"
echo "  mkdir -p experiments/qwen3_4b/probe_bus/logs"
echo "  nohup uv run scripts/train.py --config configs/pdt_qwen3_4b.yaml \\"
echo "    --optimizer-probe --telemetry-dir experiments/qwen3_4b/probe_bus \\"
echo "    > experiments/qwen3_4b/probe_bus/logs/train.log 2>&1 &"
echo "Poll that log every 15 seconds. Do not start the 32-example run until"
echo "optimizer_probe.json and step_0000002.pt both exist."
