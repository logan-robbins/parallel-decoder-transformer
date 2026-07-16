#!/usr/bin/env bash
# Bootstrap a canonical single-H100 CUDA host for one pinned dense PDT profile.
# Assumes Ubuntu 22.04+ with an NVIDIA driver and at least 200GB persistent disk.

set -euo pipefail

PROFILE="qwen3_4b_instruct_2507"
if [[ $# -gt 0 ]]; then
  if [[ $# -ne 2 || "$1" != "--trunk-profile" ]]; then
    echo "usage: $0 [--trunk-profile qwen3_4b_instruct_2507|qwen3_14b]" >&2
    exit 2
  fi
  PROFILE="$2"
fi
if [[ "$PROFILE" != "qwen3_4b_instruct_2507" && "$PROFILE" != "qwen3_14b" ]]; then
  echo "unsupported trunk profile: $PROFILE" >&2
  exit 2
fi

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
run_logged qwen_contract uv run scripts/check_qwen3_config.py --trunk-profile "$PROFILE"
run_logged smoke_contracts uv run pytest tests/smoke/ -v

echo "Bootstrap complete for profile $PROFILE. Generate and retokenize the"
echo "long-form data for this profile before the first optimizer probe."
echo "Poll every long-running command every 15 seconds. Do not train until"
echo "optimizer_probe.json and step_0000002.pt both exist."
