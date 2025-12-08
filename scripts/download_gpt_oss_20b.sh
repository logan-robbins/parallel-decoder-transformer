#!/usr/bin/env bash
set -euo pipefail

# Download OpenAI GPT-OSS 20B weights into a local directory layout
# expected by this repo (both directories are REQUIRED):
#   gpt-oss-20b/
#     ├─ original/    # full Hugging Face model repo
#     └─ tokenizer/   # tokenizer files required by the CLI

MODEL_ID="openai/gpt-oss-20b"
ROOT_DIR="gpt-oss-20b"
ORIG_DIR="${ROOT_DIR}/original"
TOK_DIR="${ROOT_DIR}/tokenizer"

echo "==> Preparing target directories"
mkdir -p "${ORIG_DIR}" "${TOK_DIR}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "ERROR: huggingface-cli not found. Install it with:"
  echo "  python -m pip install -U 'huggingface_hub[cli]'"
  exit 1
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "==> Logging into Hugging Face via token from HF_TOKEN env"
  huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential true || true
fi

echo "==> Downloading ${MODEL_ID} to ${ORIG_DIR} (resume enabled)"
huggingface-cli download "${MODEL_ID}" \
  --local-dir "${ORIG_DIR}" \
  --local-dir-use-symlinks False \
  --resume-download \
  --exclude "*.gguf" || {
    echo "Download failed. If the repo is gated, run:"
    echo "  huggingface-cli login"
    echo "and/or export HF_TOKEN=... then re-run this script."
    exit 1
  }

echo "==> Staging tokenizer files to ${TOK_DIR}"
# Copy common tokenizer artifacts if present; ignore missing matches.
shopt -s nullglob
declare -a TOK_FILES=(
  tokenizer.json tokenizer_config.json special_tokens_map.json
  vocab.json merges.txt spiece.model sentencepiece.bpe.model
  added_tokens.json
)
for f in "${TOK_FILES[@]}"; do
  if [[ -f "${ORIG_DIR}/${f}" ]]; then
    cp -f "${ORIG_DIR}/${f}" "${TOK_DIR}/"
  fi
done
shopt -u nullglob

# Verify that tokenizer artifacts exist
if [[ -z "$(ls -A "${TOK_DIR}" 2>/dev/null)" ]]; then
  echo "ERROR: No tokenizer files were found in '${ORIG_DIR}'."
  echo "Expected one of: ${TOK_FILES[*]}"
  echo "The tokenizer directory '${TOK_DIR}' is required by the CLI."
  echo "If your model repo organizes files differently, copy the tokenizer files into '${TOK_DIR}' manually."
  exit 1
fi

echo "==> Download complete. Summary:"
du -hs "${ROOT_DIR}" || true
echo "Contents of ${ORIG_DIR}:"
ls -1 "${ORIG_DIR}" | head -n 50
echo "Contents of ${TOK_DIR}:"
ls -1 "${TOK_DIR}" || true

echo
echo "Next steps:"
echo "- Point configs to base_model: '${ORIG_DIR}' (already default in project configs)."
echo "- The CLI requires '${TOK_DIR}' to exist."
echo "- To re-run download with auth: export HF_TOKEN=... and re-run this script."
