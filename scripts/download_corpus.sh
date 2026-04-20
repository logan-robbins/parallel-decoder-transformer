#!/usr/bin/env bash
# Download the PDT teacher-corpus Parquet tarball (teacher plans + notes + rollback
# payloads + raw text for all 10k training examples). Extract into data/datasets/.
#
# This tarball preserves every expensive GPT-5.1 + GPT-4.1 teacher call as plain
# JSON/text columns and is tokenizer- and V_p-agnostic. Reusing it costs $0 of
# LLM spend; retokenizing for Qwen3 and re-hashing planner IDs at V_p=8192 is a
# local CPU job (see src/pdt/datasets/retokenize.py).
set -euo pipefail

# -----------------------------------------------------------------------------
BASE_URL="https://storage.googleapis.com/parallel-decoder-transformer/data/archives"
ARCHIVE_URL="${PDT_CORPUS_URL:-${BASE_URL}/pdt_10k_gpt41_parquet.tar.gz}"
ARCHIVE_NAME="$(basename "${ARCHIVE_URL}")"
TARGET_ROOT="data/datasets"
TARGET_DIR="${TARGET_ROOT}/pdt_10k_gpt41"

# -----------------------------------------------------------------------------
echo "==> Preparing target directory: ${TARGET_DIR}"
mkdir -p "${TARGET_ROOT}"

if [[ -d "${TARGET_DIR}" && -n "$(ls -A "${TARGET_DIR}" 2>/dev/null || true)" ]]; then
  echo "    ${TARGET_DIR} already exists and is non-empty. Skipping download."
  echo "    Remove it to re-pull: rm -rf ${TARGET_DIR}"
  exit 0
fi

echo "==> Downloading ${ARCHIVE_NAME} from:"
echo "    ${ARCHIVE_URL}"
cd "${TARGET_ROOT}"
if command -v curl >/dev/null 2>&1; then
  curl -L --retry 3 --fail --output "${ARCHIVE_NAME}" "${ARCHIVE_URL}"
elif command -v wget >/dev/null 2>&1; then
  wget --tries=3 --output-document "${ARCHIVE_NAME}" "${ARCHIVE_URL}"
else
  echo "ERROR: neither curl nor wget found." >&2
  exit 1
fi

echo "==> Extracting into ${TARGET_ROOT}/"
tar -xzf "${ARCHIVE_NAME}"
rm -f "${ARCHIVE_NAME}"

cd - >/dev/null
echo "==> Done."
echo "    Parquet splits: ${TARGET_DIR}/{train,validation,test}.parquet"
echo
echo "Next step: run retokenization + re-hash to emit Qwen3 JSONL at V_p=8192, d_notes=256:"
echo "    uv run python scripts/retokenize_corpus.py \\"
echo "      --input-dir ${TARGET_DIR} \\"
echo "      --output-dir data/processed/pdt_10k_qwen3_4b \\"
echo "      --tokenizer Qwen/Qwen3-4B-Base \\"
echo "      --plan-hash-buckets 8192 \\"
echo "      --notes-dim 256"
echo
echo "Alternative (if the Parquet archive is unavailable, fall back to the"
echo "pre-tokenized JSONL archives which require no retokenization -- but"
echo "note they embed GPT-OSS tokenizer IDs which must be replaced):"
echo "    ${BASE_URL}/pdt_10k_gpt41_jsonl_train.tar.gz"
echo "    ${BASE_URL}/pdt_10k_gpt41_jsonl_eval.tar.gz"
