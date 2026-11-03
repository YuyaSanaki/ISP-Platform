#!/usr/bin/env bash
# One parallel ISP chunk on the GPU in CUDA_VISIBLE_DEVICES (default 0).
#
# Required: ISP_CONFIG, CHUNK_ID, CHUNK_COUNT, RUN_SUBDIR
# Optional: ISP_OUTPUT_DATE, ISP_FORWARD_BATCH_SIZE, CUDA_VISIBLE_DEVICES
set -euo pipefail

H100_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${H100_DIR}/env.sh"
h100_install_fail_trace
h100_activate

: "${ISP_CONFIG:?Set ISP_CONFIG to a host YAML with absolute paths.}"
: "${CHUNK_ID:?Set CHUNK_ID (0 .. CHUNK_COUNT-1).}"
: "${CHUNK_COUNT:?Set CHUNK_COUNT (at least 2).}"
: "${RUN_SUBDIR:?Set RUN_SUBDIR to one shared folder name (no slashes).}"

if [[ ! -f "${ISP_CONFIG}" ]]; then
  echo "ERROR: config not found: ${ISP_CONFIG}" >&2
  exit 1
fi
if grep -E '(^|[[:space:]\"'\''=:])/app/' "${ISP_CONFIG}" >/dev/null 2>&1; then
  echo "ERROR: ${ISP_CONFIG} still contains Docker /app/ paths." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

args=(
  python3 "${ISP_ROOT}/core/run_isp.py"
  --config "${ISP_CONFIG}"
  --chunk-id "${CHUNK_ID}"
  --chunk-count "${CHUNK_COUNT}"
  --output-subdir "${RUN_SUBDIR}"
)
if [[ -n "${ISP_OUTPUT_DATE:-}" ]]; then
  args+=(--output-date "${ISP_OUTPUT_DATE}")
fi
if [[ -n "${ISP_FORWARD_BATCH_SIZE:-}" ]]; then
  args+=(--forward-batch-size "${ISP_FORWARD_BATCH_SIZE}")
fi

echo "=== host=$(hostname) chunk ${CHUNK_ID} of ${CHUNK_COUNT} $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="
nvidia-smi -L || true
"${args[@]}"
echo "=== chunk ${CHUNK_ID} done $(date -Is) ==="
