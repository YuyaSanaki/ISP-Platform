#!/usr/bin/env bash
# Statistics for a finished parallel ISP run. Run only after every chunk has exited 0.
#
# Required: ISP_CONFIG, CHUNK_COUNT, RUN_SUBDIR
# Optional: ISP_OUTPUT_DATE, ISP_FORWARD_BATCH_SIZE
set -euo pipefail

H100_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${H100_DIR}/env.sh"
h100_install_fail_trace
h100_activate

: "${ISP_CONFIG:?Set ISP_CONFIG to the same host YAML the chunks used.}"
: "${CHUNK_COUNT:?Set CHUNK_COUNT.}"
: "${RUN_SUBDIR:?Set RUN_SUBDIR to the same name the chunks used.}"

if [[ ! -f "${ISP_CONFIG}" ]]; then
  echo "ERROR: config not found: ${ISP_CONFIG}" >&2
  exit 1
fi

args=(
  python3 "${ISP_ROOT}/core/run_isp_merge.py"
  --config "${ISP_CONFIG}"
  --chunk-count "${CHUNK_COUNT}"
  --output-subdir "${RUN_SUBDIR}"
)
if [[ -n "${ISP_OUTPUT_DATE:-}" ]]; then
  args+=(--output-date "${ISP_OUTPUT_DATE}")
fi
if [[ -n "${ISP_FORWARD_BATCH_SIZE:-}" ]]; then
  args+=(--forward-batch-size "${ISP_FORWARD_BATCH_SIZE}")
fi

echo "=== merge host=$(hostname) $(date -Is) chunks=${CHUNK_COUNT} subdir=${RUN_SUBDIR} ==="
"${args[@]}"
echo "=== merge done $(date -Is) ==="
