#!/usr/bin/env bash
# Genome-wide ISP, one process per local GPU. Default chunk count is the GPU count.
#
# Usage:
#   bash H100/run_isp_chunks_local.sh /path/to/isp.yaml [CHUNK_COUNT] [RUN_SUBDIR]
#
# Each process sees one GPU via CUDA_VISIBLE_DEVICES. Stats run only if every chunk exits 0.
set -euo pipefail

H100_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${H100_DIR}/env.sh"

ISP_CONFIG="${1:?Pass the host ISP YAML as the first argument.}"
N="${2:-}"
if [[ -z "${N}" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Pass the chunk count as the second argument." >&2
    exit 1
  fi
  N="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
  if [[ -z "${N}" || "${N}" -lt 1 ]]; then
    echo "ERROR: could not detect a GPU. Pass the chunk count as the second argument." >&2
    exit 1
  fi
fi
if [[ "${N}" -lt 2 ]]; then
  echo "ERROR: parallel mode needs at least 2 chunks. A single GPU run is: python3 core/run_isp.py --config ..." >&2
  exit 1
fi

export ISP_OUTPUT_DATE="${ISP_OUTPUT_DATE:-$(date +%Y%m%d)}"
RUN_SUBDIR="${3:-gwisp_${ISP_OUTPUT_DATE}_$(date -u +%H%M%S)}"
export ISP_CONFIG CHUNK_COUNT="${N}" RUN_SUBDIR
LOGDIR="${LOGDIR:-$(dirname "${ISP_ROOT}")/logs}"
mkdir -p "${LOGDIR}"

echo "Parallel ISP: ${N} chunks, subdir ${RUN_SUBDIR}, date ${ISP_OUTPUT_DATE}"
echo "Logs: ${LOGDIR}/isp_chunk_*.log"

pids=()
for ((i = 0; i < N; i++)); do
  CUDA_VISIBLE_DEVICES="${i}" CHUNK_ID="${i}" \
    bash "${H100_DIR}/run_isp_one_chunk.sh" > "${LOGDIR}/isp_chunk_${i}.log" 2>&1 &
  pids+=("$!")
  echo "  chunk ${i} pid ${pids[$i]} GPU ${i}"
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "ERROR: chunk ${i} failed. See ${LOGDIR}/isp_chunk_${i}.log" >&2
    fail=1
  fi
done
if [[ "${fail}" -ne 0 ]]; then
  echo "Not merging. Rerun the failed chunk, then bash H100/run_isp_merge.sh" >&2
  exit 1
fi

bash "${H100_DIR}/run_isp_merge.sh"
