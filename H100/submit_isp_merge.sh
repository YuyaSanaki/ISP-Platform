#!/usr/bin/env bash
# Submit the statistics job. Run this only after every chunk job has finished.
set -euo pipefail

H100_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${H100_DIR}/env.sh"

: "${ISP_CONFIG:?}"
: "${CHUNK_COUNT:?}"
: "${RUN_SUBDIR:?}"
: "${ISP_OUTPUT_DATE:?}"

QUEUE="${PEGASUS_QUEUE:-gen_S}"
LOGDIR="${LOGDIR:-$(dirname "${ISP_ROOT}")/logs}"
mkdir -p "${LOGDIR}"

qsub_common=(-q "${QUEUE}" -l "elapstim_req=${ISP_MERGE_WALLTIME:-02:00:00}")
if [[ -n "${PEGASUS_PROJECT:-}" ]]; then
  qsub_common+=(-A "${PEGASUS_PROJECT}")
fi

qsub "${qsub_common[@]}" \
  -N isp_merge \
  -o "${LOGDIR}/isp_merge.out" \
  -v "CHUNK_COUNT=${CHUNK_COUNT}" \
  -v "RUN_SUBDIR=${RUN_SUBDIR}" \
  -v "ISP_CONFIG=${ISP_CONFIG}" \
  -v "ISP_OUTPUT_DATE=${ISP_OUTPUT_DATE}" \
  "${H100_DIR}/job_isp_merge_pbs.sh"
