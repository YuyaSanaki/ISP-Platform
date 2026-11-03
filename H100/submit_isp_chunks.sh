#!/usr/bin/env bash
# Submit N independent gen_S jobs (one H100 each). Default N is 8.
#
#   export ISP_CONFIG=/work/.../my_isp.yaml
#   export PEGASUS_PROJECT=LABGENPH          # optional if #PBS -A is already edited
#   bash H100/submit_isp_chunks.sh [N]
#
# Values passed with qsub -v have no spaces and no commas.
# Merge is a separate qsub. This script prints that command and does not wait.
set -euo pipefail

H100_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${H100_DIR}/env.sh"

: "${ISP_CONFIG:?Set ISP_CONFIG to a host YAML.}"
N="${1:-8}"
if [[ "${N}" -lt 2 ]]; then
  echo "ERROR: chunk count must be at least 2." >&2
  exit 1
fi

export ISP_OUTPUT_DATE="${ISP_OUTPUT_DATE:-$(date +%Y%m%d)}"
RUN_SUBDIR="${RUN_SUBDIR:-gwisp_${ISP_OUTPUT_DATE}_$(date -u +%H%M%S)}"
QUEUE="${PEGASUS_QUEUE:-gen_S}"
WALLTIME="${ISP_CHUNK_WALLTIME:-04:00:00}"
LOGDIR="${LOGDIR:-$(dirname "${ISP_ROOT}")/logs}"
mkdir -p "${LOGDIR}"

echo "Submitting ${N} ${QUEUE} jobs. subdir=${RUN_SUBDIR} date=${ISP_OUTPUT_DATE} walltime=${WALLTIME}"

qsub_common=(-q "${QUEUE}" -l "elapstim_req=${WALLTIME}")
if [[ -n "${PEGASUS_PROJECT:-}" ]]; then
  qsub_common+=(-A "${PEGASUS_PROJECT}")
fi

for ((i = 0; i < N; i++)); do
  qsub "${qsub_common[@]}" \
    -N "isp_c${i}" \
    -o "${LOGDIR}/isp_chunk_${i}.out" \
    -v "CHUNK_ID=${i}" \
    -v "CHUNK_COUNT=${N}" \
    -v "RUN_SUBDIR=${RUN_SUBDIR}" \
    -v "ISP_CONFIG=${ISP_CONFIG}" \
    -v "ISP_OUTPUT_DATE=${ISP_OUTPUT_DATE}" \
    "${H100_DIR}/job_isp_chunk_pbs.sh"
done

echo
echo "After qstat shows every isp_c* job has finished, submit the merge:"
echo "  ISP_CONFIG=${ISP_CONFIG} CHUNK_COUNT=${N} RUN_SUBDIR=${RUN_SUBDIR} ISP_OUTPUT_DATE=${ISP_OUTPUT_DATE} bash ${H100_DIR}/submit_isp_merge.sh"
