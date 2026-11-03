#!/usr/bin/env bash
#SBATCH --job-name=isp_chunk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/isp-chunk-%A_%a.out
#SBATCH --error=logs/isp-chunk-%A_%a.err
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --partition=gpu

# Slurm array, one GPU per task. The array must be 0 .. CHUNK_COUNT-1.
# Default CHUNK_COUNT is 8, matching --array=0-7. Change both together.
# Pegasus is NQSV, not Slurm. Use submit_isp_chunks.sh there.
#
#   export ISP_CONFIG=/path/to/isp.yaml
#   export RUN_SUBDIR=gwisp_run1
#   sbatch H100/job_isp_chunks_slurm.sh

set -euo pipefail

H100_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SLURM_SUBMIT_DIR:-.}/logs" logs

: "${ISP_CONFIG:?}"
: "${RUN_SUBDIR:?}"
export CHUNK_COUNT="${CHUNK_COUNT:-8}"
export CHUNK_ID="${SLURM_ARRAY_TASK_ID:?Submit with sbatch --array, not srun.}"
if [[ "${CHUNK_ID}" -ge "${CHUNK_COUNT}" ]]; then
  echo "ERROR: array task ${CHUNK_ID} is outside chunk count ${CHUNK_COUNT}." >&2
  exit 1
fi

# Keep the device Slurm assigned. If it assigned none, use 0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
bash "${H100_DIR}/run_isp_one_chunk.sh"
