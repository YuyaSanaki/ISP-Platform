#!/usr/bin/env bash
#SBATCH --job-name=isp_merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/isp-merge-%j.out
#SBATCH --error=logs/isp-merge-%j.err
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --partition=gpu

# Submit after the array in job_isp_chunks_slurm.sh has finished.
#   export ISP_CONFIG=... RUN_SUBDIR=... CHUNK_COUNT=8
#   sbatch H100/job_isp_merge_slurm.sh

set -euo pipefail

H100_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p logs
: "${ISP_CONFIG:?}"
: "${RUN_SUBDIR:?}"
: "${CHUNK_COUNT:?}"
bash "${H100_DIR}/run_isp_merge.sh"
