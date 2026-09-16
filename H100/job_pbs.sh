#!/usr/bin/env bash
#------- qsub / PBS / NQSV-style options (edit for your site) -----------
#PBS -N isp
#PBS -A REPLACE_WITH_ACCOUNT
#PBS -q REPLACE_WITH_GPU_QUEUE
#PBS -l elapstim_req=24:00:00
#PBS -j o
#PBS -o REPLACE_WITH_LOG_DIR/isp.out
#PBS -v OMP_NUM_THREADS=8
#PBS -v WANDB_DISABLED=true
#
# Notes:
# - On some Ubuntu HPC images /bin/sh is dash; keep this shebang as bash.
# - Walltime flag names differ (elapstim_req vs walltime). Adjust to your site.
# - Submit from a login node: qsub H100/job_pbs.sh

set -euo pipefail

# --- edit these ---
export ISP_ROOT="${ISP_ROOT:-REPLACE_WITH_ABSOLUTE_PATH/ISP-Platform}"
# Must be a host YAML with absolute paths (not stock core/config/* /app/ templates).
export PIPELINE_CONFIG="${PIPELINE_CONFIG:-${ISP_ROOT}/core/config/my_study.yaml}"
export CONDA_ENV="${CONDA_ENV:-isp}"
# export CUDA_MODULE=cuda/12.3.2
LOGDIR="${LOGDIR:-$(dirname "${ISP_ROOT}")/logs}"
# --- end edit ---

H100_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${LOGDIR}"
cd "${PBS_O_WORKDIR:-${H100_DIR}}"

exec > >(tee -a "${LOGDIR}/isp.${PBS_JOBID:-manual}.log") 2>&1

bash "${H100_DIR}/run_pipeline.sh" "${PIPELINE_CONFIG}"
