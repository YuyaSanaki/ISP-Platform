#!/usr/bin/env bash
#SBATCH --job-name=isp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/isp-%j.out
#SBATCH --error=logs/isp-%j.err
## Uncomment / edit for your site:
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --partition=gpu
##SBATCH --qos=normal

# Slurm batch wrapper for H100 native ISP (no containers).
# Edit ISP_ROOT / PIPELINE_CONFIG below, then: sbatch H100/job_slurm.sh

set -euo pipefail

# --- edit these ---
export ISP_ROOT="${ISP_ROOT:-REPLACE_WITH_ABSOLUTE_PATH/ISP-Platform}"
# Must be a host YAML with absolute paths (not stock core/config/* /app/ templates).
export PIPELINE_CONFIG="${PIPELINE_CONFIG:-${ISP_ROOT}/core/config/my_study.yaml}"
export CONDA_ENV="${CONDA_ENV:-isp}"
# export CUDA_MODULE=cuda/12.3.2
# --- end edit ---

H100_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SLURM_SUBMIT_DIR:-.}/logs" "$(dirname "${ISP_ROOT}")/logs" 2>/dev/null || mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-${H100_DIR}}"
bash "${H100_DIR}/run_pipeline.sh" "${PIPELINE_CONFIG}"
