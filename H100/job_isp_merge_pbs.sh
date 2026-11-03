#!/usr/bin/env bash
# Statistics after every gen_S chunk has finished. Submit with H100/submit_isp_merge.sh.
# Do not chain this with qsub --after. Submit it yourself once qstat shows the chunks are done.

#PBS -N isp_merge
#PBS -A REPLACE_WITH_ACCOUNT
#PBS -q gen_S
#PBS -l elapstim_req=02:00:00
#PBS -j o
#PBS -o REPLACE_WITH_LOG_DIR/isp_merge.out
#PBS -v OMP_NUM_THREADS=8
#PBS -v WANDB_DISABLED=true

set -euo pipefail

H100_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${H100_DIR}/run_isp_merge.sh"
