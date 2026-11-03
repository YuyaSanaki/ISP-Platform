#!/usr/bin/env bash
# One Pegasus / NQSV chunk: gen_S, one H100, gpu id 0.
# Submit with H100/submit_isp_chunks.sh (do not use gen_M or gen_L).
#
# Edit the account, or pass -A from the submit script via PEGASUS_PROJECT.
# Walltime 04:00:00 fits max_ncells 2000 split into 8 chunks (about 2 h plus margin).
# A full cell set that would take ~30 h on one H100 is about 4 h per chunk at N=8.

#PBS -N isp_chunk
#PBS -A REPLACE_WITH_ACCOUNT
#PBS -q gen_S
#PBS -l elapstim_req=04:00:00
#PBS -j o
#PBS -o REPLACE_WITH_LOG_DIR/isp_chunk.out
#PBS -v OMP_NUM_THREADS=8
#PBS -v WANDB_DISABLED=true

set -euo pipefail

H100_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=0
bash "${H100_DIR}/run_isp_one_chunk.sh"
