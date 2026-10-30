#!/bin/bash
#PBS -N ispbfollow
#PBS -A LABGENPH
#PBS -q gen_S
#PBS -l elapstim_req=02:00:00
#PBS -j o
#PBS -o /work/LABGENPH/yuya_1/AsanoISP/isp_bench_20261029/logs/ispbfollow.out
#PBS -v OMP_NUM_THREADS=8
#PBS -v WANDB_DISABLED=true

set -uo pipefail
BENCH=/work/LABGENPH/yuya_1/AsanoISP/isp_bench_20261029
ISP_ROOT=/work/LABGENPH/yuya_1/AsanoISP/ISP-Platform
export ISP_ROOT CONDA_ENV=Asano_isp
# shellcheck source=/dev/null
source "${ISP_ROOT}/H100/env.sh"
h100_activate
exec > >(tee -a "${BENCH}/logs/follow.${PBS_JOBID:-local}.log") 2>&1
echo "=== host $(hostname) $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

run_case() {
  local subset="$1" batch="$2" pairs="$3" tag="$4"
  local out="${BENCH}/pegasus/${tag}"
  echo "=== ${tag} $(date -Is) ==="
  python3 "${BENCH}/run_one.py" \
    --dataset "${BENCH}/subsets/${subset}" \
    --model "${ISP_ROOT}/models/human-Geneformer-V2-104M" \
    --batch-size "${batch}" \
    --output "${out}" \
    --pairs "${pairs}" \
    --machine pegasus \
    --subset "${subset}" \
    --repo "${ISP_ROOT}"
  echo "exit $?"
}

# GPU-memory sample at the fast plateau, then push until OOM.
run_case long1 64 4096 long1_b64_gpu
for batch in 256 512 1024; do
  run_case long1 "${batch}" 4096 "long1_b${batch}"
  if python3 -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get('oom') or r.get('exit_code') else 1)" "${BENCH}/pegasus/long1_b${batch}/result.json"; then
    echo "stop after batch ${batch}"
    break
  fi
done
run_case repr 64 12384 repr_b64
echo "=== followup done $(date -Is) ==="
