#!/bin/bash
#PBS -N ispbench
#PBS -A LABGENPH
#PBS -q gen_S
#PBS -l elapstim_req=08:00:00
#PBS -j o
#PBS -o /work/LABGENPH/yuya_1/AsanoISP/isp_bench_20261029/logs/ispbench.out
#PBS -v OMP_NUM_THREADS=8
#PBS -v WANDB_DISABLED=true

set -uo pipefail

BENCH=/work/LABGENPH/yuya_1/AsanoISP/isp_bench_20261029
ISP_ROOT=/work/LABGENPH/yuya_1/AsanoISP/ISP-Platform
export ISP_ROOT
export CONDA_ENV=Asano_isp
# shellcheck source=/dev/null
source "${ISP_ROOT}/H100/env.sh"
h100_activate

mkdir -p "${BENCH}/logs" "${BENCH}/pegasus"
exec > >(tee -a "${BENCH}/logs/sweep.${PBS_JOBID:-local}.log") 2>&1
echo "=== host $(hostname) $(date -Is) job ${PBS_JOBID:-local} ==="
nvidia-smi -L || true
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,power.limit --format=csv || true
python3 -c 'import torch,transformers,sys; print("py",sys.version.split()[0]); print("torch",torch.__version__); print("cuda",torch.version.cuda); print("transformers",transformers.__version__); print("gpu",torch.cuda.get_device_name(0)); f,t=torch.cuda.mem_get_info(); print("torch_gib",round(f/1024**3,2),round(t/1024**3,2))'

PAIRS=4096
for batch in 1 2 4 8 16 32 64 128; do
  out="${BENCH}/pegasus/long1_b${batch}"
  echo "=== batch ${batch} $(date -Is) ==="
  python3 "${BENCH}/run_one.py" \
    --dataset "${BENCH}/subsets/long1" \
    --model "${ISP_ROOT}/models/human-Geneformer-V2-104M" \
    --batch-size "${batch}" \
    --output "${out}" \
    --pairs "${PAIRS}" \
    --machine pegasus \
    --subset long1 \
    --repo "${ISP_ROOT}"
  code=$?
  echo "batch ${batch} exit ${code}"
  if [[ -f "${out}/result.json" ]] && python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('oom') else 1)" "${out}/result.json"; then
    echo "OOM at batch ${batch}; stop sweep"
    break
  fi
  if [[ "${code}" -ne 0 ]]; then
    echo "non-zero exit ${code}; stop sweep"
    break
  fi
done
echo "=== sweep done $(date -Is) ==="
