#!/bin/bash
# Continue the Spark long-cell batch sweep after batch 1, which is started separately.
set -uo pipefail
ROOT=/home/yuya-sanaki/ISP-Platform
OUT=${ROOT}/output/benchmarks/isp_machine_compare/spark
PAIRS=4096
for batch in 1 2 4 8 16 32 64 128; do
  dest=${OUT}/long1_b${batch}
  if [[ -f ${dest}/result.json ]]; then
    echo "skip batch ${batch}: result exists"
    python3 - << PY
import json,sys
r=json.load(open("${dest}/result.json"))
print(r.get("forward_batch_size"), "oom", r.get("oom"), "exit", r.get("exit_code"), "phase2", r.get("phase2_seconds"))
sys.exit(0 if r.get("oom") else 1)
PY
    if [[ $? -eq 0 ]]; then
      echo "OOM already recorded at batch ${batch}; stop"
      break
    fi
    continue
  fi
  echo "=== spark batch ${batch} $(date -Is) ==="
  mkdir -p "${dest}"
  docker exec isp_platform_webui python3 /app/benchmarks/isp_machine_compare/run_one.py \
    --dataset /app/output/benchmarks/isp_machine_compare/subsets/long1 \
    --model /app/models/human-Geneformer-V2-104M \
    --batch-size "${batch}" \
    --output "/app/output/benchmarks/isp_machine_compare/spark/long1_b${batch}" \
    --pairs "${PAIRS}" \
    --machine spark \
    --subset long1 \
    --repo /app
  code=$?
  echo "batch ${batch} exit ${code}"
  if [[ -f ${dest}/result.json ]] && python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('oom') else 1)" "${dest}/result.json"; then
    echo "OOM at batch ${batch}; stop"
    break
  fi
  if [[ "${code}" -ne 0 ]]; then
    echo "stop on exit ${code}"
    break
  fi
done
echo "=== spark sweep done $(date -Is) ==="
