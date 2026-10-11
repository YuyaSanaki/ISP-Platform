#!/bin/bash
# Method comparison of Fig. 4A, B (n = 300 somatic cells, K->M->S->O, decoder seeds 0/1/2).
#
#   bash run_methods.sh CODE_ROOT DATASET MODEL OUT [N] [N_EVAL] [PER_JOB]
#
# CODE_ROOT  checkout of commit 9cc9f74 (the code the comparison ran with; see README.md)
# DATASET    tokenized human dataset (somatic / pluripotent), MODEL its fine-tuned checkpoint
# OUT        output folder; then: python3 analyze_x7c_methods.py OUT
#
# Per seed: stage1 (decoders and correction maps -> OUT/seed<k>/state.pt), then the feedback
# loop for OSKM, the positive controls OSK and OSNL and the evaluation placebos, PER_JOB chains
# per call. The calls are independent after stage1 and can be run in parallel.
set -euo pipefail

CODE="${1:?CODE_ROOT}"; DS="${2:?DATASET}"; MODEL="${3:?MODEL}"; OUT="${4:?OUT}"
N="${5:-300}"; NEVAL="${6:-30}"; PER="${7:-3}"; NEST=10; SEEDS=(0 1 2)
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
EXTRA=(--posctrl pc_OSK,pc_OSNL --variants ORE,V0,V1,V2,med_generic,med_specific)

ALL=(OSKM pc_OSK pc_OSNL)
for k in $(seq 0 $((NEVAL - 1))); do ALL+=("eval${k}"); done

export ISP_ROOT="${CODE}"
cd "${CODE}"
for seed in "${SEEDS[@]}"; do
  "${PY}" -u "${HERE}/x7_causal.py" --code-root "${CODE}" --dataset "${DS}" --model "${MODEL}" \
    --out "${OUT}" --max-ncells "${N}" --n-est "${NEST}" --n-eval "${NEVAL}" "${EXTRA[@]}" \
    --mode stage1 --seed "${seed}"
  g=0
  for ((i = 0; i < ${#ALL[@]}; i += PER)); do
    chains="$(IFS=,; echo "${ALL[*]:i:PER}")"
    "${PY}" -u "${HERE}/x7_causal.py" --code-root "${CODE}" --dataset "${DS}" --model "${MODEL}" \
      --out "${OUT}" --max-ncells "${N}" --n-est "${NEST}" --n-eval "${NEVAL}" "${EXTRA[@]}" \
      --mode loop --seed "${seed}" --chains "${chains}" --tag "g${g}"
    g=$((g + 1))
  done
done
