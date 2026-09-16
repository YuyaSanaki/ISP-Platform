#!/usr/bin/env bash
# Run ISP Platform end-to-end pipeline on H100 (native Python, no containers).
# Equivalent to: docker compose run --rm pipeline
#
# Usage:
#   export ISP_ROOT=/path/to/ISP-Platform
#   export PIPELINE_CONFIG=$ISP_ROOT/core/config/my_study.yaml   # host absolute paths
#   bash H100/run_pipeline.sh
#
# Or:
#   bash H100/run_pipeline.sh /path/to/custom.yaml
#
set -euo pipefail

H100_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${H100_DIR}/env.sh"
h100_activate

if [[ $# -ge 1 ]]; then
  PIPELINE_CONFIG="$1"
fi

if [[ -z "${PIPELINE_CONFIG:-}" ]]; then
  echo "ERROR: set PIPELINE_CONFIG to a host YAML (absolute paths), or pass it as \$1." >&2
  echo "  Repo templates under core/config/ still use Docker /app/... paths — copy one and edit." >&2
  echo "  Example:" >&2
  echo "    cp \"\${ISP_ROOT}/core/config/pipeline_1w_human_v2.yaml\" \"\${ISP_ROOT}/core/config/my_study.yaml\"" >&2
  echo "    # edit paths, then:" >&2
  echo "    export PIPELINE_CONFIG=\"\${ISP_ROOT}/core/config/my_study.yaml\"" >&2
  exit 1
fi

if [[ ! -f "${PIPELINE_CONFIG}" ]]; then
  echo "ERROR: config not found: ${PIPELINE_CONFIG}" >&2
  exit 1
fi

# Stock templates use /app/... for Docker Compose; those paths fail on bare metal.
if grep -E '(^|[[:space:]\"'\''=:])/app/' "${PIPELINE_CONFIG}" >/dev/null 2>&1; then
  echo "ERROR: ${PIPELINE_CONFIG} still contains Docker /app/ paths." >&2
  echo "  Copy it and replace input_dir / output_root (and any other paths) with absolute host paths under ${ISP_ROOT}." >&2
  echo "  See H100/README.md Step 5." >&2
  exit 1
fi

echo "=== host=$(hostname) $(date -Is) ==="
nvidia-smi -L || true
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.cuda.get_device_name(0))"

echo "[H100] Running: python3 ${ISP_ROOT}/core/run_pipeline.py --config ${PIPELINE_CONFIG}"
python3 "${ISP_ROOT}/core/run_pipeline.py" --config "${PIPELINE_CONFIG}"
echo "=== done $(date -Is) ==="
