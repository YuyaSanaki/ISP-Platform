#!/usr/bin/env bash
# Verify NVIDIA GPU visibility and PyTorch CUDA on an H100 node.
#
# Usage:
#   export ISP_ROOT=/path/to/ISP-Platform   # optional if env.sh can infer it
#   bash H100/smoke_cuda.sh
#
set -euo pipefail

H100_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${H100_DIR}/env.sh"
h100_activate

echo "=== hostname=$(hostname) $(date -Is) ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
  echo "ERROR: nvidia-smi not found (are you on a GPU node?)" >&2
  exit 1
fi

python3 - <<'PY'
import torch
print("torch", torch.__version__)
ok = torch.cuda.is_available()
print("cuda_available", ok)
if not ok:
    raise SystemExit("CUDA unavailable in this Python environment")
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
print("vram_gb", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1))
PY

echo "[H100] smoke_cuda.sh OK"
