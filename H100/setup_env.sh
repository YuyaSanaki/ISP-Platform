#!/usr/bin/env bash
# Create conda env and install ISP Platform deps for H100 (native, no containers).
#
# Usage:
#   export ISP_ROOT=/path/to/ISP-Platform
#   # optional: export CUDA_MODULE=cuda/12.3.2
#   # optional: export CONDA_ENV=isp
#   # optional: export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
#   bash H100/setup_env.sh
#
set -euo pipefail

H100_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${H100_DIR}/env.sh"

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if [[ ! -f "${ISP_ROOT}/requirements.txt" ]]; then
  echo "ERROR: ${ISP_ROOT}/requirements.txt not found" >&2
  exit 1
fi

_h100_source_conda

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  echo "[H100] Creating conda env ${CONDA_ENV} (python=${PYTHON_VERSION})"
  conda create -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y
else
  echo "[H100] Conda env ${CONDA_ENV} already exists"
fi

conda activate "${CONDA_ENV}"

echo "[H100] Installing PyTorch from ${TORCH_INDEX_URL}"
pip install --upgrade pip
pip install torch --index-url "${TORCH_INDEX_URL}"

echo "[H100] Installing filtered requirements (excluding torch/nvidia pins)"
REQ_FILTERED="$(mktemp)"
trap 'rm -f "${REQ_FILTERED}"' EXIT
grep -vE '^(nvidia-|torch|tbb|triton|tensorflow|keras)' "${ISP_ROOT}/requirements.txt" \
  | sed 's/==.*//' > "${REQ_FILTERED}"
pip install -r "${REQ_FILTERED}"
pip install "transformers>=4.40,<5" gdown "huggingface_hub>=0.23" "typing-extensions>=4.13"

echo "[H100] setup_env.sh done. Next: bash ${H100_DIR}/smoke_cuda.sh"
