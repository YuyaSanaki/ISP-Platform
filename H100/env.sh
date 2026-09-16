#!/usr/bin/env bash
# Shared environment for H100 native ISP Platform runs.
# Usage: source /path/to/H100/env.sh
#
# Required: ISP_ROOT=/path/to/ISP-Platform
# Optional: CONDA_ENV, CUDA_MODULE

# Do not use set -e when sourced.
_H100_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

if [[ -z "${ISP_ROOT:-}" ]]; then
  # 1) H100/ shipped inside the clone: .../ISP-Platform/H100
  if [[ -f "${_H100_DIR}/../core/run_pipeline.py" ]]; then
    ISP_ROOT="$(cd "${_H100_DIR}/.." && pwd)"
  # 2) Sibling layout: .../H100 next to .../ISP-Platform
  elif [[ -d "${_H100_DIR}/../ISP-Platform/core" ]]; then
    ISP_ROOT="$(cd "${_H100_DIR}/../ISP-Platform" && pwd)"
  # 3) Already sitting in the repo root
  elif [[ -d "${PWD}/core" && -f "${PWD}/core/run_pipeline.py" ]]; then
    ISP_ROOT="${PWD}"
  else
    echo "ERROR: set ISP_ROOT to your ISP-Platform checkout" >&2
    return 1 2>/dev/null || exit 1
  fi
fi

export ISP_ROOT
export CONDA_ENV="${CONDA_ENV:-isp}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ISP_ROOT}/core:${ISP_ROOT}/contracts:${ISP_ROOT}/webui${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${CUDA_MODULE:-}" ]]; then
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi

_h100_source_conda() {
  if [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PREFIX}/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
    return 0
  fi
  local _conda_sh
  for _conda_sh in \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/miniforge3/etc/profile.d/conda.sh" \
    "${HOME}/mambaforge/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "${HOME}/.conda/etc/profile.d/conda.sh"
  do
    if [[ -f "${_conda_sh}" ]]; then
      # shellcheck source=/dev/null
      source "${_conda_sh}"
      return 0
    fi
  done
  if command -v conda >/dev/null 2>&1; then
    local _base
    _base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${_base}" && -f "${_base}/etc/profile.d/conda.sh" ]]; then
      # shellcheck source=/dev/null
      source "${_base}/etc/profile.d/conda.sh"
      return 0
    fi
  fi
  echo "ERROR: conda not found. Install Miniconda/Mambaforge or load your site conda module." >&2
  return 1
}

h100_activate() {
  _h100_source_conda || return 1
  conda activate "${CONDA_ENV}"
}

echo "[H100] ISP_ROOT=${ISP_ROOT}  CONDA_ENV=${CONDA_ENV}"
