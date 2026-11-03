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
  # Prefer the env's bin/ on PATH. On Pegasus, `conda activate` under `set -u`
  # dies in about 5s: Intel oneAPI deactivate.d reads unset OCL_ICD_* variables,
  # and that abort does not run an ERR trap. `set +u` around activate is not
  # enough when the activate scripts turn nounset back on.
  local _env_bin="${HOME}/.conda/envs/${CONDA_ENV}/bin"
  if [[ -x "${_env_bin}/python3" ]]; then
    export PATH="${_env_bin}:${PATH}"
    echo "[H100] using ${_env_bin}/python3 (PATH; skip conda activate)"
    return 0
  fi
  _h100_source_conda || return 1
  set +u
  conda activate "${CONDA_ENV}"
  set +u
}

# Record the last shell command so a nounset abort or `set -e` exit names a line.
# ERR alone misses the Intel oneAPI failure above. Call once, before h100_activate.
h100_install_fail_trace() {
  # A function, not a trap string: only then BASH_LINENO is the caller line.
  h100_debug_trace() {
    case $BASH_COMMAND in
      h100_on_fail_exit|h100_on_fail_exit\ *|smoke_on_exit|smoke_on_exit\ *)
        H100_FAIL_LINE=$H100_LAST_LINE
        H100_FAIL_CMD=$H100_LAST_CMD
        return 0
        ;;
    esac
    # Bash emits one extra DEBUG hit on the way out, line 1, repeating the
    # last command. Keep the real line.
    if [[ "${BASH_LINENO[0]:-0}" -le 1 && -n "${H100_LAST_CMD:-}" && "$H100_LAST_CMD" == "$BASH_COMMAND" ]]; then
      return 0
    fi
    H100_LAST_LINE=${BASH_LINENO[0]:-?}
    H100_LAST_CMD=$BASH_COMMAND
  }
  h100_on_fail_exit() {
    local rc="$1"
    if [[ "$rc" -eq 0 ]]; then
      return 0
    fi
    printf 'H100_FAIL rc=%s line=%s cmd=%s\n' "$rc" "${H100_FAIL_LINE:-${H100_LAST_LINE:-?}}" "${H100_FAIL_CMD:-${H100_LAST_CMD:-?}}" >&2
  }
  trap h100_debug_trace DEBUG
  # $? must be read in the trap text. A function body sees the DEBUG trap's status.
  trap 'h100_on_fail_exit $?' EXIT
}

echo "[H100] ISP_ROOT=${ISP_ROOT}  CONDA_ENV=${CONDA_ENV}"
