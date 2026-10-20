# Source this file to run ISP Platform natively on an Apple silicon Mac:
#   source scripts/macos/env.sh
# Activates .venv-macos (scripts/macos/setup.sh) and sets the variables the Docker image sets.
_isp_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"
_isp_venv="${ISP_MACOS_VENV:-$_isp_root/.venv-macos}"
if [ ! -f "$_isp_venv/bin/activate" ]; then
  echo "ERROR: $_isp_venv not found. Run: bash scripts/macos/setup.sh" >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1091
. "$_isp_venv/bin/activate"
export GENEFORMER_ROOT="$_isp_root"
export WEBUI_ROOT="$_isp_root"
export PYTHONPATH="$_isp_root/core:$_isp_root/contracts:$_isp_root/webui${PYTHONPATH:+:$PYTHONPATH}"
# Ops without a Metal kernel run on the CPU instead of raising.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
unset _isp_root _isp_venv
