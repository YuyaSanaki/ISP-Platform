#!/usr/bin/env bash
# Native ISP Platform environment for Apple silicon Macs (no Docker).
#
# Docker Desktop on macOS cannot use the Apple GPU, so ISP Platform runs natively here:
# PyTorch uses the Metal backend (MPS). Creates .venv-macos with Python 3.12 (uv), installs
# the same packages as the Docker image (requirements.txt with the CUDA-only ones removed
# and == pins dropped, as in the Dockerfile), then downloads models, token dictionaries
# and gene set libraries into the checkout.
#
#   bash scripts/macos/setup.sh              # environment + default assets
#   DOWNLOAD_MODELS=minimal bash scripts/macos/setup.sh
#   DOWNLOAD_MODELS=none bash scripts/macos/setup.sh   # environment only
#
# Requires: macOS on Apple silicon, uv (https://docs.astral.sh/uv/), git, curl.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${ISP_MACOS_VENV:-$REPO_ROOT/.venv-macos}"
cd "$REPO_ROOT"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: this script is for macOS on Apple silicon (got $(uname -s) $(uname -m))." >&2
  echo "On Linux use Docker (README: Quick start)." >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    install_cmd="brew install uv"
  else
    install_cmd="curl -LsSf https://astral.sh/uv/install.sh | sh"
  fi
  cat >&2 <<EOF
ERROR: uv is not installed.

uv is a free tool that sets up Python and the packages ISP Platform needs.
It is installed once per Mac; it does not change your other Python setups.

  1. Install uv. Copy this line into the Terminal and press Enter:

       $install_cmd

  2. Close this Terminal window and open a new one (so the uv command is found),
     then go back to this folder:

       cd "$REPO_ROOT"

  3. Run the setup again:

       bash scripts/macos/setup.sh

Check that uv works with:  uv --version
More help: https://docs.astral.sh/uv/getting-started/installation/
EOF
  exit 1
fi

echo "=== Python environment: $VENV ==="
uv venv --python 3.12 --allow-existing "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

REQ="$(mktemp)"
trap 'rm -f "$REQ"' EXIT
# Same filter as the Dockerfile, plus packages that only build against CUDA.
grep -v -E '^(nvidia-|torch|tbb|triton|tensorflow|keras|deepspeed)' requirements.txt \
  | sed 's/==.*//' > "$REQ"
uv pip install -r "$REQ"
uv pip install "torch>=2.5" "transformers>=4.40,<5" gdown "huggingface_hub>=0.23" \
  "typing-extensions>=4.13" "datasets==5.0.1"

python - <<'PY'
import torch
print("torch", torch.__version__, "| MPS available:", torch.backends.mps.is_available())
if not torch.backends.mps.is_available():
    raise SystemExit("ERROR: PyTorch cannot use MPS on this Mac (macOS 13.3 or later needed).")
PY

echo "=== Assets (DOWNLOAD_MODELS=${DOWNLOAD_MODELS:-default}) ==="
GENEFORMER_ROOT="$REPO_ROOT" DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-default}" \
  bash "$REPO_ROOT/scripts/download_build_assets.sh"

echo
echo "Done. Start the Web UI with:  bash scripts/macos/webui.sh"
echo "Command-line jobs:            source scripts/macos/env.sh && python3 core/run_isp.py --config ..."
