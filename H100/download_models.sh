#!/usr/bin/env bash
# Download Geneformer models / dictionaries / orthologs into ISP_ROOT.
#
# Usage:
#   export ISP_ROOT=/path/to/ISP-Platform
#   # optional: DOWNLOAD_MODELS=default|minimal|all
#   # optional: HF_TOKEN=hf_xxx
#   bash H100/download_models.sh
#
set -euo pipefail

H100_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${H100_DIR}/env.sh"
h100_activate

export DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-default}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

cd "${ISP_ROOT}"
bash scripts/download_build_assets.sh
echo "[H100] Models under ${ISP_ROOT}/models/"
ls -la "${ISP_ROOT}/models/" || true
