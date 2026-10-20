#!/usr/bin/env bash
# ISP Platform Web UI on an Apple silicon Mac (native, MPS). Open http://localhost:8502
#   bash scripts/macos/webui.sh            # port 8502, this Mac only
#   PORT=8600 bash scripts/macos/webui.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/macos/env.sh"
cd "$REPO_ROOT"
exec streamlit run webui/streamlit_app/app.py \
  --server.address="${ADDRESS:-127.0.0.1}" \
  --server.port="${PORT:-8502}" \
  --server.maxUploadSize 1024000 \
  --browser.gatherUsageStats=false
