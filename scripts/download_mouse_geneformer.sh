#!/usr/bin/env bash
# Download Mouse-Geneformer pretrained weights (base + optional 12L-E20) and
# Mouse-Genecorpus-20M dictionaries.
#
# Usage:
#   bash scripts/download_mouse_geneformer.sh           # base + 12L + dicts
#   bash scripts/download_mouse_geneformer.sh --base     # base only
#   bash scripts/download_mouse_geneformer.sh --12l      # 12L-E20 only
#   bash scripts/download_mouse_geneformer.sh --dicts    # dictionaries only
#   bash scripts/download_mouse_geneformer.sh --prune-bin  # drop .bin after safetensors
set -euo pipefail

ROOT="${GENEFORMER_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL_ROOT="$ROOT/models"
DICT_DIR="$ROOT/core/geneformer/dicts/mouse"
BASE_DIR="$MODEL_ROOT/mouse-Geneformer"
L12_DIR="$MODEL_ROOT/mouse-Geneformer-12L-E20"

BASE_DRIVE_ID="1gM3gcc3DlNGt5bAcqHbeRxtdMktGeDEg"
L12_DRIVE_ID="1xKMyFA4JJeRigcJPsU2XNyxEW25Q247u"
HF_DICT_BASE="https://huggingface.co/datasets/MPRG/Mouse-Genecorpus-20M/resolve/main"
TOKEN_DICT_URL="${HF_DICT_BASE}/MLM-re_token_dictionary_v1.pkl"
MEDIAN_DICT_URL="${HF_DICT_BASE}/mouse_gene_median_dictionary.pkl"
SYMBOL_DICT_URL="${HF_DICT_BASE}/MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl"

DO_BASE=0
DO_12L=0
DO_DICTS=0
PRUNE_BIN=0
EXPLICIT_TARGET=0

if [[ $# -eq 0 ]]; then
  DO_BASE=1
  DO_12L=1
  DO_DICTS=1
else
  for arg in "$@"; do
    case "$arg" in
      --base) DO_BASE=1; EXPLICIT_TARGET=1 ;;
      --12l|--12l-e20) DO_12L=1; EXPLICIT_TARGET=1 ;;
      --dicts) DO_DICTS=1; EXPLICIT_TARGET=1 ;;
      --prune-bin) PRUNE_BIN=1 ;;
      -h|--help)
        sed -n '2,13p' "$0"
        exit 0
        ;;
      *)
        echo "Unknown option: $arg" >&2
        exit 1
        ;;
    esac
  done
  # Allow `...sh --prune-bin` to mean "defaults + prune"
  if [[ "$EXPLICIT_TARGET" -eq 0 ]]; then
    DO_BASE=1
    DO_12L=1
    DO_DICTS=1
  fi
fi

mkdir -p "$MODEL_ROOT" "$DICT_DIR"

ensure_gdown() {
  if ! python3 -c "import gdown" >/dev/null 2>&1; then
    echo "Installing gdown..."
    pip install -q gdown
  fi
}

ensure_safetensors() {
  # BUG-4: newer transformers prefer model.safetensors (or torch≥2.6 for .bin).
  # Convert with torch+safetensors only — AutoModelForMaskedLM pulls Triton and
  # fails in GPU-less Docker build environments.
  local dest_dir="$1"
  if [[ -f "$dest_dir/model.safetensors" ]]; then
    echo "Already has model.safetensors: $dest_dir"
    chmod a+r "$dest_dir/model.safetensors" 2>/dev/null || true
    if [[ "$PRUNE_BIN" -eq 1 && -f "$dest_dir/pytorch_model.bin" ]]; then
      rm -f "$dest_dir/pytorch_model.bin"
      echo "Pruned pytorch_model.bin (--prune-bin): $dest_dir"
    fi
    return 0
  fi
  if [[ ! -f "$dest_dir/pytorch_model.bin" ]]; then
    echo "Skip safetensors (missing pytorch_model.bin): $dest_dir" >&2
    return 0
  fi
  echo "Converting pytorch_model.bin → model.safetensors in $dest_dir ..."
  python3 - <<PY
from pathlib import Path
import torch
from safetensors.torch import save_file

dest = Path("${dest_dir}")
bin_path = dest / "pytorch_model.bin"
out = dest / "model.safetensors"
state = torch.load(bin_path, map_location="cpu", weights_only=True)
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
# Clone so tied BERT weights (embeddings ↔ decoder) are not shared-memory views;
# safetensors refuses to save shared tensors.
clean = {
    k: v.detach().cpu().contiguous().clone()
    for k, v in state.items()
}
save_file(clean, str(out))
if not out.is_file():
    raise SystemExit(f"Failed to write {out}")
# safetensors.save_file uses mode 0600; image bake must be readable by the
# non-root Compose user when the entrypoint seeds into the bind mount.
out.chmod(0o644)
print(f"Wrote {out}")
PY
  if [[ "$PRUNE_BIN" -eq 1 && -f "$dest_dir/pytorch_model.bin" ]]; then
    rm -f "$dest_dir/pytorch_model.bin"
    echo "Pruned pytorch_model.bin (--prune-bin): $dest_dir"
  fi
}

download_drive_zip() {
  local drive_id="$1"
  local dest_dir="$2"
  local label="$3"
  local tmp_zip tmp_extract

  if [[ -f "$dest_dir/config.json" && -f "$dest_dir/pytorch_model.bin" ]]; then
    echo "Already present: $dest_dir"
    ensure_safetensors "$dest_dir"
    return 0
  fi

  ensure_gdown
  tmp_zip="$(mktemp "$MODEL_ROOT/${label}.XXXXXX.zip")"
  tmp_extract="$(mktemp -d "$MODEL_ROOT/${label}.XXXXXX.extract")"
  cleanup() {
    rm -f "$tmp_zip"
    rm -rf "$tmp_extract"
  }
  trap cleanup EXIT

  echo "Downloading $label from Google Drive ($drive_id)..."
  python3 - <<PY
import gdown
gdown.download("https://drive.google.com/uc?id=${drive_id}", "${tmp_zip}", quiet=False)
PY

  python3 - <<PY
import shutil, zipfile
from pathlib import Path
extract = Path("${tmp_extract}")
dest = Path("${dest_dir}")
with zipfile.ZipFile("${tmp_zip}") as zf:
    zf.extractall(extract)
cfgs = list(extract.rglob("config.json"))
if not cfgs:
    raise SystemExit("No config.json found in downloaded archive for ${label}")
src = cfgs[0].parent
if dest.exists():
    shutil.rmtree(dest)
shutil.move(str(src), str(dest))
print(f"Installed ${label} → {dest}")
PY

  trap - EXIT
  cleanup
  ensure_safetensors "$dest_dir"
}

if [[ "$DO_BASE" -eq 1 ]]; then
  download_drive_zip "$BASE_DRIVE_ID" "$BASE_DIR" "mouse-Geneformer"
fi

if [[ "$DO_12L" -eq 1 ]]; then
  download_drive_zip "$L12_DRIVE_ID" "$L12_DIR" "mouse-Geneformer-12L-E20"
fi

download_hf_file() {
  local url="$1"
  local dest="$2"
  if [[ -f "$dest" ]]; then
    echo "Already present: $dest"
    return 0
  fi
  echo "Downloading $(basename "$dest")..."
  if command -v wget >/dev/null 2>&1; then
    wget -q -O "$dest" "$url" || { rm -f "$dest"; return 1; }
  else
    curl -fsSL -o "$dest" "$url" || { rm -f "$dest"; return 1; }
  fi
  if [[ ! -s "$dest" ]]; then
    echo "ERROR: empty download: $dest" >&2
    rm -f "$dest"
    return 1
  fi
}

if [[ "$DO_DICTS" -eq 1 ]]; then
  echo "Downloading Mouse-Genecorpus-20M dictionaries..."
  download_hf_file "$TOKEN_DICT_URL" "$DICT_DIR/MLM-re_token_dictionary_v1.pkl" \
    || { echo "ERROR: failed to download token dictionary" >&2; exit 1; }
  download_hf_file "$MEDIAN_DICT_URL" "$DICT_DIR/mouse_gene_median_dictionary.pkl" \
    || { echo "ERROR: failed to download gene median dictionary" >&2; exit 1; }
  download_hf_file "$SYMBOL_DICT_URL" "$DICT_DIR/MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl" \
    || { echo "ERROR: failed to download GeneSymbol→Ensembl dictionary" >&2; exit 1; }
fi

echo "Done."
ls -la "$BASE_DIR" "$L12_DIR" "$DICT_DIR" 2>/dev/null || true
