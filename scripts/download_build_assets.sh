#!/usr/bin/env bash
# Download pretrained weights + dictionaries for Docker build / offline install.
#
# Dest root (models/ + core/geneformer/dicts/):
#   GENEFORMER_ROOT=/opt/geneformer-assets  (Docker image bake)
#   or repo root (default)
#
# Profiles (DOWNLOAD_MODELS):
#   default  — mouse base + 12L-E20 + human V2-104M + mouse/human dicts +
#              mouse↔human orthologs (enough for default WebUI / pipeline / ISP /
#              finetune across Platform species selectors). ~0.6–0.8 GB.
#   minimal  — mouse base + mouse dicts only (~80–140 MB).
#   all      — default + fly ortholog tables (BioMart; slower).
#   none     — no-op (Dockerfile dry-run / CI without network).
#
# Optional:
#   HF_TOKEN / HUGGING_FACE_HUB_TOKEN — if HF rate-limits anonymous downloads.
#   SKIP_ORTHOLOGS=1 — skip BioMart ortholog fetch even in default/all.
#
# Failures exit non-zero with a clear message (set -e).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export GENEFORMER_ROOT="${GENEFORMER_ROOT:-$REPO_ROOT}"
PROFILE="${DOWNLOAD_MODELS:-default}"

mkdir -p \
  "$GENEFORMER_ROOT/models" \
  "$GENEFORMER_ROOT/core/geneformer/dicts/mouse" \
  "$GENEFORMER_ROOT/core/geneformer/dicts/human" \
  "$GENEFORMER_ROOT/core/geneformer/dicts/orthologs" \
  "$GENEFORMER_ROOT/core/geneformer/dicts/drosophila"

# Ship tiny curated overrides with the image even when BioMart is skipped.
seed_curated_from_repo() {
  local src dst
  for src in \
    "$REPO_ROOT/core/geneformer/dicts/orthologs/"*_curated.tsv \
    "$REPO_ROOT/core/geneformer/dicts/drosophila/fly_symbol_to_fbgn.tsv"
  do
    [[ -f "$src" ]] || continue
    dst="$GENEFORMER_ROOT/${src#"$REPO_ROOT"/}"
    mkdir -p "$(dirname "$dst")"
    if [[ ! -f "$dst" ]]; then
      cp -a "$src" "$dst"
      echo "Seeded curated: $dst"
    fi
  done
}

echo "=== Geneformer build assets ==="
echo "GENEFORMER_ROOT=$GENEFORMER_ROOT"
echo "DOWNLOAD_MODELS=$PROFILE"

case "$PROFILE" in
  default|minimal|all|none) ;;
  *)
    echo "ERROR: unknown DOWNLOAD_MODELS='$PROFILE' (use default|minimal|all|none)" >&2
    exit 1
    ;;
esac

if [[ "$PROFILE" == "none" ]]; then
  echo "Skipping downloads (DOWNLOAD_MODELS=none)."
  seed_curated_from_repo
  exit 0
fi

# gdown + huggingface_hub used by the stage scripts
python3 -c "import gdown" >/dev/null 2>&1 || pip install -q gdown
python3 -c "import huggingface_hub" >/dev/null 2>&1 || pip install -q "huggingface_hub>=0.23"

seed_curated_from_repo

case "$PROFILE" in
  minimal)
    bash "$SCRIPT_DIR/download_mouse_geneformer.sh" --base --dicts --prune-bin
    ;;
  default|all)
    bash "$SCRIPT_DIR/download_mouse_geneformer.sh" --prune-bin
    bash "$SCRIPT_DIR/download_human_geneformer_v2_104m.sh"
    # Drop HF download cache to keep the image lean
    rm -rf "$GENEFORMER_ROOT/models/_hf_cache"
    if [[ "${SKIP_ORTHOLOGS:-0}" != "1" ]]; then
      bash "$SCRIPT_DIR/download_mouse_human_orthologs.sh"
      if [[ "$PROFILE" == "all" ]]; then
        bash "$SCRIPT_DIR/download_drosophila_orthologs.sh"
      fi
    else
      echo "SKIP_ORTHOLOGS=1 — BioMart ortholog tables not fetched."
    fi
    ;;
esac

verify() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "ERROR: expected asset missing after download: $path" >&2
    exit 1
  fi
}

verify_model() {
  local dir="$1"
  verify "$dir/config.json"
  if [[ ! -f "$dir/model.safetensors" && ! -f "$dir/pytorch_model.bin" ]]; then
    echo "ERROR: no weights in $dir (need model.safetensors or pytorch_model.bin)" >&2
    exit 1
  fi
}

echo "Verifying required assets..."
verify "$GENEFORMER_ROOT/core/geneformer/dicts/mouse/MLM-re_token_dictionary_v1.pkl"
verify "$GENEFORMER_ROOT/core/geneformer/dicts/mouse/mouse_gene_median_dictionary.pkl"
verify "$GENEFORMER_ROOT/core/geneformer/dicts/mouse/MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl"
verify_model "$GENEFORMER_ROOT/models/mouse-Geneformer"

if [[ "$PROFILE" != "minimal" ]]; then
  verify_model "$GENEFORMER_ROOT/models/mouse-Geneformer-12L-E20"
  verify_model "$GENEFORMER_ROOT/models/human-Geneformer-V2-104M"
  verify "$GENEFORMER_ROOT/core/geneformer/dicts/human/token_dictionary_gc104M.pkl"
  verify "$GENEFORMER_ROOT/core/geneformer/dicts/human/gene_median_dictionary_gc104M.pkl"
  if [[ "${SKIP_ORTHOLOGS:-0}" != "1" ]]; then
    verify "$GENEFORMER_ROOT/core/geneformer/dicts/orthologs/mouse_to_human.tsv"
    verify "$GENEFORMER_ROOT/core/geneformer/dicts/orthologs/human_to_mouse.tsv"
  fi
fi

if [[ "$PROFILE" == "all" && "${SKIP_ORTHOLOGS:-0}" != "1" ]]; then
  verify "$GENEFORMER_ROOT/core/geneformer/dicts/orthologs/drosophila_to_human.tsv"
  verify "$GENEFORMER_ROOT/core/geneformer/dicts/orthologs/drosophila_to_mouse.tsv"
fi

# Non-root Compose runtime (DOCKER_UID) must be able to read baked assets when
# the entrypoint seeds into the bind-mounted /app tree.
chmod -R a+rX "$GENEFORMER_ROOT/models" "$GENEFORMER_ROOT/core/geneformer/dicts" 2>/dev/null || true

echo "=== Build assets ready under $GENEFORMER_ROOT ==="
du -sh \
  "$GENEFORMER_ROOT/models" \
  "$GENEFORMER_ROOT/core/geneformer/dicts" \
  2>/dev/null || true
