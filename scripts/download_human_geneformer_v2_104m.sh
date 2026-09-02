#!/usr/bin/env bash
# Download Human Geneformer V2-104M weights and Genecorpus-104M dictionaries.
# Uses `huggingface-cli` when available, otherwise `hf download` / Python API.
set -euo pipefail
ROOT="${GENEFORMER_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL_DIR="$ROOT/models/human-Geneformer-V2-104M"
DICT_DIR="$ROOT/core/geneformer/dicts/human"
CACHE="$ROOT/models/_hf_cache"

mkdir -p "$MODEL_DIR" "$DICT_DIR" "$CACHE"

ensure_hf() {
  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_DL=(huggingface-cli download)
    return 0
  fi
  if command -v hf >/dev/null 2>&1; then
    HF_DL=(hf download)
    return 0
  fi
  echo "Installing huggingface_hub..."
  pip install -q "huggingface_hub>=0.23"
  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_DL=(huggingface-cli download)
  elif command -v hf >/dev/null 2>&1; then
    HF_DL=(hf download)
  else
    echo "ERROR: neither huggingface-cli nor hf found after install." >&2
    exit 1
  fi
}

ensure_hf

echo "Downloading Geneformer-V2-104M weights..."
"${HF_DL[@]}" ctheodoris/Geneformer \
  Geneformer-V2-104M/config.json \
  Geneformer-V2-104M/model.safetensors \
  Geneformer-V2-104M/generation_config.json \
  --local-dir "$CACHE"

cp -f "$CACHE/Geneformer-V2-104M/"* "$MODEL_DIR/"

echo "Downloading Genecorpus-104M dictionaries..."
"${HF_DL[@]}" ctheodoris/Geneformer \
  geneformer/token_dictionary_gc104M.pkl \
  geneformer/gene_median_dictionary_gc104M.pkl \
  geneformer/gene_name_id_dict_gc104M.pkl \
  --local-dir "$CACHE"

cp -f "$CACHE/geneformer/token_dictionary_gc104M.pkl" "$DICT_DIR/"
cp -f "$CACHE/geneformer/gene_median_dictionary_gc104M.pkl" "$DICT_DIR/"
cp -f "$CACHE/geneformer/gene_name_id_dict_gc104M.pkl" "$DICT_DIR/"

echo "Done."
ls -la "$MODEL_DIR" "$DICT_DIR"
