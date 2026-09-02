#!/bin/sh
# OpenMP / sklearn workaround: only on 64-bit ARM Linux (e.g. DGX Spark).
# x86_64 and other arches: leave LD_PRELOAD unset.
# Override: set LD_PRELOAD in the environment before start (docker compose / -e).
if [ -z "${LD_PRELOAD:-}" ] && [ "$(uname -m)" = "aarch64" ]; then
  _sys_gomp=/usr/lib/aarch64-linux-gnu/libgomp.so.1
  _sklearn_gomp=$(ls /usr/local/lib/python3.12/dist-packages/scikit_learn.libs/libgomp-*.so.* 2>/dev/null | head -1)
  if [ -f "$_sys_gomp" ] && [ -n "$_sklearn_gomp" ]; then
    export LD_PRELOAD="$_sys_gomp:$_sklearn_gomp"
  fi
fi

# Seed image-baked models/dicts into the bind-mounted /app tree when missing.
# Compose uses `.:/app`, which would otherwise hide /app/models from the image.
# Host files always win: we only copy when the destination is absent/incomplete.
# Disable: SKIP_MODEL_SEED=1
# Bake location: GENEFORMER_IMAGE_ASSETS (default /opt/geneformer-assets)
_seed_geneformer_assets() {
  if [ "${SKIP_MODEL_SEED:-0}" = "1" ]; then
    return 0
  fi
  _assets="${GENEFORMER_IMAGE_ASSETS:-/opt/geneformer-assets}"
  if [ ! -d "$_assets" ]; then
    return 0
  fi

  _model_ready() {
    _d="$1"
    [ -f "$_d/config.json" ] || return 1
    [ -f "$_d/model.safetensors" ] || [ -f "$_d/pytorch_model.bin" ]
  }

  if [ -d "$_assets/models" ]; then
    mkdir -p /app/models
    for _src in "$_assets/models"/*; do
      [ -d "$_src" ] || continue
      _name=$(basename "$_src")
      case "$_name" in
        _*) continue ;;
      esac
      _dst="/app/models/$_name"
      if _model_ready "$_dst"; then
        continue
      fi
      echo "Seeding model $_name → $_dst (from image assets)"
      rm -rf "$_dst"
      cp -a "$_src" "$_dst"
    done
  fi

  if [ -d "$_assets/core/geneformer/dicts" ]; then
    mkdir -p /app/core/geneformer/dicts
    # Copy missing files only (preserve host overrides / partial local dicts).
    find "$_assets/core/geneformer/dicts" -type f | while read -r _src; do
      _rel="${_src#$_assets/core/geneformer/dicts/}"
      _dst="/app/core/geneformer/dicts/$_rel"
      if [ -f "$_dst" ]; then
        continue
      fi
      mkdir -p "$(dirname "$_dst")"
      cp -a "$_src" "$_dst"
      echo "Seeding dict $_rel"
    done
  fi
}

_seed_geneformer_assets

exec "$@"
