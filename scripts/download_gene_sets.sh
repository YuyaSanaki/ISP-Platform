#!/usr/bin/env bash
# Download the Enrichr gene set libraries used by ISP rank enrichment and verify them.
# Output: core/geneformer/dicts/gene_sets/<library>.gmt (checked against SHA256SUMS).
#
# Only the library name is sent to Enrichr. Enrichment itself runs locally
# (core/gene_set_enrichment.py); gene lists never leave the machine.
#
# Files already present and matching SHA256SUMS are kept (no network access).
# GENE_SETS_ALLOW_MISMATCH=1 keeps a download whose checksum differs from SHA256SUMS
# (Enrichr changed the library; results will differ from the pinned version).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${GENEFORMER_ROOT:-$REPO_ROOT}"
PINNED_DIR="${REPO_ROOT}/core/geneformer/dicts/gene_sets"
OUT_DIR="${ROOT}/core/geneformer/dicts/gene_sets"
URL="https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName="
mkdir -p "${OUT_DIR}"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

for f in SHA256SUMS libraries.json README.md; do
  if [[ "${PINNED_DIR}" != "${OUT_DIR}" && -f "${PINNED_DIR}/${f}" ]]; then
    cp -f "${PINNED_DIR}/${f}" "${OUT_DIR}/${f}"
  fi
done

status=0
while read -r want name; do
  [[ -n "${name:-}" ]] || continue
  dest="${OUT_DIR}/${name}"
  lib="${name%.gmt}"
  if [[ -f "${dest}" && "$(sha256_of "${dest}")" == "${want}" ]]; then
    echo "OK      ${name}"
    continue
  fi
  tmp="$(mktemp)"
  echo "Fetch   ${lib}"
  if ! curl -fsSL --retry 3 --retry-delay 5 --max-time 600 -o "${tmp}" "${URL}${lib}"; then
    echo "ERROR: download failed: ${lib}" >&2
    rm -f "${tmp}"
    status=1
    continue
  fi
  got="$(sha256_of "${tmp}")"
  if [[ "${got}" != "${want}" ]]; then
    if [[ "${GENE_SETS_ALLOW_MISMATCH:-0}" == "1" ]]; then
      echo "WARNING: ${name} differs from SHA256SUMS (got ${got}); kept because GENE_SETS_ALLOW_MISMATCH=1" >&2
    else
      echo "ERROR: ${name} does not match SHA256SUMS (got ${got}, want ${want})." >&2
      echo "Enrichr may have changed the library. Set GENE_SETS_ALLOW_MISMATCH=1 to keep it anyway." >&2
      rm -f "${tmp}"
      status=1
      continue
    fi
  fi
  mv -f "${tmp}" "${dest}"
  chmod a+r "${dest}"
  echo "Saved   ${dest}"
done < "${PINNED_DIR}/SHA256SUMS"

exit "${status}"
