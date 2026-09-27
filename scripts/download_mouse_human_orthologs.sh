#!/usr/bin/env bash
# Install mouse ↔ human Ensembl ortholog tables.
# Output: core/geneformer/dicts/orthologs/{mouse_to_human,human_to_mouse}.tsv
#   columns: source_id, target_id, orthology_type
#
# Default: use the pinned tables shipped in the repository (Ensembl release 116,
# retrieved 2026-08-07; see ensembl_release.json) and verify them against
# SHA256SUMS. No network access.
#
# ORTHOLOG_REFRESH=1: query the live Ensembl BioMart instead. The live release
# changes over time, so the result will usually not match SHA256SUMS and
# conversions will differ from the paper runs.
#
# Symbol aliases and corrected Ensembl pairs (e.g. Igfbp2) remain in
# mouse_to_human_curated.tsv and override BioMart rows when merged at load time.
#
# Many-to-many handling is applied at load time via species.ortholog_policy
# (one2one | best_of_n | legacy_sum). This script keeps all BioMart rows.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${GENEFORMER_ROOT:-$REPO_ROOT}"
PINNED_DIR="${REPO_ROOT}/core/geneformer/dicts/orthologs"
OUT_DIR="${ROOT}/core/geneformer/dicts/orthologs"
OUT="${OUT_DIR}/mouse_to_human.tsv"
mkdir -p "${OUT_DIR}"

if [[ "${ORTHOLOG_REFRESH:-0}" != "1" ]]; then
  for f in SHA256SUMS ensembl_release.json mouse_to_human.tsv human_to_mouse.tsv; do
    if [[ ! -f "${PINNED_DIR}/${f}" ]]; then
      echo "ERROR: pinned ortholog file missing: ${PINNED_DIR}/${f}" >&2
      echo "Restore it from git, or set ORTHOLOG_REFRESH=1 to query live BioMart." >&2
      exit 1
    fi
    if [[ "${PINNED_DIR}" != "${OUT_DIR}" ]]; then
      cp -f "${PINNED_DIR}/${f}" "${OUT_DIR}/${f}"
    fi
  done
  if ! (cd "${OUT_DIR}" && sha256sum --check --strict SHA256SUMS); then
    echo "ERROR: ortholog tables in ${OUT_DIR} do not match SHA256SUMS." >&2
    echo "Restore them with: git checkout -- core/geneformer/dicts/orthologs/" >&2
    exit 1
  fi
  echo "Pinned Ensembl ortholog tables verified (release 116, retrieved 2026-08-07): ${OUT_DIR}"
  exit 0
fi

echo "ORTHOLOG_REFRESH=1 — querying live Ensembl BioMart (tables will not match the pinned release)."
QUERY_FILE="$(mktemp)"
TMP="$(mktemp)"
trap 'rm -f "${QUERY_FILE}" "${TMP}"' EXIT

cat > "${QUERY_FILE}" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">
  <Dataset name="mmusculus_gene_ensembl" interface="default">
    <Filter name="with_hsapiens_homolog" excluded="0"/>
    <Attribute name="ensembl_gene_id"/>
    <Attribute name="hsapiens_homolog_ensembl_gene"/>
    <Attribute name="hsapiens_homolog_orthology_type"/>
  </Dataset>
</Query>
EOF

# Ensembl BioMart is occasionally flaky (405 / empty / timeout). Try mirrors.
BIOMART_URLS=(
  "https://www.ensembl.org/biomart/martservice"
  "https://useast.ensembl.org/biomart/martservice"
  "https://asia.ensembl.org/biomart/martservice"
)

echo "Fetching mouse→human orthologs from Ensembl BioMart (this may take a few minutes)..."
ok=0
for url in "${BIOMART_URLS[@]}"; do
  for attempt in 1 2 3; do
    echo "BioMart POST ${url} (attempt ${attempt})..."
    if curl -fSL --retry 2 --retry-delay 5 --max-time 900 --progress-bar \
      -X POST "${url}" \
      --data-urlencode "query@${QUERY_FILE}" \
      -o "${TMP}" \
      && [[ -s "${TMP}" ]] \
      && ! grep -qiE 'ERROR|Exception|Query ERROR' "${TMP}"
    then
      ok=1
      break 2
    fi
    echo "BioMart attempt failed; sleeping before retry..." >&2
    sleep $((attempt * 5))
  done
done

if [[ "${ok}" -ne 1 ]]; then
  echo "ERROR: BioMart returned an empty or error response from all mirrors." >&2
  head -20 "${TMP}" >&2 || true
  exit 1
fi

if [[ ! -s "${TMP}" ]]; then
  echo "ERROR: BioMart returned an empty file." >&2
  exit 1
fi

{
  echo -e "source_id\ttarget_id\torthology_type"
  awk -F'\t' 'NR>1 {
    gsub(/\r$/, "", $1); gsub(/\r$/, "", $2); gsub(/\r$/, "", $3)
    split($1, a, ".")
    split($2, b, ".")
    otype = $3
    if (a[1] != "" && b[1] != "" && b[1] != "nan") print a[1]"\t"b[1]"\t"otype
  }' "${TMP}" | sort -u
} > "${OUT}.new"

LINES=$(($(wc -l < "${OUT}.new") - 1))
if [[ "${LINES}" -lt 1000 ]]; then
  echo "ERROR: expected thousands of ortholog pairs, got ${LINES}" >&2
  head -5 "${TMP}" >&2 || true
  rm -f "${OUT}.new"
  exit 1
fi

mv "${OUT}.new" "${OUT}"
echo "Wrote ${LINES} Ensembl pairs to ${OUT}"
echo "Curated symbol aliases: ${ROOT}/core/geneformer/dicts/orthologs/mouse_to_human_curated.tsv"

# Reverse human→mouse table: keep all pairs (policy applied at load time).
REV_OUT="${ROOT}/core/geneformer/dicts/orthologs/human_to_mouse.tsv"
{
  echo -e "source_id\ttarget_id\torthology_type"
  awk -F'\t' 'NR>1 && $1 != "" && $2 != "" {
    print $2"\t"$1"\t"$3
  }' "${OUT}"
} > "${REV_OUT}.new"
REV_LINES=$(($(wc -l < "${REV_OUT}.new") - 1))
mv "${REV_OUT}.new" "${REV_OUT}"
echo "Wrote ${REV_LINES} reverse Ensembl pairs to ${REV_OUT}"
echo "Curated reverse aliases: ${ROOT}/core/geneformer/dicts/orthologs/human_to_mouse_curated.tsv"
echo "Note: species.ortholog_policy filters many-to-many at load (default: one2one)."

if [[ -f "${PINNED_DIR}/SHA256SUMS" ]]; then
  if (cd "${OUT_DIR}" && sha256sum --check --status SHA256SUMS 2>/dev/null); then
    echo "Live tables are identical to the pinned release."
  else
    echo "WARNING: live tables differ from the pinned release (SHA256SUMS)." >&2
    echo "  New hashes:" >&2
    (cd "${OUT_DIR}" && sha256sum mouse_to_human.tsv human_to_mouse.tsv) >&2
    echo "  To adopt them, update SHA256SUMS and ensembl_release.json and commit both tables." >&2
  fi
fi
