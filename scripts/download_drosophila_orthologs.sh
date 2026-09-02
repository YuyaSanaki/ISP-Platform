#!/usr/bin/env bash
# Download Drosophila → human and Drosophila → mouse Ensembl ortholog pairs via BioMart.
# Outputs:
#   core/geneformer/dicts/orthologs/drosophila_to_human.tsv
#   core/geneformer/dicts/orthologs/drosophila_to_mouse.tsv
#   columns: source_id, target_id, orthology_type
# Symbol → FBgn aliases: core/geneformer/dicts/drosophila/fly_symbol_to_fbgn.tsv
#   (p53/Tp53→FBgn0039044, Brca2→FBgn0050169; do not invent mammal-only names like Igfbp2)
# Curated overrides: drosophila_to_*_curated.tsv (merged after main tables; keep empty unless verified)
#
# Many-to-many handling is applied at load time via species.ortholog_policy.
set -euo pipefail
ROOT="${GENEFORMER_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT_HUMAN="${ROOT}/core/geneformer/dicts/orthologs/drosophila_to_human.tsv"
OUT_MOUSE="${ROOT}/core/geneformer/dicts/orthologs/drosophila_to_mouse.tsv"
mkdir -p "$(dirname "$OUT_HUMAN")" "${ROOT}/core/geneformer/dicts/drosophila"
QUERY_FILE="$(mktemp)"
TMP="$(mktemp)"
trap 'rm -f "${QUERY_FILE}" "${TMP}"' EXIT

cat > "${QUERY_FILE}" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">
  <Dataset name="dmelanogaster_gene_ensembl" interface="default">
    <Attribute name="ensembl_gene_id"/>
    <Attribute name="hsapiens_homolog_ensembl_gene"/>
    <Attribute name="hsapiens_homolog_orthology_type"/>
    <Attribute name="mmusculus_homolog_ensembl_gene"/>
    <Attribute name="mmusculus_homolog_orthology_type"/>
  </Dataset>
</Query>
EOF

# Ensembl BioMart is occasionally flaky (405 / empty / timeout). Try mirrors.
BIOMART_URLS=(
  "https://www.ensembl.org/biomart/martservice"
  "https://useast.ensembl.org/biomart/martservice"
  "https://asia.ensembl.org/biomart/martservice"
)

echo "Fetching fly→human and fly→mouse orthologs from Ensembl BioMart..."
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

# Keep all BioMart rows; species.ortholog_policy filters at load time.
# sort -u: BioMart can emit duplicate (gene, homolog, type) rows.
{
  echo -e "source_id\ttarget_id\torthology_type"
  awk -F'\t' 'NR>1 {
    gsub(/\r$/, "", $1); gsub(/\r$/, "", $2); gsub(/\r$/, "", $3)
    split($1, f, ".")
    split($2, h, ".")
    src = f[1]
    tgt = h[1]
    otype = $3
    if (src != "" && tgt != "" && tgt != "nan") print src "\t" tgt "\t" otype
  }' "${TMP}" | sort -u
} > "${OUT_HUMAN}.new"

{
  echo -e "source_id\ttarget_id\torthology_type"
  awk -F'\t' 'NR>1 {
    gsub(/\r$/, "", $1); gsub(/\r$/, "", $4); gsub(/\r$/, "", $5)
    split($1, f, ".")
    split($4, m, ".")
    src = f[1]
    tgt = m[1]
    otype = $5
    if (src != "" && tgt != "" && tgt != "nan") print src "\t" tgt "\t" otype
  }' "${TMP}" | sort -u
} > "${OUT_MOUSE}.new"

LINES_H=$(($(wc -l < "${OUT_HUMAN}.new") - 1))
LINES_M=$(($(wc -l < "${OUT_MOUSE}.new") - 1))
MIN_PAIRS=500
if [[ "${LINES_H}" -lt "${MIN_PAIRS}" ]] || [[ "${LINES_M}" -lt "${MIN_PAIRS}" ]]; then
  echo "ERROR: expected at least ${MIN_PAIRS} fly ortholog pairs; got human=${LINES_H} mouse=${LINES_M}" >&2
  head -5 "${TMP}" >&2 || true
  rm -f "${OUT_HUMAN}.new" "${OUT_MOUSE}.new"
  exit 1
fi

mv "${OUT_HUMAN}.new" "${OUT_HUMAN}"
mv "${OUT_MOUSE}.new" "${OUT_MOUSE}"
echo "Wrote ${LINES_H} fly→human pairs to ${OUT_HUMAN}"
echo "Wrote ${LINES_M} fly→mouse pairs to ${OUT_MOUSE}"
echo "Curated overrides: ${ROOT}/core/geneformer/dicts/orthologs/drosophila_to_*_curated.tsv"
echo "Fly symbols:         ${ROOT}/core/geneformer/dicts/drosophila/fly_symbol_to_fbgn.tsv"
echo "Note: species.ortholog_policy filters many-to-many at load (default: one2one)."
