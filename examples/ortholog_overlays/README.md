# Project curated overlay (example)

A project curated overlay adds ortholog pairs for one analysis on top of the platform tables. It is a separate file: do not edit the platform `core/geneformer/dicts/orthologs/*_curated.tsv`, which are pinned by `SHA256SUMS`.

Load order for a cross-species run: pinned Ensembl table → `ortholog_policy` (default `one2one`) → platform `*_curated.tsv` → project overlay (only when selected). An overlay row adds a pair, or replaces the target of the same source gene.

## Files

| File | Direction |
|------|-----------|
| `curated_bridge_human_to_mouse.tsv.example` | human data → Mouse-Geneformer |
| `curated_bridge_mouse_to_human.tsv.example` | mouse data → Human-Geneformer |

Two tab-separated columns, `source_id` and `target_id`, one pair per row. Use Ensembl gene IDs without version suffix. The example row is POU5F1 ↔ Pou5f1 (`ENSG00000204531` ↔ `ENSMUSG00000024406`), which the platform table already restores, so it only shows the format and changes no mapping.

The `.example` suffix keeps these files from being picked up when this folder is selected.

## Use

1. Copy the file for your direction to a project folder and drop `.example`, e.g. `analysis/my_project/ortholog_overlay/curated_bridge_human_to_mouse.tsv`.
2. Replace the example row with the pairs you need. List only established one-to-one orthologs; never map a paralogue (e.g. POU5F1B, NANOGP8) onto the gene's ortholog. The ortholog loss gate rejects an overlay that maps POU5F1B.
3. Select the folder (or the TSV file):
   - Web UI → **Project curated overlay**, as a container path (the repository is `/app`), e.g. `/app/analysis/my_project/ortholog_overlay`;
   - YAML: `species.ortholog_curated_overlay: /app/analysis/my_project/ortholog_overlay`;
   - CLI `--ortholog-curated-overlay PATH`, or env `GENEFORMER_ORTHOLOG_CURATED_OVERLAY`.

Details: [ortholog README](../../core/geneformer/dicts/orthologs/README.md#project-curated-overlays-analysis-scoped), [tokenization.md](../../docs/tokenization.md#cross-species-gene-conversion).
