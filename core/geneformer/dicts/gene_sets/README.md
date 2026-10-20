# Gene set libraries (offline enrichment)

Gene set libraries for the over-representation analysis of ISP rank enrichment
(`core/gene_set_enrichment.py`, `core/run_isp_rank_enrichment.py`). Enrichment runs on
this machine: gene lists are never sent to Enrichr or any other web service.

| File | Library | Used for |
|------|---------|----------|
| `GO_Biological_Process_2025.gmt` | GO Biological Process 2025 | human and mouse models |
| `GO_Molecular_Function_2025.gmt` | GO Molecular Function 2025 | human and mouse models |
| `GO_Cellular_Component_2025.gmt` | GO Cellular Component 2025 | human and mouse models |
| `Reactome_Pathways_2024.gmt` | Reactome Pathways 2024 | human and mouse models |
| `KEGG_2021_Human.gmt` | KEGG 2021 Human | human model |
| `KEGG_2019_Mouse.gmt` | KEGG 2019 Mouse | mouse model |

GO Biological Process, Reactome and KEGG are the Enrichr libraries used for the paper's
enrichment (Fig. 2). GO Molecular Function and Cellular Component were added in v1.4.0. The `.gmt` files are
not in git (KEGG does not allow redistribution). `scripts/download_gene_sets.sh` downloads them
from Enrichr by library name only and checks them against `SHA256SUMS` (retrieved 2026-10-14;
`libraries.json` records the source). The Docker image downloads them at build time.

Symbols in the libraries are uppercase. Mouse model genes are matched after upper-casing, as
Enrichr does for mouse gene lists.

Verify: `cd core/geneformer/dicts/gene_sets && sha256sum -c SHA256SUMS` (macOS: `shasum -a 256 -c SHA256SUMS`).
