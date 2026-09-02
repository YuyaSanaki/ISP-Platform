# Ortholog mapping tables

Tab-separated `source_id` → `target_id` mappings used by `geneformer.gene_converter`.
Optional third column `orthology_type` (from BioMart) is used when present.

| File | Use |
|------|-----|
| `mouse_to_human.tsv` | Full Ensembl mouse→human map (`scripts/download_mouse_human_orthologs.sh`; includes `orthology_type`) |
| `mouse_to_human_curated.tsv` | Symbol aliases + corrected Ensembl pairs (merged **after** main table; overrides BioMart) |
| `human_to_mouse.tsv` | Human→mouse Ensembl pairs (same download; includes `orthology_type`) |
| `human_to_mouse_curated.tsv` | Human symbol aliases → mouse Ensembl |
| `drosophila_to_*.tsv` | Full fly→human / fly→mouse maps (`scripts/download_drosophila_orthologs.sh`; includes `orthology_type`) |
| `core/geneformer/dicts/drosophila/fly_symbol_to_fbgn.tsv` | Fly symbol → `FBgn` for ISP (`p53`/`Tp53`→`FBgn0039044`, `Brca2`→`FBgn0050169`) |
| `*_curated.tsv` | Same merge rule for every conversion pair |

**Fly aliases:** do not invent mammal gene names (`Igfbp2`) for fly IDs. `FBgn0283477` is SF2, not IGFBP2; `FBgn0004644` is hedgehog (`hh`), not p53. True fly p53 (`FBgn0039044`) is BioMart `ortholog_one2many` to the p53 family → dropped under default `one2one`.

## Quick pick (`species.ortholog_policy`)

Only used when **input species ≠ model species**. Same-species runs ignore it.

| Value | Pick when… | Effect |
|-------|------------|--------|
| `one2one` (**default**) | Unsure / normal analysis | Keep clear 1:1 only; never sum counts |
| `best_of_n` | Want fewer genes dropped | Among N→1, keep highest-median gene |
| `legacy_sum` | Reproducing an old run | Old last-write-wins + expression sum |

Web UI: **Species / model** controls → dropdown appears only for cross-species runs  
YAML: `species.ortholog_policy` in `core/config/pipeline.yaml`

Many-to-many BioMart rows are resolved at **load / remap** time:

| Policy | 1→N (one source, many targets) | N→1 (many sources, one target) |
|--------|--------------------------------|--------------------------------|
| `one2one` (**default**) | Drop ambiguous sources | Drop ambiguous primary IDs; **never sum** counts |
| `best_of_n` | Deterministic pick (lexicographically smallest target) | Keep source with **highest median** expression; no sum |
| `legacy_sum` | Last-write-wins | **Sum** expression (old behavior; distorts Geneformer ranks) |

```yaml
species:
  model_organism: mouse
  model: human_geneformer
  ortholog_policy: one2one   # one2one | best_of_n | legacy_sum
```

When `orthology_type` is present in the TSV, `one2one` prefers rows labeled `ortholog_one2one`, then still requires unique source/target among primary IDs.

## Resolution order (`resolve_gene_for_model`)

1. Ortholog table on user input (symbol or Ensembl — curated wins over main).
2. Input symbol → input Ensembl (species symbol dict) → ortholog.
3. Passthrough if already model-native Ensembl / `FBgn`.

**Igfbp2 / IGFBP2 (mouse↔human only):** curated maps to human `ENSG00000115457` and mouse `ENSMUSG00000039323` so tokenization, ISP, and UMAP stay consistent (BioMart Ensembl-only pairs may differ). There is no fly `Igfbp2` alias.

**Gapdh / GAPDH:** curated keeps `ENSMUSG00000057666` ↔ `ENSG00000111640`. Without it, one2one drops mouse Gapdh (collision with `Gm*` rows) and human→mouse can resolve to `Gm10358`.

**Curated Ensembl overrides:** only add an Ensembl→Ensembl row when BioMart is wrong for that gene. Symbol aliases (e.g. `Tp53` / `Trp53` / `Brca2` / `Gapdh`) are fine; they must point at the correct target IDs and must not remap unrelated genes (e.g. Lypla1 / Maoa / Gnai3).

## Project curated overlays (analysis-scoped)

Platform `*_curated.tsv` files stay global. Analysis projects may add an **explicit overlay** without editing them:

| Mechanism | Example |
|-----------|---------|
| YAML | `species.ortholog_curated_overlay: /app/analysis/.../ortholog_policy/v1` |
| Env | `GENEFORMER_ORTHOLOG_CURATED_OVERLAY=...` |
| CLI | `--ortholog-curated-overlay ...` (tokenize / pipeline) |

If the path is a **directory**, the loader picks `curated_bridge_{pair}.tsv` (e.g. `curated_bridge_human_to_mouse.tsv`). Overlay rows are merged **after** the platform curated TSV. Prefer Ensembl ID→ID rows for reproducibility.

## Ortholog loss gate

Tokenize can run `ortholog_loss_gate` when `tokenizer.ortholog_audit` is set. **Block** = critical gene **present in input** but dropped by policy (e.g. POU5F1 one2many under one2one). Absent-from-input criticals are **Warn** only. The gate does **not** auto-write overlay / curated rows — use an explicit overlay + approval choice **B**.

Operator docs: [docs/tokenization.md](../../../../docs/tokenization.md#ortholog-loss-gate).
