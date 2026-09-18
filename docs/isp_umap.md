# in-sillico perturbation visualization on UMAP Plot Service

The **ISP UMAP** service calculates and visualizes how **each cell** moves when a gene (or gene set) is perturbed in silico. Unlike standard ISP (gene-level ranking via cosine similarity across the cohort), this service exports **per-cell shift metrics** and a UMAP trajectory plot.

**Embedding readout:** **mean-pooled** non-padding token hidden states — the same `cell_emb` representation used by ISP `goal_state_shift` statistics. Fig.2 endpoint UMAP (`build_day14_endpoint_umap.py --v2-ft`) uses the same readout for cross-figure consistency.

**Core output:** [`per_cell_isp_shift.csv`](#3-per-cell-shift-table-per_cell_isp_shiftcsv) — one row per start-state cell (e.g. `Disease`) with `shift_l2` (embedding-space perturbation magnitude), `shift_toward_<end_state>` (movement toward e.g. `Ctrl`), and UMAP before/after coordinates. Use this table to identify which cells (and later, which cell types after you join annotations) are most affected by the perturbation.

**Note:** UMAP arrow length (`umap_shift_l2`) is the 2D L2 distance in the jointly fitted UMAP plane. It is **not** calibrated to ISP bar-chart `goal_state_shift` (Δ cosine to goal centroid); use UMAP for spatial intuition only.

`max_cells_per_state` is applied **after** filtering start/end state. The cap is spread across `sample_id` in proportion to each sample’s cell count (tiny samples keep ≥1 cell when the budget allows), then shuffled. This avoids taking the first N tokenized cells, which would over-represent whichever loom/h5ad files were read first. Set `umap.sample_key` to `null` for an unstratified shuffle.

## Configuration

The service is fully configured via [`core/config/isp_umap.yaml`](../core/config/isp_umap.yaml). You may modify this file to point to different datasets, fine-tuned models, or target genes.

Key configurations to note:

| YAML area | What to set |
|-----------|-------------|
| `paths.dataset` | Tokenized `.dataset` directory containing your original and condition-annotated cells. |
| `paths.geneformer_model` | Path to your pre-trained or fine-tuned sequence classification model. |
| `umap.show_trajectory_arrows` | Draw Start→Perturbed arrows (`true` / `false`). |
| `umap.num_trajectory_arrows` | Approximate number of arrows when enabled (default `100`). |
| `umap.max_cells_per_state` | Cap cells per start/end state (default `2000`). Stratified by `sample_key`, not prefix-of-N. |
| `umap.sample_key` | Column used to spread the cap across input samples (default `sample_id`). Empty/`null` = shuffle only. |
| `umap.seed` | Seed for cell subsampling, UMAP (and PCA when enabled). Default `42`; use `0` with `pca_components: 50` for Fig.2/3/4. |
| `umap.pca_components` | `0` = direct UMAP on embeddings (default). `50` = PCA(50)→UMAP (Fig.2/3/4 manuscript style). |
| `perturbation.genes_to_perturb` | One or more gene symbols / Ensembl IDs perturbed **together** (group KD/OE). |
| `perturbation.type` | `delete` (KD) or `overexpress` (length-preserving OE; Fig.3 OSKM4-style). |
| `perturbation.state_key` | Label column that divides your cells (e.g., `disease`). |
| `perturbation.start_state` | Condition you are perturbing (e.g. `Disease`). |
| `perturbation.end_state` | Condition you are comparing against (e.g. `Ctrl`). |

### Gene Symbol Auto-Detection

The `gene_to_perturb` parameter supports gene symbols (e.g. `TargetGene`) and Ensembl IDs. Symbols are resolved via internal Geneformer dictionaries (`GENE_NAME_ID_DICTIONARY_FILE`) to the proper `.dataset` token.

## Running

```bash
docker compose run --rm isp_umap
```

From a completed Pipeline run (recommended; reuses that run’s dataset + fine-tuned model):

```bash
python3 core/run_isp_umap.py --run-dir /app/output/YYYYMMDD/pipeline_… --gene Oct4 Sox2 Klf4 Myc
```

`--gene` accepts one or more symbols/IDs; all are perturbed **together** (default). Use `--per-gene` for separate UMAPs per gene. Required when the pipeline ISP was genome-wide (`genes_to_perturb` empty). Outputs go under `{run-dir}/isp_umap/` unless `--output-dir` is set.

Fig.2/3/4-aligned UMAP coordinates (PCA then UMAP, seed 0):

```bash
python3 core/run_isp_umap.py --run-dir /app/output/.../pipeline_... \
  --gene POU5F1 SOX2 KLF4 MYC --pca-components 50 --umap-seed 0
```

## Outputs

All generated assets are safely routed to the `output/[DATE]/isp_umap_[UTC TIME]` directory.

| File | Role |
|------|------|
| **`per_cell_isp_shift.csv`** | **Essential:** per-cell perturbation magnitude and direction (see below) |
| `umap_*.png` | Visual summary (white L-axes, Fig.2 endpoint style); arrows = same cells as `umap_shift_l2` in the CSV |
| `*_embs.npy` | Raw embedding matrices for custom downstream analysis |

### 1. UMAP Figure (`umap_*.png`)
A matplotlib scatter (Fig.2 endpoint style: **white background**, no grid, L-shaped **UMAP-1 / UMAP-2** axes) comparing:
- Target / goal-state cells (e.g. `Ctrl` in green)
- Start-state cells (e.g. `Disease` in ochre)
- ISP-perturbed cells (e.g. `Disease + ISP(TargetGene)` in navy)

Navy arrows track the same start-state cell before vs after perturbation (`umap_shift_l2` in the CSV).

### 2. Raw Embeddings arrays (`*.npy`)
The script exports native `.npy` representations of intermediate embeddings (`Ctrl_embs.npy`, `Disease_embs.npy`, `Disease_ISP..._.npy`) for custom downstream analysis.

### 3. Per-cell shift table (`per_cell_isp_shift.csv`)
One row per **start-state** cell (e.g. each `Disease` cell in the run), with how much that cell moved under ISP:

| Column | Meaning |
|--------|---------|
| `shift_l2` | L2 distance between embeddings before vs after perturbation (larger = more perturbed in model space) |
| `shift_toward_<end_state>` | Reduction in distance to the end-state (e.g. `Ctrl`) centroid; positive values move closer to that reference |
| `umap1_before` / `umap2_before` | UMAP position before perturbation |
| `umap1_after` / `umap2_after` | UMAP position after perturbation |
| `umap_shift_l2` | L2 distance between before/after positions in UMAP space (matches the grey arrows on the plot) |

Dataset metadata columns present on the tokenized `.dataset` (e.g. `sample_id`, `disease`) are included so you can join cell-type labels later after classification.

## Troubleshooting

- **`cuML failed: nvrtc...`**: By default, the script will aim for RAPIDS `cuML` GPU acceleration. If your CUDA hardware architecture is not inherently supported or fails compilation, it will gracefully fallback to multi-threaded CPU `umap-learn` without interrupting execution.
