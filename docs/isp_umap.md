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
| `postprocess.enabled` | Optional `cluster_coexpr_analysis/` after the main UMAP (**default `false`**). |
| `postprocess.n_clusters` | KMeans clusters when no `cluster` column. Integer (>=2) or `auto` (silhouette over k=2..15; default `4`). |
| `postprocess.celltype_prediction` | Marker-gene labels on start-state cells (default `true` when postprocess is on). |
| `postprocess.prefer_metadata_celltype` | `auto` (default): use pre-ISP platform `cell_type` when `celltype_annotator` is `isp_expression*`; `true`: trust any dataset `cell_type`; `false`: marker scoring only. |
| `postprocess.celltype_rank_weights` | Rank-weight marker hits (earlier Geneformer ranks = higher expression; default `true`). |
| `postprocess.celltype_negative_markers` | Subtract negative-marker scores to sharpen boundaries (default `true`). |
| `postprocess.celltype_negative_weight` | Penalty weight for negatives (default `0.55`). |

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

### Downstream plots (`cluster_coexpr_analysis/`, optional)

After the main UMAP finishes, `run_isp_umap.py` can write joint overlays and L2-by-group figures under `{run-dir}/cluster_coexpr_analysis/`. This is **off by default** (`postprocess.enabled: false`). Enable via YAML, Web UI **Cluster / cell-type analysis**, or `--enable-postprocess`.

1. **Cell-type prediction** — when postprocess cell-type is on, **pre-ISP platform labels** (`cell_type` written at tokenize with `celltype_annotator=isp_expression_v1`) are preferred (`prefer_metadata_celltype: auto`). If those columns are absent, species-aware marker panels (mouse / human, matched to the Geneformer backend) are scored on start-state `input_ids` → `pred_cell_type` / `coarse_type` / `celltype_plot`. Raw user-filled `cell_type` without platform provenance is still ignored unless `prefer_metadata_celltype: true`. Rank weighting prefers markers that appear early in the Geneformer rank list.
2. **Joint UMAP overlays** — `umap_joint_l2_cluster_celltype.png` (+ enriched CSV, `joint_umap_coords.npy`).
3. **L2 by group** — `l2_by_coarse_celltype.png` and `l2_mean_by_coarse_celltype.png`.
4. **Cell-type trajectory tracking** — `umap_celltype_trajectories.png` (arrows + mean displacement vectors colored by cell type), `celltype_shift_summary.csv`, and `l2_mean_by_pred_celltype.png`.

Failures in postprocess are logged as warnings; core UMAP outputs stay intact.

### Cell-type classification (accuracy / cross-species)

Marker panels live in [`core/isp_umap_celltype.py`](../core/isp_umap_celltype.py):

| Backend | Panel |
|---------|--------|
| `mouse_geneformer` | Mouse brain markers (title-case symbols, e.g. `Cx3cr1`) |
| `human_geneformer` | Human HGNC orthologs (uppercase, e.g. `CX3CR1`) |

Because tokenization remaps genes into the **model** vocabulary, panels are chosen by the model-native organism—not the input species. Cross-species runs (e.g. mouse data → human Geneformer) therefore use the **curated human panel** (mouse-only genes like `Ly6c1` dropped; human-preferable markers such as `CDH5`/`VWF`/`CSF1R` added). Unresolved symbols are dropped from the panel; types with fewer than 2 resolved positives are skipped. **Negative markers** (e.g. microglia markers against SMC) subtract from the score to reduce boundary mix-ups.

**Pre-ISP expression annotation:** tokenize (`tokenizer.celltype_annotation`, default on) scores curated whole-body marker panels on the count matrix before loom write (`core/celltype_annotate_expression.py`, panel `isp_expression_v1.json`). Those labels propagate into the HF dataset. When ISP UMAP cluster/cell-type postprocess is on, `prefer_metadata_celltype: auto` (default) uses that platform metadata; set `false` to force marker scoring on tokens, or `true` for any trusted external `cell_type`.

## Outputs

All generated assets are safely routed to the `output/[DATE]/isp_umap_[UTC TIME]` directory.

| File | Role |
|------|------|
| **`per_cell_isp_shift.csv`** | **Essential:** per-cell perturbation magnitude and direction (see below) |
| `umap_*.png` | Visual summary (white L-axes, Fig.2 endpoint style); arrows = same cells as `umap_shift_l2` in the CSV |
| `*_embs.npy` | Raw embedding matrices for custom downstream analysis |
| `cluster_coexpr_analysis/*` | Optional (when `postprocess.enabled`): joint overlays, L2-by-group, **cell-type trajectories**, enriched CSV |

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
