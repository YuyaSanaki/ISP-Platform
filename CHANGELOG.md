# Changelog

All notable changes to **ISP³ Platform** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- ISP UMAP postprocess **cell-type trajectory tracking**: `umap_celltype_trajectories.png` (per-cell arrows + mean displacement by type), `celltype_shift_summary.csv`, `l2_mean_by_pred_celltype.png`.
- Script `scripts/validate_asano_igfbp2_celltype_tracking.py` — Asano PIPseq (1w) Igfbp2 delete validation for mouse + human Geneformer with the new cell-type tracking plots.

### Changed

- ISP UMAP marker cell-type prediction is **species-aware** (mouse vs human panels matched to the Geneformer backend), with optional **rank-weighted** marker scoring (`celltype_rank_weights`). Dataset `cell_type` metadata is **ignored by default** (`prefer_metadata_celltype: false`).
- Marker panels: **human panel curated** for cross-species (drop mouse-only genes; add CDH5/VWF/CSF1R/…); **negative markers** subtract from scores (`celltype_negative_markers`, default on).
- Pipeline E2E TOP1 ISP UMAP now honors `stages.isp.postprocess.enabled` (`--enable-postprocess`).

## [1.2.0] - 2026-09-20

### Changed

- E2E Pipeline auto ISP UMAP now selects the **TOP1 significant** gene (`significant_genes.csv` / parquet `Sig==1`), not the raw top positive shifter.

### Added

- Optional ISP UMAP postprocess **`cluster_coexpr_analysis`** (default **off**): marker cell-type labels, joint UMAP overlays (L2 / KMeans cluster / cell type), and L2-by-group box/mean figures.
- Config: `postprocess.enabled` / `n_clusters` / `celltype_prediction` in `isp_umap.yaml` and `stages.isp.postprocess` (pipeline → E2E TOP1 UMAP).
- Web UI: **Cluster / cell-type analysis** on ISP UMAP Plot options and Pipeline Advanced options (`n_clusters`: fixed K or `auto`).
- CLI: `--enable-postprocess` / `--skip-postprocess` / `--postprocess-n-clusters` / `--postprocess-celltype`.

### Notes

- Existing 1.0 / 1.1 runs are unchanged until `postprocess.enabled` is set true.
- Docker image / Compose default tag is `isp-platform:v1.2.0`.

## [1.1.0] - 2026-09-18

### Changed

- Fine-tune default warmup is **500 steps** (`warmup_ratio: null`) so virtual genetic screens (ISP gene-effect size) match the Apr-22 Asano schedule. v1.0.0 used `warmup_ratio: 0.05`, which became ~2000 steps on a 10-epoch Asano-scale run and compressed ISP cosine shifts.
- Web UI **Advanced options** can switch warmup between **500 steps** (virtual genetic screen) and **rate 0.05** (classification-oriented FT).
- Docker image / Compose default tag is `isp-platform:v1.1.0`.
- Short smoke-matrix configs still set `warmup_ratio: 0.05` so 1-epoch smoke FT is not dominated by a fixed 500-step warmup (the 10% cap still applies).

### Notes

- Pin analysis repos that need the previous schedule to branch **`ver1.0.0`** / image `isp-platform:v1.0.0`.

## [1.0.0] - 2026-09-02

### Added

- Initial public release split from the private analysis monolith.
- End-to-end pipeline: tokenize → fine-tune → ISP (+ optional UMAP).
- Mouse and Human Geneformer backends with cross-species ortholog conversion.
- Web UI (Streamlit) and Docker Compose CLI.
- Sequential multi-gene ISP.
- Ortholog loss gate, conversion reports, and smoke-matrix test harness.

### Notes

- Private paper-specific workflows live in **ISP³ Platform Analysis** (separate repo).
- Analysis repos should pin runtime to Docker image `isp-platform:v1.0.0`.
