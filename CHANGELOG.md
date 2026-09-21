# Changelog

All notable changes to **ISP³ Platform** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Work in progress on branch `cursor/pre-isp-celltype-annotation-d63e` (and related ISP UMAP cell-type tracking).

### Added

- **Pre-ISP expression-matrix cell-type annotation** at tokenize (`tokenizer.celltype_annotation`, default **on**):
  - Scores curated whole-body marker panels on the count matrix (`scanpy.tl.score_genes`, with a mean-difference fallback on tiny matrices).
  - Writes `cell_type`, `tissue`, `celltype_score`, `celltype_annotator=isp_expression_v1` onto loom → HF dataset.
  - Implementation: [`core/celltype_annotate_expression.py`](core/celltype_annotate_expression.py), panel [`core/geneformer/dicts/celltype_panels/isp_expression_v1.json`](core/geneformer/dicts/celltype_panels/isp_expression_v1.json).
- ISP UMAP postprocess **cell-type trajectory tracking**: `umap_celltype_trajectories.png`, `celltype_shift_summary.csv`, `l2_mean_by_pred_celltype.png`.
- Script [`scripts/validate_asano_igfbp2_celltype_tracking.py`](scripts/validate_asano_igfbp2_celltype_tracking.py) (+ DGX Spark brief [`scripts/SPARK_AGENT_ASANO_IGFBP2.md`](scripts/SPARK_AGENT_ASANO_IGFBP2.md)).

### Changed

- ISP UMAP `prefer_metadata_celltype` default is **`auto`**: when cluster/cell-type postprocess is on, prefer platform `cell_type` if `celltype_annotator` is `isp_expression*`; otherwise marker scoring on tokens. Use `true` for any trusted external labels, or `false` to force markers.
- Marker cell-type prediction is **species-aware** (mouse vs human panels), with **rank-weighted** hits and **negative markers** (defaults on).
- Pipeline E2E TOP1 ISP UMAP honors `stages.isp.postprocess.enabled` (`--enable-postprocess`).

### Docs

- README (ISP UMAP quick + tokenize note), [docs/tokenization.md](docs/tokenization.md), [docs/isp_umap.md](docs/isp_umap.md), [docs/web-ui.md](docs/web-ui.md).

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
