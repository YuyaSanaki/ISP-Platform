# Changelog

All notable changes to **ISP³ Platform** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- ISP UMAP `max_cells_per_state` now samples proportionally across `sample_id` (then shuffles) instead of taking the first N tokenized cells.
- ISP UMAP / WebUI remap `stage_configs/isp.yaml` dataset and model paths onto the selected local `--run-dir` when cluster absolute paths (e.g. `/work/...`) are missing after sync.
- E2E Pipeline auto ISP UMAP now selects the **TOP1 significant** gene (`significant_genes.csv` / parquet `Sig==1`), not the raw top positive shifter.

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
