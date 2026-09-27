# Changelog

All notable changes to **ISP³ Platform** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- State-feedback ISP (`core/run_state_feedback_isp.py`, `core/state_feedback/`): a learned residual Δrank decoder that reorders each cell's genes from perturbation-induced hidden-state changes, compared against the Ordered rank-edit path, parameter-free baselines and an oracle ceiling. Direction fidelity is scored against the observed Δrank on held-out genes. It now includes a base-rank control: a cross-fitted base-rank-only predictor, and partial Spearman given base rank for every method. The verdict requires signal beyond base rank. An optional perturbation-specificity stage (`state_feedback.specificity`, `--specificity`) feeds the same decoder Δh from named gene sets and random draws (optionally detection-matched), with results in `perturbation_specificity/`. The model-space results are in `docs/state_feedback_decode_methods.md`: BBRC OSKM / 3F / 7F against random controls, n=50 and n=300, and external validity on Asano PIPseq. These are in-silico results only.
- Web UI: **State-feedback ISP** run type. It reuses the Sequential ISP source-run picker and step editor, can reuse a Δrank decoder trained in an earlier run on the same pipeline run (`--decoder-checkpoint`) or run direction fidelity only (`--eval-only`), and exposes the conditions, feedback step and perturbation-specificity options. The Outputs section shows the verdict, per-method direction fidelity, specificity tables and the Phase 1-2 gate.

### Changed

- Renamed from **Geneformer Platform** to **ISP³ Platform**. GitHub repo is now `ISP-Platform`; Docker image / Compose service is `isp-platform:v1.0.0` (env `ISP_PLATFORM_IMAGE`).
- Mouse↔human Ensembl ortholog tables (`human_to_mouse.tsv`, `mouse_to_human.tsv`; release 116, retrieved 2026-08-07) are now distributed in the repository with `SHA256SUMS` and `ensembl_release.json`. `scripts/download_mouse_human_orthologs.sh` and the Docker build verify these pinned tables offline instead of querying the live BioMart; `ORTHOLOG_REFRESH=1` restores the live query.

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
