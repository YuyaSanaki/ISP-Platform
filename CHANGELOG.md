# Changelog

All notable changes to **ISP³ Platform** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
