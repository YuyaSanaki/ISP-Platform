# Changelog

All notable changes to **ISP³ Platform** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
