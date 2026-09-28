# Changelog

All notable changes to **ISP³ Platform** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.0.1] - 2026-09-28

### Added

- State-feedback ISP (`core/run_state_feedback_isp.py`, `core/state_feedback/`): a learned residual Δrank decoder that reorders each cell's genes from perturbation-induced hidden-state changes, compared against the Ordered rank-edit path, parameter-free baselines and an oracle ceiling. Direction fidelity is scored against the observed Δrank on held-out genes. It now includes a base-rank control: a cross-fitted base-rank-only predictor, and partial Spearman given base rank for every method. The verdict requires signal beyond base rank. An optional perturbation-specificity stage (`state_feedback.specificity`, `--specificity`) feeds the same decoder Δh from named gene sets and random draws (optionally detection-matched), with results in `perturbation_specificity/`. The model-space results are in `docs/state_feedback_decode_methods.md`: BBRC OSKM / 3F / 7F against random controls, n=50 and n=300, and external validity on Asano PIPseq. These are in-silico results only.
- State-feedback ISP multi-step guardrails (`core/state_feedback/multistep.py`, `state_feedback.multi_step`). With `feedback_every_step: true`, each cell is halted on a 2-cycle (the proposed rerank is rejected) or stopped once its feedback changes stay small (Spearman > 0.995, or top-K Jaccard > 0.99 when the cell has more than K genes) for two events in a row. The chain is capped at `max_feedback_events` (default 5) feedback events. Per-cell halt reasons are written to `feedback_guard.csv` and per-condition counts to `run_manifest.json`. Whether multi-step feedback helps, or stays stable, has not been evaluated.
- Guide `docs/ordered_rank_edit_and_state_feedback_isp.md`: how Ordered rank-edit ISP and State-feedback ISP work, what each State-feedback condition does, how results are judged, and every Web UI setting. It states that Ordered rank-edit does not pass the model's response to the next step, so the final shift depends only on the final encoding. Linked from the README, `docs/web-ui.md`, `docs/in-silico pertabation.md`, `docs/ordered_rank_edit_isp.md` and `docs/state_feedback_decode_methods.md`.
- Web UI: **State-feedback ISP** run type. It reuses the Ordered rank-edit ISP source-run picker and step editor, can reuse a Δrank decoder trained in an earlier run on the same pipeline run (`--decoder-checkpoint`) or run direction fidelity only (`--eval-only`), and exposes the conditions, feedback step and perturbation-specificity options. The Outputs section shows the verdict, per-method direction fidelity, specificity tables and the Phase 1-2 gate.

### Changed

- **Sequential ISP is renamed Ordered rank-edit ISP**, to separate it from State-feedback ISP. The algorithm and its outputs are unchanged. What changed:
  - Web UI run type: **Ordered rank-edit ISP**.
  - Runner: `core/run_ordered_rank_edit_isp.py`.
  - Operators: `core/ordered_rank_edit.py`.
  - Template: `core/config/ordered_rank_edit_isp.yaml`, with an `ordered_rank_edit:` block.
  - Compose service: `ordered_rank_edit_isp`.
  - Output folders: `ordered_rank_edit_isp/` and `ordered_rank_edit_isp_<UTC>/`.
  - Doc: `docs/ordered_rank_edit_isp.md`.
  - Manifest `mode` values: `ordered_rank_edit_steps` and `ordered_rank_edit_oskm_orders`.
  - Auto-batch cache task: `ordered_rank_edit_isp_group` (the first run re-probes the GPU).

  The old names still work: `core/run_sequential_isp.py` and `core/sequential_oe.py` are aliases, the `sequential_isp` Compose service is kept, and a `sequential:` config block is read when `ordered_rank_edit:` is absent. State-feedback ISP configs now list their steps under `state_feedback.steps`; a top-level `sequential.steps` is still read.
- Renamed from **Geneformer Platform** to **ISP³ Platform**. GitHub repo is now `ISP-Platform`; Docker image / Compose service is `isp-platform:v1.0.0` (env `ISP_PLATFORM_IMAGE`).
- Mouse↔human Ensembl ortholog tables (`human_to_mouse.tsv`, `mouse_to_human.tsv`; release 116, retrieved 2026-08-07) are now distributed in the repository with `SHA256SUMS` and `ensembl_release.json`. `scripts/download_mouse_human_orthologs.sh` and the Docker build verify these pinned tables offline instead of querying the live BioMart; `ORTHOLOG_REFRESH=1` restores the live query.

## [1.0.0] - 2026-09-02

### Added

- Initial public release split from the private analysis monolith.
- End-to-end pipeline: tokenize → fine-tune → ISP (+ optional UMAP).
- Mouse and Human Geneformer backends with cross-species ortholog conversion.
- Web UI (Streamlit) and Docker Compose CLI.
- Sequential multi-gene ISP (renamed Ordered rank-edit ISP after 1.0.0).
- Ortholog loss gate, conversion reports, and smoke-matrix test harness.

### Notes

- Private paper-specific workflows live in **ISP³ Platform Analysis** (separate repo).
- Analysis repos should pin runtime to Docker image `isp-platform:v1.0.0`.
