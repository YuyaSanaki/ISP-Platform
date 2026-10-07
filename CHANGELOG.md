# Changelog

All notable changes to **ISP³ Platform** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- **State-feedback ISP baseline condition renamed `ordered_rank_edit` → `no_feedback`.** It applies the same steps with no feedback between them. Its output folder is `no_feedback/` and the stability runner's endpoint columns are `no_feedback_endpoint` and `gain_over_no_feedback` (`phase12_gate.csv`: `no_feedback_final_median`, `beats_no_feedback`). `ordered_rank_edit` is still accepted in configs and `--conditions`, steps under a top-level `ordered_rank_edit:` or `sequential:` block are still read, and `aggregate` still reads runs written with the old folder and column names. The step operators moved to `core/rank_edit.py` and the runner helpers (start-cell selection, scoring, GPU batch sizing) to `core/state_feedback/runtime.py`.
- **State-feedback ISP always feeds back after every step; the other schedules and the event cap are removed.** Feedback only after some steps, or only after the last step, makes the result depend on the final encoding alone, as without feedback, so these schedules were not sequential. `feedback_every_step`, `feedback_after_step`, `feedback_after_last_step`, `multi_step.max_feedback_events` and `eval.perturbation: feedback_point` are gone from the runner, the config and the Web UI; the number of feedback events now always equals the number of steps. A config that still sets these keys to anything other than feedback after every step stops with an error. The Web UI, the config and the guide now state that the error grows with the number of steps. The stability runner drops its single-event mode.
- **Random control chains are matched to the configured genes' position in the start cells** (`core/state_feedback/random_chains.py`). Random genes used to be drawn from genes detected in both states, so on BBRC OSKM they were mostly genes already present near the top of the encoding, while OSKM are absent (SOX2, POU5F1) or rare near the bottom (KLF4, MYC). Overexpressing them was a different edit. Each configured gene is now replaced by a gene drawn uniformly from the whole vocabulary among genes with the same start-cell detection and position. Draws with several random seeds are independent, and the draw is written with gene symbols and balance to `random_chains.json`. The specific-gain CI now resamples random chains as well as cells, and the aggregate reports the gain per random draw, the running random mean and a leave-one-out null.
- **State-feedback ISP Δrank decoder reads Δh only.** The linear layer no longer takes base rank or a bias, so zero perturbation gives exactly zero displacement for any trained weights, and the identity loss term (`decoder.lam_identity`) is gone. On BBRC n=300 the Δh-only prediction of the previous decoder already matched it beyond base rank (partial ρ 0.332 vs 0.329). Decoders saved before this change cannot be loaded; retrain instead of passing them to `--decoder-checkpoint`. The `delta_h_only` control is dropped because it is now the decoder itself; `delta_h_shuffled` stays. Reported direction-fidelity and specificity numbers are from Pegasus re-measurement at `46035be` (2026-09-29); see `docs/state_feedback_decode_methods.md`.

### Added

- Example ortholog overlay `examples/ortholog_overlays/pou5f1_bridge/`: the POU5F1 ↔ Pou5f1 bridge (POU5F1B excluded) used for the paper's cross-species runs, byte-identical to the analysis overlay. Pass it as `species.ortholog_curated_overlay` to reproduce that conversion; the default tables are unchanged, so `one2one` still drops POU5F1 without it.
- State-feedback ISP `pin_overexpressed` (default `true`): after each reorder, the genes overexpressed so far stay at the front in their pre-reorder order, and only the other genes are reordered. This applies to every rerank condition, including `oracle`. Rerank diagnostics are computed on the pinned order.
- `docs/upstream_overexpression.md` (review response, R2-Major2): how ISP³ length-preserving group OE differs from official Geneformer (`ctheodoris/Geneformer` `1f7fbae`, the tip of `main` on 2026-09-30). Official Geneformer cuts the perturbed cell to the model input size and cuts the original by an overflow count taken from the cut length. With k OE genes absent from a cell, that count is wrong for lengths `max_len` − 2k < L < `max_len`; on Geneformer V2-104M with OSKM, a cell of length 4,094 stops with a 4,092 vs 4,090 size mismatch, and all other tested lengths run. The `overexpress_tokens` docstring no longer says that upstream Geneformer often fails; it names this range and the Mouse-Geneformer path that ISP³ replaced. New tests in `tests/test_overexpress_length_preserve.py` run the failing input and the whole range through the ISP³ operators.
- State-feedback ISP guide, design doc and Web UI help: numbers on reusing a Δrank decoder for other steps. On BBRC OSKM (n=3000, 24 orders), a decoder trained on simultaneous OSKM and reused for every order scored lower direction fidelity than one trained per order (pooled Spearman 0.38 vs 0.46), except for the 6 orders ending in POU5F1, and gave an uncorrelated order ranking. Train a new decoder (the default) for reported results.
- `docs/in-silico pertabation.md`: column glossary for the ISP stats table (review response, R2-Minor1). `N_Detections` is the number of perturbed start-state cells whose encoding contains the gene, i.e. the number of per-cell shifts averaged into `Shift_to_goal_end`; the `N_Detections` ≥ 20 cut-off (`min_n_detections`) applies to the lollipop figure only.

### Fixed

- Group-OE scoring that assumes the overexpressed genes sit at the sequence front (`state_feedback.runtime.compute_goal_state_shifts`, used by the oracle runner's no-feedback path) counted a gene overexpressed in two steps twice. Scoring strips one leading position per token, so it also dropped the genes right after the factor block from both embeddings. Repeats are now dropped before scoring. Steps that repeat no gene (all OSKM orders) are unchanged, and the State-feedback ISP runner's cell-mean scoring was never affected.
- State-feedback ISP endpoint gate now reads the feedback row at the last step when there is one, instead of relying on how step names sort against `feedback`.

### Removed

- **Ordered rank-edit ISP as a run type.** Multi-step ISP is State-feedback ISP; the same steps without feedback remain as its `no_feedback` baseline. Removed: `core/run_ordered_rank_edit_isp.py`, `core/ordered_rank_edit.py`, the aliases `core/run_sequential_isp.py` and `core/sequential_oe.py`, `core/run_oskm4_then_7factor_isp.py`, `core/config/ordered_rank_edit_isp.yaml`, the Compose services `ordered_rank_edit_isp` and `sequential_isp`, the Web UI run type **Ordered rank-edit ISP**, `docs/ordered_rank_edit_isp.md`, and `tests/test_ordered_rank_edit.py` (operator tests kept in `tests/test_rank_edit.py`). The guide `docs/ordered_rank_edit_and_state_feedback_isp.md` is replaced by `docs/state_feedback_isp.md`, and the README diagram now compares conventional ISP with State-feedback ISP.
- State-feedback ISP per-cell multi-step stops. The convergence stop treated small whole-encoding changes as convergence even when a few genes moved a lot (one gene moving bottom to top in a 2048-gene cell still gives Spearman 0.997) or when a new perturbation was still to come. The 2-cycle stop required an exact return to the order from two events ago, which practically never happens because a new perturbation enters between events. `feedback_guard.csv` is no longer written. A `state_feedback.multi_step` block, including `converge_*` and `halt_on_cycle`, is ignored with a warning, so older configs still load.

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
- Renamed from **Geneformer Platform** to **ISP³ Platform**. GitHub repo is now `ISP-Platform`; Docker image / Compose service is `isp-platform` (env `ISP_PLATFORM_IMAGE`).
- Compose default image is `isp-platform:v1.0.1`. The image is rebuilt because the build now requires `typing-extensions>=4.13` and verifies the pinned ortholog tables.
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
