# Changelog

All notable changes to **ISP Platform** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Sample metadata manifest.** A TSV in the study root (`metadata.tsv`, `sample-metadata.tsv`, or `manifest.tsv`, or `single_cell_settings.metadata_manifest`) supplies cell labels instead of splitting the folder name. Required columns are `sample-id`, `disease`, and `species`. Further columns are kept and can be an ISP `state_key`. `species` must match `species.model_organism`. The repository root has `metadata.tsv.example` for the CLI. The Web UI **Manifest** table under **Data input** fills sample names from the uploaded folders, edits `disease` and `species`, and adds optional metadata with **＋**. With no manifest, `Time-Condition-Replicate` folder names work as before. Tests: `tests/test_sample_manifest.py`.

### Fixed

- **State-feedback Δrank decoder checkpoints record `training_steps`.** Loading `--decoder-checkpoint` against a different step list prints a transfer warning and sets `decoder.mode` to `transfer` in the run manifest (same-step reuse is `reuse_same_steps`). Order rankings that train one decoder per step list still mix order and decoder by design. Tests: `tests/test_state_feedback_phase12.py`.
- **State-feedback rerank keeps genes beyond `model_input_size`.** After a feedback event, the tail past the model window stays in original order (`keep_tail`), so a long encoding is no longer permanently shortened. Test: `tests/test_state_feedback_tail.py`.
- **10x→loom failures exit non-zero.** A sample that cannot be converted is counted; after the summary the stage exits non-zero (partial looms may remain for debugging). For `input_type: loom`, missing `custom_attr_name_dict` column attributes are back-filled with empty strings before tokenize. Test: `tests/test_tenx_loom_and_truncate.py`.
- **Cross-species tokenize warns when `max_cells` truncates a file.** Default behaviour is unchanged (each file still truncated). Truncations are recorded in provenance. Set `tokenizer.cross_species_truncate_to_max_cells: false` to keep all filter-pass cells per file. Test: `tests/test_tenx_loom_and_truncate.py`.
- **Web UI / pipeline hygiene (v1.4.1 follow-ups).** Optional login and disk-backed uploads / run zips; URL import limited to public hosts; approval card names the YAML keys it sets; Web UI imports only listed light core helpers (Geneformer loads lazily); Docker / macOS / H100 share `requirements-pins.txt` and Python 3.12 with package versions in run provenance; ISP warns on underrepresented states and offers opt-in `isp.max_ncells_per_state`; ISP UMAP records readout and offers opt-in `umap.readout: isp`; oracle mid-chain OE pin keeps overexpressed genes at the front through permute.

### Notes

- Intentional residuals documented under `.omm/` (image-build transitive pins; ISP / UMAP cell-set differences; approval-card overlay not shipped; study-ingest Streamlit uploader memory and DNS rebinding).

## [1.4.1] - 2026-10-21


### Fixed

- The automatic ISP UMAP of Pipeline (E2E) on genome-wide runs could project a gene seen in only a few cells. It took the largest shift toward goal among all Sig == 1 genes, so a gene detected in a handful of cells could outrank one detected in most cells. It now takes the largest shift among Sig == 1 genes detected in at least 20 cells (`N_Detections` ≥ 20), the same cut as the significant-genes lollipop, and skips the UMAP with a message when no gene passes (`pick_e2e_top1_significant_gene` in `core/run_pipeline.py`). **No ISP results change**: only the gene drawn in `isp_umap/` and its cluster / cell-type analysis can differ. Gene-list runs are unchanged. Test: `tests/test_e2e_top1_significant_umap.py`.

### Notes

- Docker image / Compose default tag is `isp-platform:v1.4.1`.

## [1.4.0] - 2026-10-20

Adds **gene set enrichment after ISP** (rank enrichment of the top Start, Goal and ISP genes on Reactome, GO and KEGG, computed locally) for normal ISP and State-feedback ISP, switched on together with cluster / cell-type analysis (`postprocess.enabled`, Web UI **Cluster / cell-type / enrichment analysis**). Also adds Apple silicon support. ISP scores and τ are unchanged.

### Added

- **Rank enrichment of Start, Goal and ISP, under one switch with cluster / cell-type analysis.** `postprocess.enabled` (CLI `--enable-postprocess` / `--skip-postprocess`, Web UI **Cluster / cell-type / enrichment analysis**) now turns on both the cluster / cell-type analysis after ISP UMAP and rank enrichment. `postprocess.rank_enrichment: false` (CLI `--no-rank-enrichment`) leaves rank enrichment out, as `celltype_prediction: false` leaves out cell-type prediction. Off by default, so existing runs, ISP scores and τ are unchanged (`core/postprocess_switch.py`).
  - **What it compares.** The top 10% of a shared background (genes detected in ≥5 cells of both Start and Goal, perturbed genes excluded) by Start pseudobulk rank, by Goal pseudobulk rank and by ISP go into an over-representation test (one-sided Fisher, BH) on Enrichr's Reactome Pathways 2024, GO Biological Process 2025, GO Molecular Function 2025, GO Cellular Component 2025 and KEGG (2021 Human or 2019 Mouse). The libraries are run locally from files downloaded once with pinned SHA256 (`scripts/download_gene_sets.sh`, also called by `download_build_assets.sh` and the Docker build), so gene lists are not sent to any web service. Pseudobulk rank = mean normalized position over cells (1 where the gene is not encoded).
  - **Normal ISP** (`core/run_isp.py`, Pipeline (E2E) stage `stages.isp.postprocess`): for per-gene screens (genome-wide or a gene list scored per gene), the ISP list holds the genes shifted toward Goal with FDR < 0.05, ranked by shift and capped at the size of the Start list. Its ORA universe is the background genes ISP tested. Start and Goal ranks come from the ISP dataset (`filter_data` applied, up to `rank_enrichment.max_cells_per_state` cells per state, default 3000). Group and per-cell outputs are skipped with a message.
  - **State-feedback ISP** (placebo-corrected mode, `core/run_state_feedback_stability.py`, meant for one step such as all factors overexpressed at once): the ISP list ranks genes by Δspec, the rank displacement after the placebo-corrected rerank minus the mean displacement under the comparison placebo chains, averaged over decoder seeds. **Closeness to Goal** (`core/state_feedback/closeness.py`) quantifies how far ISP moves Start toward Goal, per gene and per gene set of each library (progress, share of the most-changing genes or terms moved toward Goal, cosine, Spearman, distance closed, goal-gain AUC, similarity of enrichment profiles), compared with leave-one-out values from the comparison placebo chains (z, empirical p) and given per decoder seed.
  - **Figures** (paper style, PNG and PDF, `rank_enrichment/figures/`): one figure per library with −log10 FDR bars for Start, Goal and ISP (`enrich_<library>`), terms with the same genes and FDR merged into one row, and `closeness` for State-feedback ISP. Output: `rank_enrichment/` in the run folder. `core/run_isp_rank_enrichment.py --run-dir <run>` recomputes it with other settings for either kind of run. Guides: `docs/state_feedback_isp.md` 3.4, `docs/pipeline.md`. Test: `tests/test_rank_enrichment.py`.
- **Apple silicon (macOS, native).** ISP Platform runs on Apple silicon Macs without Docker (Docker on macOS has no GPU access): `scripts/macos/setup.sh` builds a uv virtual environment with PyTorch on the Metal GPU (MPS) and downloads the models, `scripts/macos/env.sh` sets the environment, and `scripts/macos/webui.sh` starts the Web UI on http://localhost:8502. Device selection (`core/compute_device.py`) picks CUDA, then MPS, then CPU (`ISP_DEVICE` overrides). fp16 fine-tuning stays CUDA-only. Configs and saved runs that point to `/app/...` are mapped to the checkout when `/app` does not exist (`core/platform_paths.py`). On NVIDIA hosts nothing changes. Test: `tests/test_platform_portability.py`.
- Web UI **Stop job** under **Logs & status**: stops the running job and its child processes (SIGTERM to the job's process group, SIGKILL after 15 s). The run folder keeps the files written so far.

### Changed

- **Web UI State-feedback ISP runs the paper's placebo-corrected method by default.** The run type used to start only `core/run_state_feedback_isp.py`, whose decoder reads Δh itself and which draws no placebos, so Web UI results differed from the paper's τ and recorded no placebo genes. A new **Mode** selector defaults to **Placebo-corrected (paper method)**, which runs `core/run_state_feedback_stability.py` with `placebo_contrast.enabled: true` and the paper's settings as defaults (10 correction placebos with draw seed 1, 30 comparison placebo chains with draw seed 0, decoder seeds 0–2, 5 idle reranks). All of these can be changed in the Web UI. Output goes to `{pipeline_run}/state_feedback_isp/state_feedback_stability_<time>/`, as on the command line, including `estimation_placebos.json` (correction placebo genes per step) and `random_chains.json` (comparison placebo genes with covariates). The Outputs panel shows τ per seed with CI, z and rank, the S1–S4 verdict, and both placebo gene tables with CSV downloads. The previous behaviour remains as **Decoder diagnostics (no placebo correction)**, which now writes `placebo_contrast.enabled: false` to its config and labels its results as not τ. **No results change:** both runners are unchanged, and command-line runs behave as before. Test: `tests/test_auto_batch_and_isp_validation.py`.

### Fixed

- Fine-tune embedding UMAP (`generate_umap` in `core/run_finetune.py`) could stall for many hours after a successful fine-tune on human V2 (seq 4096): it reloaded the model with `output_hidden_states=True` (all layers), ran every eval cell at `batch_size//2` with no progress logs, and a hang left the E2E pipeline short of ISP. It now subsamples up to `umap.max_cells` (default 2000), caps the forward batch (1 at 4096 / 4 at 2048), reads only the last-layer CLS via `.bert`, prints flushed stage progress, and treats UMAP errors as non-fatal so ISP still runs. Present since v1.0.0 (also fixed on ver1.0.2). No change to fine-tune weights or ISP scores.
- The mouse Geneformer base model was labelled "~10M" parameters in the Web UI (variant selector and its help), `README.md`, `models/README.md` and `docs/web-ui.md`. Counted from `models/mouse-Geneformer/model.safetensors`, it has 32,581,416 parameters, so the label now reads **Base (6L / ~33M)**. Only the label changes. The YAML id stays `base` and no results change.
- The State-feedback ISP guide and config said that with `placebo_contrast.enabled` every feedback event uses the placebo contrast. Only the stability runner (`core/run_state_feedback_stability.py`) reads `state_feedback.placebo_contrast`; the State-feedback ISP runner (`core/run_state_feedback_isp.py`, now the Web UI's Decoder diagnostics mode, see Changed) trains its decoder and reranks with Δh itself. The guide and config comment now say so, and the runner prints a note when the key is on. No results change: the paper's placebo-corrected results (τ, z) come from the stability runner, which applies the contrast.
- Tokenization no longer loops forever when the last group of input files yields no cells (for example an empty sample, or loom files whose cells all fail `filter_pass`). `TranscriptomeTokenizer.tokenize_data` re-read the same files without end; it now stops after the last group, and raises an error if no cells were tokenized at all. Runs that finished before are unchanged, because they never reached this case. Test: `tests/test_tokenizer_empty_rounds.py`.
- Group ISP without goal states (`InSilicoPerturber` with `genes_to_perturb` set and `cell_states_to_model=None`) stopped with a tensor size mismatch whenever the forward minibatches differed in their longest cell, because each minibatch's cosine similarities are padded only to that cell. They are now padded to a common width before they are joined. Its scores were also wrong. The cell score averaged L − k − p positions instead of the L − k genes that both encodings share (k perturbed genes, p of them present in the cell, or 1 when none is present), so it left out the p lowest-ranked shared genes, and group delete left out 2k. In `cell_and_gene` mode each value was labelled with the gene at the same position of the unperturbed encoding, which is a different gene after the first perturbed gene present in the cell, and the loop ran past the end of the row. Scores now average the L − k shared genes and carry their own gene labels (`group_compared_genes`, `add_group_cos_sims`). **No results change:** ISP Platform always passes goal states (`core/run_isp.py`), whose group scoring (`group_goal_state_mean_lengths`) is separate and unchanged. Test: `tests/test_group_cos_sims_no_states.py`.
- Genome-wide ISP without goal states (`genes_to_perturb: all`, `cell_states_to_model=None`) had the same label shift in `cell_and_gene` mode. The cosine similarities of each perturbation are taken against the original embedding with the perturbed gene's position removed, but value k was labelled with `input_ids[k]` of the unperturbed encoding, so every value after the perturbed gene was attributed to the gene one rank higher and the perturbed gene itself was listed as an affected gene. Values now carry the genes they compare (`remaining_gene_ids`, which `group_compared_genes` also uses). The values themselves and the cell scores were already correct. **No results change:** ISP Platform always passes goal states, and `emb_mode` defaults to `cell`. Test: `tests/test_genome_wide_gene_labels.py`.
- In silico perturbation with an anchor gene in combinations (`make_perturbation_batch` with `combos > 0` and `anchor_gene`) read the key `"input_ids "` (trailing space) and raised `KeyError`; it now reads `input_ids`. No results change: ISP Platform never reaches this branch, because a gene list resets `anchor_gene` and `combos`, genome-wide ISP takes the `"all"` branch, and `anchor_gene` is dropped whenever cell states are modelled. Test: `tests/test_anchor_combo_batch.py`.
- Web UI runs one job at a time across all browser sessions. The running job used to live only in the session that started it, so a second tab, or the same tab after a reload, could start another job on the same GPU. **Run job** now takes a lock file in the Web UI workspace (`active_job.json`) and is disabled everywhere while the job runs; other sessions show the running job with its log and **Stop job**. The lock clears itself when the job's process exits. No results change: jobs run the same commands. Test: `tests/test_webui_job_lock.py`.
- **The E2E pipeline (`core/run_pipeline.py`) uses every tokenized shard and the best fine-tuning run.** The tokenizer starts a new shard (`{study}_1.dataset`, …) each time the cells read so far reach `max_cells` (300,000 by default), but the pipeline passed only `{study}_0.dataset` to fine-tuning and ISP, so the cells in later shards were silently left out. With several shards, the pipeline now concatenates them in order into `tokenized_dataset/{study}_merged.dataset` (with a `{study}_merged.json` manifest of cells per shard) and uses that. ISP also always took `finetune/all_run1`; with `training.num_runs > 1` it now takes the run with the highest eval accuracy (ties go to the earlier run), the model that fine-tuning already reports as best. With `finetune.organ_key` set, the pipeline stops before ISP with an explanation instead of a missing `all_run1`. Re-tokenizing into an existing run folder (`--run-dir`) first removes shards from the earlier attempt. **Results change only for** runs where the cells read reached `max_cells` while input files remained (the run's `tokenized_dataset/` holds `{study}_1.dataset` or later), and runs with `num_runs > 1` whose best run was not run 1. Single-shard runs with `num_runs: 1` (the default) are unchanged. On cross-species input, `max_cells` still also limits each sample file, as before. Test: `tests/test_pipeline_stage_paths.py`.
- ISP `perturbation.type: inhibit` and `activate` (shift a gene to a lower / higher quartile) now stop at once with "not implemented". They were offered in the Web UI and the YAML comments but never worked: without `isp.perturb_rank_direct_shift` the option check stopped with a bare `raise`, and with it the rank shift was never applied (several branches compared against `"active"`), so `activate` scored unperturbed cells; correcting the name only moved the failure to unfinished alignment code. Both types are removed from the Web UI selector, `core/config/isp.yaml`, `core/config/pipeline.yaml` and `docs/pipeline.md`. No results change: `delete` and `overexpress` take separate branches, and no released result used `inhibit` or `activate`. Test: `tests/test_perturb_type_unimplemented.py`.
- ISP statistics accept more than one alternate state. `perturbation.alt_states` with two or more states made `InSilicoPerturberStats` (`goal_state_shift`) stop with `ValueError: too many values to unpack` after the whole perturbation had run, because it read each cell's shifts as exactly (start, goal, alt). It now reads them by position. With several alt states the alt columns are written per state (`Shift_to_alt_end_<state>`, `Alt_end_<state>_vs_random_pval`, `Alt_end_<state>_FDR`); with none or one alt state the output keeps its columns and is unchanged (checked against the previous code, including the 100,000-shift downsampling). **No results change:** runs with zero or one alt state (the default is none) produce the same tables, and runs with two or more stopped before writing any. State-feedback ISP does not use these statistics. Test: `tests/test_isp_stats_alt_states.py`.
- Group-mode ISP (`perturbation.genes_to_perturb` set) with `forward_batch_size: auto` probed one forward pass per batch, but each batch runs two (perturbed, then original). The probe now runs both and keeps only the layer that is scored from the perturbed pass, as the run does, so the chosen size fits. Probes for these runs are cached separately (`task: isp_group_forward`); genome-wide runs keep their cached sizes. Group scores do not depend on the batch size, so results do not change.
- Pipeline (E2E) runs with a gene list skipped the ISP UMAP. The E2E step picks the TOP1 gene from the per-gene statistics, which group mode does not write. Runs with `genes_to_perturb` now project the listed genes perturbed together (`run_isp_umap.py --run-dir`, PCA 50, seed 0), as the Web UI ISP UMAP run type does; genome-wide runs still project the TOP1 significant gene. ISP results are unchanged; the run writes `isp_umap/` and takes longer.
- `pipeline_run.log` held only the pipeline's own lines: tokenize, fine-tune, ISP and ISP UMAP wrote to the console only, so the log had no stage output or tracebacks. The pipeline now relays each stage's stdout and stderr to both.
- An ortholog approval record with decision `stop` (or `sensitivity_best_of_n`) made the pipeline fail with "Tokenized dataset not found". The pipeline now ends after tokenize with a message naming the decision, exit code 0 and `run_status: stopped_ortholog_approval` in `pipeline_run_metadata.yaml`.
- `pipeline_config_used.yaml` in a pipeline run was a copy of the YAML file passed to `--config`, so it missed `--ortholog-curated-overlay`, `--ortholog-audit`, `--ortholog-approval-record` and the `data.output_prefix` derived from the input folder. It now holds the config as run, with these applied (comments from the source file are not kept). `pipeline_run_metadata.yaml` keeps the source path (`config_source_path`) and lists the CLI options under `cli_overrides`. No results change. Test: `tests/test_run_pipeline_stages.py`.
- `run_pipeline.py --skip-tokenize` without `--run-dir` (or `paths.run_dir`) created a new, empty run folder and failed later with "Tokenized dataset not found". It now stops at once with a usage error and creates nothing. Test: `tests/test_run_pipeline_stages.py`.
- Config comments: in `core/config/isp.yaml` the description of `genes_to_perturb` sat above `alt_states`; each key now has its own, and the `alt_states` comment says that two or more states get alt columns per state. In `core/config/finetune.yaml` the comment said `add_columns.organ_major` is used for training; it is only a constant label unless `finetune.organ_key: organ_major`.

### Security

- Compose services no longer mount the Docker socket (`/var/run/docker.sock`). Access to it from inside a container amounts to root on the host, and nothing in the image uses it (`scripts/run_smoke_matrix.py` calls `docker compose` from the host). No results change.

### Notes

- Docker image / Compose default tag is `isp-platform:v1.4.0`.

## [1.3.3] - 2026-10-10

### Changed

- Renamed **ISP³ Platform** to **ISP Platform** (README, logo, Web UI title, docs). The repository, Docker image (`isp-platform`), Compose service and environment variables keep their names. No code or result changes from v1.3.2.

### Notes

- Docker image / Compose default tag is `isp-platform:v1.3.3`.

## [1.3.2] - 2026-10-10

Includes the peer-reviewing v1.0.1 (tag `v1.0.1`, review response for *Genes to Cells*); the full list is under [1.0.1] below. All v1.1.0–v1.3.1 features are kept (500-step warmup default, ISP UMAP postprocess, pre-ISP cell-type annotation and panels).

### Changed

- **Multi-step ISP is State-feedback ISP.** The Sequential ISP run type (ordered rank-edit without feedback) is removed: runner, config, Compose service `sequential_isp`, Web UI run type and `docs/sequential_isp.md`. The same steps without feedback remain as the `no_feedback` baseline of State-feedback ISP. Guide: `docs/state_feedback_isp.md`.
- State-feedback ISP defaults match the paper: placebo contrast on, 30 comparison placebos, decoder trained on up to 300 start cells.
- Mouse↔human Ensembl ortholog tables are pinned in the repository (release 116, `SHA256SUMS`); the download script and the Docker build verify them offline. The ortholog loss gate decides on critical genes only.

### Added

- State-feedback ISP (`core/run_state_feedback_isp.py`, `core/state_feedback/`, Web UI run type, stability runner `core/run_state_feedback_stability.py`) with the potential-outcome placebo contrast and sequential KO counterfactual.
- Web UI project curated ortholog overlay and ortholog loss gate with a critical-gene list; overlay template in `examples/ortholog_overlays/`.
- `docs/upstream_overexpression.md` and length-preserving OE tests; ISP stats column glossary in `docs/in-silico pertabation.md`.

### Fixed

- Group delete / overexpress goal-state scores no longer include padding (`isp.legacy_padding_mean: true` restores the earlier scores).

### Notes

- Docker image / Compose default tag is `isp-platform:v1.3.2`.

## [1.3.1] - 2026-09-22

### Added

- Pred L2 mean bar restricted to types with n>20: `l2_mean_by_pred_celltype_n_gt20.png`.
- Pre-ISP panel `isp_expression_v2`: split near-neighbor types (AT1/AT2, cDC1/cDC2/pDC, basal/suprabasal keratinocyte), core/support/anti markers, and gates for core-hit count plus cross-compartment `Doublet_suspected` / `ambiguous_cross_compartment`. Fiber type is a `celltype_state` on `Skeletal_myocyte`. Mural and myeloid winners also store an in-group alternate in `celltype_state`. `celltype_confidence=low` when fewer than half of a type's core genes are in the Geneformer vocabulary. `C1QC_APOE_macrophage` replaces a broad resident-macrophage label. Existing tokenized datasets stay `isp_expression_v1` until re-tokenized.

### Notes

- Docker image / Compose default tag is `isp-platform:v1.3.1`.

## [1.3.0] - 2026-09-21

### Changed

- ISP UMAP cell-type postprocess is **pre-ISP only** (`isp_expression_v1` platform labels). Brain token-rank marker panels were removed.
- Pred L2 mean bars (`l2_mean_by_pred_celltype.png`) are ordered by **mean shift_l2 descending** and omit Unknown/Ambiguous plus types with fewer than 2 cells (singleton false calls such as Neuron).
- Coarse L2 figures (`l2_by_coarse_celltype.png`, `l2_mean_by_coarse_celltype.png`) are no longer generated by the postprocess pipeline.

### Notes

- Docker image / Compose default tag is `isp-platform:v1.3.0`.

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

## [1.0.1] - 2026-10-09

Peer-reviewing version (review response, *Genes to Cells*). Tag `v1.0.1` was moved on 2026-10-09 from `3b8616f` (first release, 2026-09-28, now tag `v1.0.1-initial`) to include the peer-review changes. Clones made before the move keep the old tag until `git fetch --tags --force`.

### Changed

- **State-feedback ISP defaults match the method in the paper** (`core/config/state_feedback_isp.yaml`): the placebo contrast is on (`placebo_contrast.enabled: true`, 10 estimation placebos), the stability runner draws one set of 30 comparison placebos (`stability.n_random_chains: 30`, `random_seeds: [0]`; previously 2 draws of 20), and the decoder is trained on up to 300 start cells (`decoder.train_max_ncells: 300`; previously 200). Runs with the default config therefore take longer and report smaller, placebo-corrected gains than before. Set `placebo_contrast.enabled: false` for the previous behaviour.
- State-feedback ISP guide and Web UI help state that the method is a sequential simulation, not a time course: each step acts on the state inferred after the previous steps, so the order of the steps matters, but a step is not a unit of time.
- **The ortholog loss gate decides on critical genes only.** It blocks when a critical gene present in the input is dropped by the mapping (or when the mapping provenance is incomplete) and warns when a critical gene is absent from the input. The percentage of input features mapped is still reported but no longer blocks below 50% or warns below 90% by default: its denominator is every annotated input feature, so it reflects the annotation of the input more than the genes that reach the model (the paper's human data map 48.9% to Mouse-Geneformer and were blocked). Set `block_mapped_pct_min` / `warn_mapped_pct_min` / `pass_mapped_pct_min` in the audit to restore the previous thresholds. An audit must now list at least one critical gene; one without critical genes stops with an error.
- **The platform curated ortholog tables now restore POU5F1 ↔ Pou5f1 and NANOG ↔ Nanog** (both directions). Ensembl labels both pairs one-to-many because of the human paralogues POU5F1B and NANOGP8, so `one2one` dropped these core pluripotency factors from every cross-species conversion. The paralogues stay unmapped. Each direction now maps 17,149 pairs (17,147 before). Cross-species runs therefore differ from earlier versions by these two genes; same-species runs are unchanged.

- **State-feedback ISP baseline condition renamed `ordered_rank_edit` → `no_feedback`.** It applies the same steps with no feedback between them. Its output folder is `no_feedback/` and the stability runner's endpoint columns are `no_feedback_endpoint` and `gain_over_no_feedback` (`phase12_gate.csv`: `no_feedback_final_median`, `beats_no_feedback`). `ordered_rank_edit` is still accepted in configs and `--conditions`, steps under a top-level `ordered_rank_edit:` or `sequential:` block are still read, and `aggregate` still reads runs written with the old folder and column names. The step operators moved to `core/rank_edit.py` and the runner helpers (start-cell selection, scoring, GPU batch sizing) to `core/state_feedback/runtime.py`.
- **State-feedback ISP always feeds back after every step; the other schedules and the event cap are removed.** Feedback only after some steps, or only after the last step, makes the result depend on the final encoding alone, as without feedback, so these schedules were not sequential. `feedback_every_step`, `feedback_after_step`, `feedback_after_last_step`, `multi_step.max_feedback_events` and `eval.perturbation: feedback_point` are gone from the runner, the config and the Web UI; the number of feedback events now always equals the number of steps. A config that still sets these keys to anything other than feedback after every step stops with an error. The Web UI, the config and the guide now state that the error grows with the number of steps. The stability runner drops its single-event mode.
- **Random control chains are matched to the configured genes' position in the start cells** (`core/state_feedback/random_chains.py`). Random genes used to be drawn from genes detected in both states, so on BBRC OSKM they were mostly genes already present near the top of the encoding, while OSKM are absent (SOX2, POU5F1) or rare near the bottom (KLF4, MYC). Overexpressing them was a different edit. Each configured gene is now replaced by a gene drawn uniformly from the whole vocabulary among genes with the same start-cell detection and position. Draws with several random seeds are independent, and the draw is written with gene symbols and balance to `random_chains.json`. The specific-gain CI now resamples random chains as well as cells, and the aggregate reports the gain per random draw, the running random mean and a leave-one-out null.
- **State-feedback ISP Δrank decoder reads Δh only.** The linear layer no longer takes base rank or a bias, so zero perturbation gives exactly zero displacement for any trained weights, and the identity loss term (`decoder.lam_identity`) is gone. On BBRC n=300 the Δh-only prediction of the previous decoder already matched it beyond base rank (partial ρ 0.332 vs 0.329). Decoders saved before this change cannot be loaded; retrain instead of passing them to `--decoder-checkpoint`. The `delta_h_only` control is dropped because it is now the decoder itself; `delta_h_shuffled` stays. Reported direction-fidelity and specificity numbers are from Pegasus re-measurement at `46035be` (2026-09-29); see `docs/state_feedback_decode_methods.md`.

### Added

- Template for a project curated ortholog overlay (`examples/ortholog_overlays/`): `curated_bridge_{human_to_mouse,mouse_to_human}.tsv.example` with one format row (POU5F1 ↔ Pou5f1, already in the platform table, so it changes no mapping) and a README on copying, editing and selecting it. The `.example` suffix keeps the folder itself from being used as an overlay.
- Web UI, cross-species runs: **Project curated overlay** (path to a TSV or a folder with `curated_bridge_<pair>.tsv`, written to `species.ortholog_curated_overlay`; the run does not start if no table is found for the current pair) and **Ortholog loss gate** with a **Critical genes** list (written to `stages.tokenize.tokenizer.ortholog_audit_inline`). Both could previously only be set in YAML or on the command line.
- State-feedback ISP placebo contrast (`core/state_feedback/placebo_contrast.py`, `state_feedback.placebo_contrast`). Most of the feedback gain is not specific to the perturbed genes: matched placebo genes get most of it too. With the contrast, the decoder is trained on, and every feedback event uses, Δh minus the mean Δh after swapping the chain's genes for each of `n_estimation` matched estimation placebos in the same cell. The stability runner reports the gain of the configured chain minus the mean gain of separately drawn comparison placebo chains (`specific_gain`). Delete steps use the counterfactual in `docs/state_feedback_deletion_counterfactual.md`.
- Diagram of how the default ortholog tables and a project overlay combine (`docs/ortholog_tables_and_overlay.png`), with the load order and POU5F1 / NANOG / GAPDH as examples, in the ortholog README and `docs/tokenization.md`.
- State-feedback ISP `pin_overexpressed` (default `true`): after each reorder, the genes overexpressed so far stay at the front in their pre-reorder order, and only the other genes are reordered. This applies to every rerank condition, including `oracle`. Rerank diagnostics are computed on the pinned order.
- `docs/upstream_overexpression.md` (review response, R2-Major2): how the length-preserving group OE of ISP Platform differs from official Geneformer (`ctheodoris/Geneformer` `1f7fbae`, the tip of `main` on 2026-09-30). Official Geneformer cuts the perturbed cell to the model input size and cuts the original by an overflow count taken from the cut length. With k OE genes absent from a cell, that count is wrong for lengths `max_len` − 2k < L < `max_len`; on Geneformer V2-104M with OSKM, a cell of length 4,094 stops with a 4,092 vs 4,090 size mismatch, and all other tested lengths run. The `overexpress_tokens` docstring no longer says that upstream Geneformer often fails; it names this range and the Mouse-Geneformer path that ISP Platform replaced. New tests in `tests/test_overexpress_length_preserve.py` run the failing input and the whole range through the ISP³ operators.
- State-feedback ISP guide, design doc and Web UI help: numbers on reusing a Δrank decoder for other steps. On BBRC OSKM (n=3000, 24 orders), a decoder trained on simultaneous OSKM and reused for every order scored lower direction fidelity than one trained per order (pooled Spearman 0.38 vs 0.46), except for the 6 orders ending in POU5F1, and gave an uncorrelated order ranking. Train a new decoder (the default) for reported results.
- `docs/in-silico pertabation.md`: column glossary for the ISP stats table (review response, R2-Minor1). `N_Detections` is the number of perturbed start-state cells whose encoding contains the gene, i.e. the number of per-cell shifts averaged into `Shift_to_goal_end`; the `N_Detections` ≥ 20 cut-off (`min_n_detections`) applies to the lollipop figure only.

### Fixed

- **Group delete / overexpress goal-state scores no longer include padding.** Scoring removes k positions from the original embedding (and, for overexpression, the k leading OE positions from the perturbed one) but mean-pooled over the unremoved length L, so for cells shorter than the longest cell of their forward minibatch up to k padding hidden states entered the mean. Both sides are now pooled over their first L − k positions (a deletion encoding keeps its own length), which gives the same score at any `forward_batch_size`, i.e. the score of `forward_batch_size: 1`. On BBRC (Geneformer V2-104M, somatic → pluripotent, batch 8) the median goal-state shift of group OE was 1.0–2.1% too low (OSKM 0.01004 → 0.01025) and that of group delete too high (ACTB+B2M+GAPDH 0.00626 → 0.00563); the order of factor sets is unchanged. This applies to ISP (`run_isp.py`) and to the group-aligned scoring of the State-feedback oracle runner's no-feedback path (`run_state_feedback_oracle.py`). The State-feedback ISP runner is unchanged: its cell-mean scoring already pooled each cell over its own length (BBRC re-runs were bit-identical). Single-gene ISP is unchanged. Set `isp.legacy_padding_mean: true` (or `ISP_LEGACY_PADDING_MEAN=1`, which the oracle runner reads) to reproduce scores of earlier releases; the setting is recorded in `isp_run_metadata.yaml`. Test: `tests/test_group_goal_state_padding.py`.
- Group-OE scoring that assumes the overexpressed genes sit at the sequence front (`state_feedback.runtime.compute_goal_state_shifts`, used by the oracle runner's no-feedback path) counted a gene overexpressed in two steps twice. Scoring strips one leading position per token, so it also dropped the genes right after the factor block from both embeddings. Repeats are now dropped before scoring. Steps that repeat no gene (all OSKM orders) are unchanged, and the State-feedback ISP runner's cell-mean scoring was never affected.
- State-feedback ISP endpoint gate now reads the feedback row at the last step when there is one, instead of relying on how step names sort against `feedback`.
- Docker build pins `datasets==5.0.1`. The build installs the newest `datasets`, and 5.1 needs `huggingface_hub` ≥ 1.0 while `transformers` < 5 keeps it below 1.0, so a fresh image failed on `import geneformer`.

### Removed

- **Ordered rank-edit ISP as a run type.** Multi-step ISP is State-feedback ISP; the same steps without feedback remain as its `no_feedback` baseline. Removed: `core/run_ordered_rank_edit_isp.py`, `core/ordered_rank_edit.py`, the aliases `core/run_sequential_isp.py` and `core/sequential_oe.py`, `core/run_oskm4_then_7factor_isp.py`, `core/config/ordered_rank_edit_isp.yaml`, the Compose services `ordered_rank_edit_isp` and `sequential_isp`, the Web UI run type **Ordered rank-edit ISP**, `docs/ordered_rank_edit_isp.md`, and `tests/test_ordered_rank_edit.py` (operator tests kept in `tests/test_rank_edit.py`). The guide `docs/ordered_rank_edit_and_state_feedback_isp.md` is replaced by `docs/state_feedback_isp.md`, and the README diagram now compares conventional ISP with State-feedback ISP.
- State-feedback ISP per-cell multi-step stops. The convergence stop treated small whole-encoding changes as convergence even when a few genes moved a lot (one gene moving bottom to top in a 2048-gene cell still gives Spearman 0.997) or when a new perturbation was still to come. The 2-cycle stop required an exact return to the order from two events ago, which practically never happens because a new perturbation enters between events. `feedback_guard.csv` is no longer written. A `state_feedback.multi_step` block, including `converge_*` and `halt_on_cycle`, is ignored with a warning, so older configs still load.

## [1.0.1-initial] - 2026-09-28

First release of v1.0.1 (tag `v1.0.1-initial`, `3b8616f`). Superseded by the peer-reviewing v1.0.1 above.

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

- Private paper-specific workflows live in **ISP Platform Analysis** (separate repo).
- Analysis repos should pin runtime to Docker image `isp-platform:v1.0.0`.
