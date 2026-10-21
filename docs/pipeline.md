# End-to-end pipeline (Tokenize → Fine-tune → ISP)

One config and one command run **Tokenize → Fine-tune → ISP** in order. Dataset, checkpoint, and ISP paths are **derived automatically** from `data.input_dir` — you do not copy paths between stage YAML files.

Entrypoint: [`run_pipeline.py`](../core/run_pipeline.py). Configuration: [`core/config/pipeline.yaml`](../core/config/pipeline.yaml).

---

## Run

**CLI (default config):**

```bash
docker compose run --rm pipeline
```

**Custom config:**

```bash
docker compose run --rm pipeline python3 /app/core/run_pipeline.py --config /app/core/config/my_pipeline.yaml
```

Override via env: `PIPELINE_CONFIG=/app/core/config/my_pipeline.yaml`.

---

## Data input rules

Pipeline stage 1 uses the same layout as the tokenize service. Full reference: [**tokenization.md**](tokenization.md).

### Study root (`data.input_dir`)

Point `data.input_dir` at the **parent of sample folders** (the study root), not at a single sample folder.

```text
{input_dir}/
  1w-Disease-SingleCell/barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz
  1w-Ctrl-SingleCell/barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz
```

Or nested under an experiment name:

```text
/app/data/MyExperiment/
  1w-Disease-SingleCell/*.gz
  1w-Ctrl-SingleCell/*.gz
```

| Pattern | `data.input_dir` example |
|---------|---------------------------|
| One study, samples as direct children | `/app/data/MyExperiment/` |
| Several experiments under `/data/` | `/app/data/` (all samples under each child are discovered) |
| Pre-built `.loom` files | Directory of `*.loom`; set `data.input_type: loom` |

### Sample folder names

Use **three hyphen-separated segments**: `Time-Condition-Replicate` (e.g. `1w-Disease-SingleCell`). The **second segment** is the condition/state (e.g. `Disease`, `Ctrl`) and must match ISP `perturbation.start_state` / `end_state` when mapped to `disease` via tokenize metadata.

### Raw counts

- **Unnormalized** 10x raw counts only.
- All three matrix files per sample folder.

---

## Configure `config/pipeline.yaml`

Edit [`core/config/pipeline.yaml`](../core/config/pipeline.yaml) on the host before running. In the container it is `/app/core/config/pipeline.yaml` (repo bind-mounted at `/app`).

You only need to set **`data.input_dir`** and **`perturbation`** in most cases. Tokenize, fine-tune, and ISP paths are filled under `{paths.output_root}/{DATE}/pipeline_<UTC>/` automatically ([`pipeline_lib.py`](../core/pipeline_lib.py)).

### Example

```yaml
data:
  input_type: single-cell
  input_dir: "/app/data/MyExperiment/"
  output_prefix: null   # null → basename of input_dir (e.g. MyExperiment_0.dataset)

paths:
  output_root: /app/output
  pretrained_model: /app/models/mouse-Geneformer/

runtime:
  nproc: 16
  max_cells: 300000
  forward_batch_size: auto
  # train_batch_size: 32   # from FT batch calibrate; keep fixed once chosen

perturbation:
  type: delete  # delete | overexpress
  state_key: disease
  start_state: Disease   # must match 2nd segment of sample folder names
  end_state: Ctrl
  alt_states: []
  genes_to_perturb: []

stages:
  tokenize: {}
  finetune: {}
  isp: {}
```

### `data`

| Field | Role |
|-------|------|
| `input_type` | `single-cell` (10x folders) or `loom` |
| `input_dir` | Study root — see [Data input rules](#data-input-rules) |
| `output_prefix` | Dataset basename → `{prefix}_0.dataset`; `null` uses folder basename |

### `paths`

| Field | Role |
|-------|------|
| `output_root` | Parent for `pipeline_<UTC>/` run folders (default `/app/output`) |
| `pretrained_model` | Checkpoint for fine-tuning (default `/app/models/mouse-Geneformer/`) |

### `runtime`

Copied into each stage config under `stage_configs/` when the pipeline runs.

| Field | Applied to |
|-------|------------|
| `nproc` | **Tokenize** (`tokenizer.nproc`), **Fine-tune** (`runtime.nproc`), **ISP** (`runtime.nproc`) — CPU workers for HuggingFace `datasets` map/filter |
| `max_cells` | **Tokenize** (`tokenizer.max_cells`) — cells per dataset shard (bounds tokenizer RAM); fine-tune and ISP use all shards. Cross-species input also keeps at most `max_cells` cells per sample file |
| `forward_batch_size` | **ISP** (`runtime.forward_batch_size`) — GPU minibatch for transformer forwards (`auto` or int) |
| `train_batch_size` | **Fine-tune** (`training.batch_size`) — prefer Web UI **FT batch size (calibrate)**; keep fixed once chosen |

Standalone jobs (`docker compose run tokenize` / `finetune` / `isp`) still use [`core/config/tokenize.yaml`](../core/config/tokenize.yaml), [`core/config/finetune.yaml`](../core/config/finetune.yaml), and [`core/config/isp.yaml`](../core/config/isp.yaml) directly.

### `perturbation`

Copied into the generated ISP config. Same keys as [`core/config/isp.yaml`](../core/config/isp.yaml).

| Field | Role |
|-------|------|
| `type` | Perturbation mode: `delete` (remove gene from rank encoding) or `overexpress` (gene to front). `inhibit` / `activate` (quartile shifts) are not implemented and stop with an error |
| `state_key` | Metadata column for states (e.g. `disease`) |
| `start_state` / `end_state` | **Exact** strings as in the tokenized dataset (from folder names when `extract_metadata_from_path` is on) |
| `organ_data` | Optional output filename prefix |
| `genes_to_perturb` | Optional gene list; empty = genome-wide. E2E always runs **TOP1 significant ISP UMAP** once under `{run}/isp_umap/` (largest shift toward goal among Sig == 1 genes with `N_Detections` ≥ 20) (full trajectory UMAP: use Run type **ISP UMAP**) |

### `stages` (optional overrides)

Merged onto the default templates [`core/config/tokenize.yaml`](../core/config/tokenize.yaml), [`core/config/finetune.yaml`](../core/config/finetune.yaml), and [`core/config/isp.yaml`](../core/config/isp.yaml). Written copies land in `stage_configs/` under the run folder.

Example — add metadata before fine-tune without editing the global finetune template:

```yaml
stages:
  finetune:
    metadata:
      add_columns:
        organ_major: brain
```

Example — cap ISP cells:

```yaml
stages:
  isp:
    isp:
      max_ncells: 50000
```

Example: analyses after ISP (Web UI **Cluster / cell-type / enrichment analysis**, default off). One switch turns on the cluster / cell-type analysis after ISP UMAP (`isp_umap/.../cluster_coexpr_analysis/`, [isp_umap.md](isp_umap.md)) and rank enrichment (`rank_enrichment/`). Either can be left out on its own:

```yaml
stages:
  isp:
    postprocess:
      enabled: true
      celltype_prediction: true
      rank_enrichment: true
    rank_enrichment:          # optional, defaults shown
      top_fraction: 0.1
      background: both        # detected in >= min_detection cells of Start and Goal (either: of one)
      min_detection: 5
      fdr: 0.05
      max_cells_per_state: 3000
```

**Rank enrichment** needs per-gene ISP statistics (genome-wide, or a gene list scored per gene). The top `top_fraction` of the background genes by Start pseudobulk rank, by Goal pseudobulk rank, and the ISP hits shifted toward Goal with FDR < `fdr` (ranked by shift, at most as many as the Start list) go into an over-representation test on local Reactome 2024, GO BP, MF and CC 2025, and KEGG libraries (no web service). The ISP list is tested against the background genes ISP scored. Pseudobulk rank = mean over cells of position / (length − 1), 1 where the gene is not encoded, from up to `max_cells_per_state` Start and Goal cells of the ISP dataset. Figures: `rank_enrichment/figures/enrich_<library>.png` (and `.pdf`), one panel each for Start, Goal and ISP. Recompute with other settings: `python core/run_isp_rank_enrichment.py --run-dir <run> --out-dir <dir> --top-fraction 0.05`. `core/run_isp.py` takes `--enable-postprocess`, `--skip-postprocess` and `--no-rank-enrichment`.

### What you do not set manually

The pipeline resolves these per run:

| Derived path | Location |
|--------------|----------|
| Loom temp | `{run}/loom_files/` |
| Tokenized dataset | `{run}/tokenized_dataset/{study}_0.dataset`; if the tokenizer split the study into several shards, all of them concatenated in order as `{study}_merged.dataset` |
| Fine-tune checkpoint | `{run}/finetune/{group}_run{k}/` (`all_run1` by default); with `training.num_runs > 1`, the run with the highest eval accuracy |
| ISP results | `{run}/isp_results/`, `ispstats_results/`, `figures/` |
| Fine-tune UMAP (when enabled) | `{run}/finetune/figures/*_umap.pdf` |
| ISP UMAP (TOP1 significant, E2E standard) | `{run}/isp_umap/` — `umap_*.png`, `per_cell_isp_shift.csv` |
| Rank enrichment (when `postprocess` is on) | `{run}/rank_enrichment/` — `figures/enrich_<library>.png`, `top_genes.csv`, `gene_ranks.csv`, `enrichment/` |

Inspect resolved values after a run in `pipeline_resolved_paths.yaml`.

---

## Output layout

Each run creates:

```text
{output_root}/{YYYYMMDD}/pipeline_{HHMMSS}_{micro}Z/
  pipeline_config_used.yaml      # config as run: CLI ortholog options and derived output_prefix applied
  pipeline_resolved_paths.yaml
  pipeline_run.log
  stage_configs/
    tokenize.yaml
    finetune.yaml
    isp.yaml
  loom_files/                    # if input_type: single-cell
  tokenized_dataset/
    {study_name}_0.dataset       # _1, _2, … when the study exceeds tokenizer.max_cells
    {study_name}_merged.dataset  # only with several shards; used by fine-tune and ISP
  finetune/
    all_run1/                    # checkpoint → ISP
  isp_results/
  ispstats_results/
  figures/
  rank_enrichment/               # only with postprocess.enabled
```

The tokenizer starts a new shard once the cells read so far reach `max_cells`, so large studies give several shards. Fine-tune and ISP use every shard.

`num_classes` for ISP is read from the selected run's `label_dict.json` after fine-tuning. With `training.num_runs > 1`, ISP uses the run with the highest eval accuracy. `finetune.organ_key` (one model per group) stops the pipeline before ISP, because ISP needs a single model.

---

## Resume / partial runs

Inside the container (or with repo mounted at `/app`):

```bash
python3 run_pipeline.py --run-dir <run> --skip-tokenize    # reuse that run's tokenized shards
python3 run_pipeline.py --run-dir <run> --skip-finetune    # reuse that run's fine-tuned model
python3 run_pipeline.py --skip-isp         # stop after fine-tuning
```

`--skip-tokenize` without `--run-dir` (or `paths.run_dir`) stops at once: a new run folder has no tokenized dataset.

---

## Related docs

| Stage | Doc |
|-------|-----|
| Tokenize | [tokenization.md](tokenization.md) |
| Fine-tune | [fine-tuning.md](fine-tuning.md) |
| ISP | [in-silico pertabation.md](in-silico%20pertabation.md) |
| Web UI | [README § Streamlit](../README.md#streamlit-web-ui) · [web-ui.md](web-ui.md) |
