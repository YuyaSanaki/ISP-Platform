# Tokenization

Convert raw single-cell expression (10x, loom, or AnnData) into a Hugging Face **`.dataset`** directory for fine-tuning and ISP.

Entrypoint: [`execute_tokenizer_pipeline.py`](../core/execute_tokenizer_pipeline.py). Configuration: [`core/config/tokenize.yaml`](../core/config/tokenize.yaml). Docker: `docker compose run --rm tokenize` in [`docker-compose.yml`](../docker-compose.yml).

Install prerequisites (Docker image, `MLM-re_token_dictionary_v1.pkl`): [README § Install](../README.md#install).

---

## 1. Input layout (10x / single-cell)

You need **raw counts** (single-cell or single-nucleus; **do not** normalize before tokenization) for all conditions you will compare in ISP (e.g. `Ctrl` and `Disease`). Minimum ~500 cells per cell type of interest; more is better.

### 10x Cell Ranger / DRAGEN output

Each sample must have all three files:

- `barcodes.tsv.gz`
- `features.tsv.gz`
- `matrix.mtx.gz`

Place them in a folder named with three hyphen-separated segments:

```text
{study_root}/ExperimentName/Time-Condition-Replicate/
  barcodes.tsv.gz
  features.tsv.gz
  matrix.mtx.gz
```

Example: `1w-Disease-SingleCell/`, `1w-Ctrl-SingleCell/` under one study root.

| Field | Rule |
|-------|------|
| Time | First segment (e.g. `1w`, `3w`, `5w`) |
| Condition | Second segment (e.g. `Ctrl`, `Disease`) — used as ISP state when mapped to `disease` |
| Replicate | Third segment (e.g. `Rep1`, `SingleCell`; use a placeholder if only one replicate) |
| `sample_id` | Full folder name |

With **`single_cell_settings.extract_metadata_from_path: true`** in `config/tokenize.yaml`, `time`, `genotype`, `replicate`, `disease`, and `sample_id` are filled from folder names. Adjust naming or the script if your labels differ.

**`data.input_dir`** is the **study root** (parent of sample folders), e.g. `/app/data/my_study/` or `/app/data/ExperimentName/`. See [`data_input_layout.py`](../contracts/data_input_layout.py) for discovery rules (flat samples, nested `ExperimentName/`, multiple experiments under `/data/`).

### Alternative formats

| Format | Notes |
|--------|--------|
| **`.loom`** | Put one or more `*.loom` in `data.input_dir`; set `data.input_type: loom`. |
| **`.h5ad` (AnnData)** | Supported in Geneformer APIs with `file_format="h5ad"`; same field requirements as loom. |

### Required fields (loom / h5ad)

**Genes**

- **`ensembl_id`**: per gene; must match the **selected model** token dictionary (mouse Ensembl, human Ensembl, or fly `FBgn` for drosophila input).

**Cells**

- **`n_counts`**: total UMI/read counts per cell.

**Optional**

- **`filter_pass`**: if missing, all cells are tokenized.

**Metadata for ISP**

- Pass columns through **`tokenizer.custom_attr_name_dict`** (loom/h5ad name → dataset column name).
- Values must match **`perturbation.start_state` / `end_state`** in ISP YAML **exactly** (including casing).

### Expression assumptions

- **Raw counts** only; no prior feature selection.
- Median scaling and rank tokenization happen inside [`TranscriptomeTokenizer`](../geneformer/tokenizer.py).

---

## 2. Configure `config/tokenize.yaml`

| YAML area | What to set |
|-----------|-------------|
| `data.input_type` | `single-cell` (10x folders) or `loom` |
| `data.input_dir` | Study root (see §1) |
| `data.loom_temp_dir` | Where `.loom` files are written when converting from 10x |
| `data.output_dir` | Parent directory for the tokenized `.dataset` |
| `data.output_prefix` | Base name → `{output_prefix}_0.dataset` |
| `tokenizer.custom_attr_name_dict` | Loom/dataset column mapping |
| `tokenizer.celltype_annotation` | Pre-ISP expression cell-type labels (default **on**; see below) |
| `tokenizer.nproc` / `max_cells` | Parallelism and cell cap |
| `tokenizer.report_conversion` | Pre-tokenize ortholog summary (default **true** whenever ortholog conversion applies) |
| `tokenizer.ortholog_audit` | Path to analysis manifest / audit YAML enabling **ortholog_loss_gate** |
| `tokenizer.ortholog_loss_gate` | Force on/off (default: on when audit is set, including same-species → `not_applicable`) |
| `tokenizer.ortholog_approval_record` | NEW `status: approved` YAML matching a prior Block (not the pending request) |
| `species.model_organism` / `species.model` | Input data species vs Geneformer backend; triggers ortholog remap when they differ |
| `species.ortholog_curated_overlay` | Optional project bridge (file or dir with `curated_bridge_{pair}.tsv`) |
| `single_cell_settings.extract_metadata_from_path` | Parse sample folder names (default `true`) |

**`input_type: single-cell`**: builds `.loom` under `data.loom_temp_dir`, then runs `TranscriptomeTokenizer`.

**`input_type: loom`**: reads `*.loom` from `data.input_dir`; loom attributes must match keys in `custom_attr_name_dict`.

### Pre-ISP cell-type annotation

During 10x→loom conversion, the platform can label each cell from the **expression matrix** (not from Geneformer tokens):

```yaml
tokenizer:
  celltype_annotation:
    enabled: true          # default
    # panel: null          # default isp_expression_v1.json
    # min_score: null
    # min_margin: null
  custom_attr_name_dict:
    # …existing attrs…
    cell_type: cell_type
    tissue: tissue
    celltype_score: celltype_score
    celltype_annotator: celltype_annotator
```

- Implementation: [`celltype_annotate_expression.py`](../core/celltype_annotate_expression.py).
- Panel: [`geneformer/dicts/celltype_panels/isp_expression_v1.json`](../core/geneformer/dicts/celltype_panels/isp_expression_v1.json) (mouse + human symbols; organism from `species.model_organism`).
- Provenance column `celltype_annotator=isp_expression_v1` lets ISP UMAP postprocess trust these labels (`prefer_metadata_celltype: auto`) while still ignoring arbitrary user-filled `cell_type` without that provenance.
- Disable with `tokenizer.celltype_annotation: false` or `enabled: false`.

The image resolves token and median dictionaries from `species.model` via `core/geneformer/backends/registry.py` (mouse or human V2). When `model_organism` differs from the model’s native species, genes are remapped through ortholog tables before tokenization (`core/geneformer/gene_converter.py`).

Custom scripts: [`run_tokenizer_ad.py`](../run_tokenizer_ad.py) shows an alternate layout.

### Cross-species gene conversion

Set `species` in `config/tokenize.yaml` (or pipeline config propagated by `run_pipeline.py`):

```yaml
species:
  model_organism: mouse          # mouse | human | drosophila
  model: human_geneformer        # or mouse_geneformer
  human_variant: v2_104m         # when model: human_geneformer
  # mouse_variant: 12l_e20       # when model: mouse_geneformer (base | 12l_e20)
  ortholog_policy: one2one       # one2one | best_of_n | legacy_sum
  # ortholog_curated_overlay: /app/analysis/.../ortholog_policy/v1  # optional project bridge
```

Unmapped / ambiguous genes are dropped under the default `one2one` policy. `best_of_n` keeps the highest-median source when several map to one target; `legacy_sum` restores the old last-write-wins + expression-sum behavior.

Optional **project curated overlays** (do not edit platform `*_curated.tsv`): set `species.ortholog_curated_overlay` to a directory containing `curated_bridge_{pair}.tsv`, or use env `GENEFORMER_ORTHOLOG_CURATED_OVERLAY` / CLI `--ortholog-curated-overlay`. Overlay rows merge after the platform curated table.

Ortholog tables: `core/geneformer/dicts/orthologs/` (download via `scripts/download_mouse_human_orthologs.sh` and `scripts/download_drosophila_orthologs.sh`). Fly remapping typically keeps ~50% of protein-coding genes; mouse↔human is ~92%.

### Ortholog loss gate

After the conversion report and **before** tokenization, an optional quality gate can **Pass / Warn / Block** based on an analysis `ortholog_audit` contract.

**Block rule (strict):** a gene in `critical_sets` is **present in the raw input** (Ensembl / symbol / alias) but **dropped by the mapping policy** (e.g. `ortholog_one2many` under `one2one`).  
**Not Block:** the same critical gene is simply **absent from the input matrix** → `absent_from_input` (**Warn**).  
**Same-species:** verdict **`not_applicable`**; tokenize continues.

The gate **never** auto-adds one-to-many genes to curated overlays. Approving a bridge (`decision: curated_bridge` / **B**) only accepts an already-configured overlay path.

**Fail-closed approval:** Block writes `ortholog_approval_request.yaml` with `status: pending` (not reusable). Create a **new** file with `status: approved`, the same `request_id` + `blocked_run` hashes, then pass that path to `--ortholog-approval-record`. Pending records are always rejected.

**Web UI:** Species / model selectors, worked examples, and the Outputs **approval card** (stop / curated_bridge / reject → new run only for bridge) are documented in [web-ui.md § Species / conversion](web-ui.md#species--conversion-web-ui). The UI does not bypass gate validation.

```yaml
tokenizer:
  ortholog_audit: /app/analysis/analysis_manifest.yaml
  # ortholog_approval_record: /path/to/approved.yaml  # status: approved (not pending)
```

```bash
# Enable via CLI (also works on E2E pipeline)
docker compose run --rm pipeline python3 /app/core/execute_tokenizer_pipeline.py \
  --ortholog-audit /app/analysis/analysis_manifest.yaml

# After Block: create a NEW approved YAML (do not pass the pending request), then re-run
docker compose run --rm pipeline python3 /app/core/execute_tokenizer_pipeline.py \
  --ortholog-audit /app/analysis/analysis_manifest.yaml \
  --ortholog-approval-record /path/to/approved.yaml \
  --ortholog-curated-overlay /app/analysis/ortholog_policy/v1   # needed for curated_bridge

# Skip gate even if audit is configured
docker compose run --rm pipeline python3 /app/core/execute_tokenizer_pipeline.py \
  --no-ortholog-loss-gate
```

| Decision | Alias | Meaning |
|----------|-------|---------|
| **stop** | A | Keep strict one2one; skip tokenize |
| **curated_bridge** | B | Approve curated bridge (overlay must be activated; no auto-write) |
| **sensitivity_best_of_n** | C | Sensitivity / `best_of_n` — manual re-tokenize in v1 |
| **mark_non_critical** | D | Mark genes non-critical for this run (`non_critical_overrides`) |

See `analysis/analysis_manifest.yaml` → `ortholog_audit` in your analysis workspace when using the gate.

---

## 3. Run

```bash
docker compose run --rm tokenize
```

**Conversion report** (mapped / dropped genes before tokenization):

```bash
# Cross-species (mouse↔human, fly→…): report is on by default
docker compose run --rm tokenize

# Disable if needed
docker compose run --rm pipeline python3 /app/core/execute_tokenizer_pipeline.py --no-report-conversion
```

Or set `tokenizer.report_conversion: false` in YAML. E2E pipeline enables the report automatically whenever ortholog conversion applies.

Override config:

```bash
TOKENIZE_CONFIG=/app/core/config/my_tokenize.yaml docker compose run --rm tokenize
```

**Streamlit:** run type **Tokenize**, or **Pipeline (E2E)** (tokenize is stage 1). See [README § Streamlit Web UI](../README.md#streamlit-web-ui).

---

## 4. Output `.dataset`

Saved under `data.output_dir`, typically **`{output_prefix}_0.dataset`** (suffix increments if the run is split).

| Column / field | Role |
|----------------|------|
| `input_ids` | Rank-encoded gene tokens per cell |
| `length` | Sequence length |
| State column | e.g. `disease` — must match ISP `perturbation.state_key` and state strings |
| Other metadata | e.g. `time`, `sample_id` if mapped in `custom_attr_name_dict` |
| `cell_type` / `tissue` / `celltype_score` / `celltype_annotator` | Pre-ISP expression annotation (when `celltype_annotation.enabled`) |

Point **`paths.dataset`** in [`core/config/finetune.yaml`](../core/config/finetune.yaml) or [`core/config/isp.yaml`](../core/config/isp.yaml) at this directory.

---

## 5. Run provenance and logs

[`execute_tokenizer_pipeline.py`](../core/execute_tokenizer_pipeline.py) prints a **config summary**, then runs conversion (if `input_type: single-cell`) and tokenization.

| Artifact | Location | Role |
|----------|----------|------|
| `tokenize_config_used.yaml` | `data.output_dir` | Copy of the tokenize config used |
| `tokenize_run_metadata.yaml` | same | `started_at_utc`; **`finished_at_utc`** and **`run_status`** when the run ends |
| `tokenize_run.log` (+ `.1`, `.2`, …) | same | Rotating mirror of stdout/stderr |
| `conversion_report.json` | same | Ortholog mapping summary (when report enabled) |
| `conversion_report.txt` | same | Human-readable conversion summary |
| `conversion_unmapped_genes.tsv` | same | Dropped genes: `source_id`, `symbol`, `drop_reason`, `brief_function` |
| `conversion_gene_detail.tsv` | same | Per-gene status / candidates (when loss gate runs) |
| `critical_gene_audit.tsv` | same | Critical genes: `input_presence`, mapping status, Block eligibility |
| `ortholog_loss_summary.json` | same | Gate verdict + counts |
| `mapping_provenance.json` | same | Policy, table/overlay hashes, optional Ensembl freeze |
| `ortholog_approval_request.yaml` | same | Written on **Block** with `status: pending` (not reusable) |
| `approval_record.yaml` | same (or beside request) | UI/CLI-authored `status: approved` / `rejected` for `--ortholog-approval-record` |

| Variable | Meaning |
|----------|---------|
| `TOKENIZE_CONFIG` | Override config path |
| `TOKENIZE_LOG_MAX_BYTES` | Max log size before rotation (default 50 MiB) |
| `TOKENIZE_LOG_BACKUP_COUNT` | Rotated backups to keep (default `5`; `0` truncates) |
| `TOKENIZE_DISABLE_RUN_LOG` | Set to `1` / `true` / `yes` to skip file logging |
| `TOKENIZE_GIT_COMMIT` | Optional commit string when `git` is unavailable in the container |

The tee is attached after the output directory exists, so **loom conversion and tokenization** are both recorded. Shared behavior (what is captured, rotation, append vs. new folder): [in-silico pertabation.md § Run provenance and logs](in-silico%20pertabation.md#run-provenance-config-summary-and-rotating-logs).

Implementation: [`execute_tokenizer_pipeline.py`](../core/execute_tokenizer_pipeline.py), [`run_pipeline_log.py`](../run_pipeline_log.py).

---

## 6. Flow summary

**Raw counts + metadata → `config/tokenize.yaml` → (optional expression cell-type annotation) → `docker compose run --rm tokenize` → `{output_prefix}_0.dataset` → fine-tune or ISP / ISP UMAP.**
