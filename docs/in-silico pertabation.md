# In-silico perturbation (ISP) — new biology

End-to-end workflow to run Geneformer ISP on a **new experiment** (tissue, genotype, or condition).

Entrypoint: [`run_isp.py`](../core/run_isp.py). Configuration: [`core/config/isp.yaml`](../core/config/isp.yaml). Docker: [`docker-compose.yml`](../docker-compose.yml).

**Prerequisite:** a tokenized `.dataset` — see [**tokenization.md**](tokenization.md).

---

## 1. Define the question

- **Goal-state shift:** in silico perturbation of a gene in start-labeled cells (e.g. `Disease`) shifts the model representation toward end-labeled cells (e.g. `Ctrl`).
- **What it measures:** embedding similarity changes in the model space — not raw fold-change.
- **Data:** tokenized `.dataset` from real scRNA-seq of both states; labels must **exactly** match `perturbation.start_state` and `perturbation.end_state` in `isp.yaml`.
- **Cell types:** subset before tokenization, or filter after with `isp.filter_data` if columns exist in the dataset.

---

## 2. Configure `config/isp.yaml`

Align paths and labels with your tokenized dataset.

| YAML area | What to set |
|-----------|-------------|
| `paths.dataset` | Path to your **`.dataset`** directory |
| `paths.geneformer_model` | Pretrained or fine-tuned checkpoint directory |
| `paths.output_root` | Parent output directory |
| `paths.output_time_subdir` | `true` (default): `{output_root}/{YYYYMMDD}/isp_<UTC>/…`; `false` or `ISP_OUTPUT_TIME_SUBDIR=0` for flat layout under `<DATE>/` |
| `paths.output_subdir` / `ISP_OUTPUT_SUBDIR` | Fixed folder name under `<DATE>` (single segment) |
| `perturbation.state_key` | Metadata column for states (e.g. `disease`) |
| `perturbation.start_state` / `end_state` | **Exact** strings in that column |
| `perturbation.alt_states` | Alternate end labels, or `[]` |
| `isp.filter_data` | Optional, e.g. `{"cell_type": ["skeletal_muscle"]}` |
| `isp.max_ncells` | Cap on cells after filters |
| `runtime.forward_batch_size` / `nproc` | GPU batch size and CPU workers |

### Choosing `model.type`

- **`Pretrained`**: base `BertForMaskedLM` — usual default for stock mouse Geneformer.
- **`CellClassifier`**: fine-tuned `BertForSequenceClassification` (see [fine-tuning.md](fine-tuning.md)) — shifts along your classification axis.
- **`GeneClassifier`**: fine-tuned `BertForTokenClassification` — gene-level tasks.

---

## 3. Run ISP

**Single GPU (default):**

```bash
docker compose run --rm isp
```

Runs `accelerate launch --num_processes 1 /app/core/run_isp.py --config /app/core/config/isp.yaml`.

**Multiple GPUs:**

```bash
ISP_NUM_GPUS=4 docker compose run --rm isp
```

Statistics and figures are consolidated on the **main** process only.

**Alternate configs:**

```bash
docker compose run --rm isp accelerate launch --num_processes 1 /app/core/run_isp.py --config /app/core/config/isp_Ctrl-Disease.yaml
```

**Streamlit:** run type **ISP** or **Pipeline (E2E)** — see [README § Streamlit Web UI](../README.md#streamlit-web-ui).

---

## 4. Outputs and downstream analysis

- **Perturbation outputs:** `{paths.output_root}/{YYYYMMDD}/[isp_<UTC>/]isp_results`
- **Stats (parquet):** `…/ispstats_results`
- **Figures:** `run_isp.py` runs [`isp_analysis.py`](../isp_analysis.py) → PNGs under `…/figures/` (`shift_distribution.png`, `volcano_plot.png`, `top_significant_genes_barplot.png`, `significant_genes_lollipop.png`, etc.). The significant-gene barplot encodes `N_Detections` as color intensity on a √ scale (colorbar ticks show N). The lollipop shows shift direction, effect size, `N_Detections` / `isp.max_ncells`, and FDR. Disable with `analysis.enabled: false` or `--skip-analysis`.

### Per-cell shift (ISP UMAP)

Population ISP ranks **genes**. For per-cell trajectories and **`per_cell_isp_shift.csv`**, see [**isp_umap.md**](isp_umap.md). For an **ordered chain of OE and/or KD** (token-space sequential ISP), see [**sequential_isp.md**](sequential_isp.md).

```bash
docker compose run --rm isp_umap
docker compose run --rm sequential_isp
```

---

## 5. Run provenance, config summary, and rotating logs

On the **main process** only, `run_isp.py` prints a **config summary** (dataset path, model path, perturbation, `isp.max_ncells`, `forward_batch_size`, `nproc`, etc.) and mirrors **stdout** and **stderr** into a **rotating** log next to that run’s outputs:

| Artifact | Location (default: `paths.output_root` + date + `isp_<UTC>/`) | Role |
|----------|---------------------------------------------------------------|------|
| `isp_config_used.yaml` | `./output/<DATE>/isp_<UTC>/` | Copy of the ISP YAML used |
| `isp_run_metadata.yaml` | same | `started_at_utc` at launch; **`finished_at_utc`** and **`run_status`** (`completed` / `failed`) when the run finishes (main process) |
| `isp_run.log` (+ `.1`, `.2`, …) | same | Console capture from Python onward |

| Variable | Meaning |
|----------|---------|
| `ISP_LOG_MAX_BYTES` | Max size of `isp_run.log` before rotation (default `52428800`, 50 MiB) |
| `ISP_LOG_BACKUP_COUNT` | Rotated backups to keep (default `5`; `0` truncates instead of renaming) |
| `ISP_DISABLE_RUN_LOG` | `1` / `true` / `yes` to skip file logging (console unchanged) |
| `ISP_GIT_COMMIT` | Written into `isp_run_metadata.yaml` when `git` is unavailable |
| `ISP_OUTPUT_TIME_SUBDIR` | `1` / `true` / `yes` (default per YAML) or `0` / `false` / `no` to disable `isp_<UTC>/` under each `<DATE>` |
| `ISP_OUTPUT_SUBDIR` | Fixed folder name under `<DATE>` (single path segment) |
| `ISP_OUTPUT_DATE` / `--output-date` | Override output date folder |

**Note:** Lines from `accelerate launch` *before* `run_isp.py` starts appear only in the terminal unless you redirect at the shell level (e.g. `docker compose run ... 2>&1 | tee full_terminal.log`).

### About the run log files (`*.log`)

- **Captured:** stdout/stderr after the tee is installed — prints, HuggingFace progress bars, etc. (UTF-8 text).
- **Not captured:** `accelerate launch` startup before Python attaches the tee.
- **Multi-GPU:** only the **main** process writes `isp_run.log`; other ranks print to the terminal only.
- **Append vs. rotate:** reusing the same output folder **appends** to `isp_run.log`. Default layout uses a new `isp_<UTC>/` per launch. When size exceeds `ISP_LOG_MAX_BYTES`, the log rotates to `.log.1`, `.2`, … up to `ISP_LOG_BACKUP_COUNT`.
- **Provenance:** `*_config_used.yaml` and `*_run_metadata.yaml` are the config snapshot; `.log` is console history for debugging.

Tokenization logs (`tokenize_run.log`, etc.): [**tokenization.md § Run provenance**](tokenization.md#5-run-provenance-and-logs).

Implementation: [`run_isp.py`](../core/run_isp.py), [`run_pipeline_log.py`](../run_pipeline_log.py).

---

## 6. Do I need a new `.dataset`?

| Situation | Action |
|-----------|--------|
| New tissue, genotype, or comparison | **Yes** — [tokenize](tokenization.md), then point `paths.dataset` at the new `.dataset` |
| Same dataset; batch size / cell cap only | Adjust `runtime.*` and/or `isp.max_ncells` |
| Same experiment; paths moved | Update `paths.*` in `isp.yaml` only |

---

## 7. Hard requirements

1. **Same vocabulary** as training: mouse pretrained Geneformer + token dictionary used to build `input_ids`.
2. **Exact string match** between dataset labels and `start_state` / `end_state` / `alt_states` in YAML.

---

## 8. Memory and throughput

- Let PyTorch manage GPU memory between batches; avoid `torch.cuda.empty_cache()` / aggressive `gc.collect()` in hot loops.
- On CUDA OOM, ISP may retry once after cache clear; then lower `runtime.forward_batch_size` or `isp.max_ncells`.
- **`nproc`**: speeds dataset map/filter only, not transformer forwards.

---

## 9. Flow summary

**Design → [tokenize](tokenization.md) → edit `isp.yaml` → `docker compose run --rm isp` → stats / figures → analysis.**
