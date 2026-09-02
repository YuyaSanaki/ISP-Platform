# Sequential ISP — chained OE / KD

Apply an **ordered list of group perturbations** (overexpression and/or knockdown) on start-state cells. Each step edits the previous step’s rank-value encoding (`input_ids`), then scores `goal_state_shift` toward the goal state.

Entrypoint: [`core/run_sequential_isp.py`](../core/run_sequential_isp.py). Template: [`core/config/sequential_isp.yaml`](../core/config/sequential_isp.yaml). Compose: `docker compose run --rm sequential_isp`. Web UI: Run type **Sequential ISP**.

This is **not** the same as simultaneous group ISP (all genes in one cocktail). Sequential OE of A then B puts **B leftmost** (highest rank); simultaneous OE of `[A, B]` puts **A leftmost**.

---

## Configure `sequential.steps`

```yaml
sequential:
  save_intermediate_datasets: false
  steps:
    - name: oskm4
      type: overexpress   # OE | overexpress
      genes: [Pou5f1, Sox2, Klf4, Myc]
    - name: followup_kd
      type: delete        # KD | delete | knockdown
      genes: [Igfbp2]
```

| Field | Meaning |
|-------|---------|
| `type` | `overexpress` — length-preserving move-to-front. `delete` — remove those tokens (length shrinks). |
| `genes` | Symbols or Ensembl IDs (same resolver as ISP UMAP). All genes in a step are a **group**. |
| `name` | Optional folder tag (`steps/step01_<name>/`). |
| `save_intermediate_datasets` | Default **false**. Turn on only if you need the perturbed `.dataset` for UMAP. |

`perturbation.start_state` / `end_state` / `state_key` are the usual ISP labels. Paths point at an existing tokenized `.dataset` and a fine-tuned checkpoint (typically from a Pipeline run).

### OSKM permutation mode (research)

If `sequential.steps` is omitted, the runner keeps the BBRC 24-order Yamanaka OE sweep (`orders: [O-S-K-M, …]` or all 24). See `analysis/bbrc_oskm/configs/sequential_oskm/`.

---

## Run

```bash
docker compose run --rm sequential_isp
# or
python3 core/run_sequential_isp.py --config core/config/sequential_isp.yaml
python3 core/run_sequential_isp.py --config … --forward-batch-size auto --max-ncells 500
```

Web UI: pick a past **Pipeline (E2E)** folder (dataset + FT model + states), add ordered OE/KD steps, **Run job**. Outputs go under `{pipeline_run}/sequential_isp/`.

CLI with the default template writes `{output_root}/{YYYYMMDD}/sequential_isp_<UTC>/` unless `paths.output_time_subdir` / `output_date_subdir` are false.

Per-step files:

- `steps/stepNN_<name>/single_gene_per_cell_shifts.csv`
- `step_summary.csv` / `order_summary.csv`
- `run_manifest.json`

---

## `batch=auto` and OOM

Sequential scoring always runs **group** ISP: a perturbed forward **and** an original-encoding forward. Genome-wide ISP auto-calibration probes **one** forward and caches that batch size. Reusing it for sequential is a common CUDA OOM, especially on unified-memory GPUs (GB10) where `dataset.map` workers share the same RAM pool as VRAM.

Sequential ISP therefore:

1. Uses a **separate auto cache** (`task=sequential_isp_group`) with a **dual-forward** probe.
2. Releases the first hidden-state stack before the original forward (`quant_cos_sims`).
3. Maps the dataset with **one process** while the model is on GPU.
4. Calls `empty_cache` between steps.
5. On CUDA OOM, **halves** `forward_batch_size` and retries scoring.

Leave `runtime.forward_batch_size: auto` unless you already know a safe integer. Do not copy a genome-wide ISP batch size into sequential YAML.
