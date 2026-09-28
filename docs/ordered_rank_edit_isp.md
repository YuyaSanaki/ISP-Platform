# Ordered rank-edit ISP — ordered OE / KD steps

Apply an **ordered list of group perturbations** (overexpression and/or knockdown) on start-state cells. Each step edits the previous step’s rank-value encoding (`input_ids`), then scores `goal_state_shift` toward the goal state against the **original** start encoding. Only the edited token ranks carry over to the next step; embeddings and shifts are never fed back.

Entrypoint: [`core/run_ordered_rank_edit_isp.py`](../core/run_ordered_rank_edit_isp.py). Template: [`core/config/ordered_rank_edit_isp.yaml`](../core/config/ordered_rank_edit_isp.yaml). Compose: `docker compose run --rm ordered_rank_edit_isp`. Web UI: Run type **Ordered rank-edit ISP**.

Multi-step ISP has two forms on this platform:

| Method | What carries into the next step | Doc |
|--------|---------------------------------|-----|
| **Ordered rank-edit ISP** (this page) | The edited gene-rank tokens only | this page |
| **State-feedback ISP** | A new gene order built from the model output after each step (same gene set) | [state_feedback_decode_methods.md](state_feedback_decode_methods.md) |

How the two differ, and why only the final encoding (which genes, in what order) sets the final shift here, is explained in [ordered_rank_edit_and_state_feedback_isp.md](ordered_rank_edit_and_state_feedback_isp.md) (v1.0.1).

This is **not** the same as simultaneous group ISP (all genes in one cocktail). Ordered OE of A then B puts **B leftmost** (highest rank); simultaneous OE of `[A, B]` puts **A leftmost**.

**Former name.** This feature was called *Sequential ISP*. The old names still work: `core/run_sequential_isp.py`, the Compose service `sequential_isp`, and a `sequential:` block in the config (read when `ordered_rank_edit:` is absent). New runs write to `ordered_rank_edit_isp/` / `ordered_rank_edit_isp_<UTC>/`. Existing `sequential_isp/` output folders are left as they are.

---

## Configure `ordered_rank_edit.steps`

```yaml
ordered_rank_edit:
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

If `ordered_rank_edit.steps` is omitted, the runner keeps the 24-order Yamanaka OE sweep (`orders: [O-S-K-M, …]` or all 24). Pass `--orders` to run a subset.

---

## Run

```bash
docker compose run --rm ordered_rank_edit_isp
# or
python3 core/run_ordered_rank_edit_isp.py --config core/config/ordered_rank_edit_isp.yaml
python3 core/run_ordered_rank_edit_isp.py --config … --forward-batch-size auto --max-ncells 500
```

Web UI: pick a past **Pipeline (E2E)** folder (dataset + FT model + states), add ordered OE/KD steps, **Run job**. Outputs go under `{pipeline_run}/ordered_rank_edit_isp/`.

CLI with the default template writes `{output_root}/{YYYYMMDD}/ordered_rank_edit_isp_<UTC>/` unless `paths.output_time_subdir` / `output_date_subdir` are false.

Per-step files:

- `steps/stepNN_<name>/single_gene_per_cell_shifts.csv`
- `step_summary.csv` / `order_summary.csv`
- `run_manifest.json`

---

## `batch=auto` and OOM

Per-step scoring always runs **group** ISP: a perturbed forward **and** an original-encoding forward. Genome-wide ISP auto-calibration probes **one** forward and caches that batch size. Reusing it here is a common CUDA OOM, especially on unified-memory GPUs (GB10) where `dataset.map` workers share the same RAM pool as VRAM.

Ordered rank-edit ISP therefore:

1. Uses a **separate auto cache** (`task=ordered_rank_edit_isp_group`) with a **dual-forward** probe.
2. Releases the first hidden-state stack before the original forward (`quant_cos_sims`).
3. Maps the dataset with **one process** while the model is on GPU.
4. Calls `empty_cache` between steps.
5. On CUDA OOM, **halves** `forward_batch_size` and retries scoring.

Leave `runtime.forward_batch_size: auto` unless you already know a safe integer. Do not copy a genome-wide ISP batch size into this YAML.
