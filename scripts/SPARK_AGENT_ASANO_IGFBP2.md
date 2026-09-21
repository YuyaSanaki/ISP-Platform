# DGX Spark agent brief — Asano PIPseq Igfbp2 cell-type tracking

Copy-paste this into a **Cursor Cloud Agent on My Machine `spark-943a`**
(worker: `~/20260916AsanoISP/ISP-Platform` or `~/ISP-Platform`).

Repo / branch: `YuyaSanaki/ISP-Platform` @ `cursor/isp-umap-celltype-tracking-d63e`  
PR: https://github.com/YuyaSanaki/ISP-Platform/pull/1

---

## Goal

Asano **PIPseq only** (`data/1w`, AD vs WT). Run **Igfbp2 delete** ISP UMAP with the new postprocess for:

1. **mouse** Geneformer (native)
2. **human** Geneformer V2-104M (mouse→human ortholog)

Produce and collect:

| Artifact | Meaning |
|----------|---------|
| `*__umap_celltype_trajectories.png` | Per-cell arrows + mean vectors by cell type |
| `*__celltype_shift_summary.csv` | Mean/median `shift_l2` / `umap_shift_l2` by type |
| `*__l2_mean_by_pred_celltype.png` | Mean±SEM bar |
| `*__l2_by_coarse_celltype.png` | Boxplots (if coarse labels exist) |
| `discovery.json` / `results.json` | Paths + status |

Fix code if marker panels / trajectories fail (especially human / cross-species). Commit fixes on the same branch and push.

---

## Paste prompt (for spark agent)

```text
You are on DGX spark-943a. Validate ISP UMAP cell-type trajectory tracking on Asano PIPseq only.

Repo: ISP-Platform. Branch: cursor/isp-umap-celltype-tracking-d63e
(PR https://github.com/YuyaSanaki/ISP-Platform/pull/1).

Steps:
1. cd into ~/20260916AsanoISP/ISP-Platform or ~/ISP-Platform (whichever has Asano data/models).
2. git fetch origin cursor/isp-umap-celltype-tracking-d63e && git checkout cursor/isp-umap-celltype-tracking-d63e && git pull
3. Activate the usual GPU env (conda isp / H100/env.sh if present). Confirm nvidia-smi works.
4. python3 scripts/validate_asano_igfbp2_celltype_tracking.py --discover-only
   - Must find data_1w (1w-AD-* / 1w-WT-*) and preferably existing mouse/human pipeline runs.
5. Prefer reuse: python3 scripts/validate_asano_igfbp2_celltype_tracking.py --umap-only --models mouse human
   If no prior runs: full pipeline is OK with --max-cells 5000 --epochs 1 (Igfbp2 delete, postprocess on).
6. Copy figures into /opt/cursor/artifacts/screenshots/ (or workspace artifacts dir) with clear names:
   asano_mouse_umap_celltype_trajectories.png, asano_human_umap_celltype_trajectories.png,
   and the two celltype_shift_summary.csv files.
7. If human markers fail / all Ambiguous / trajectories missing: fix core/isp_umap_celltype.py or
   plot_isp_umap_celltype_trajectories.py, re-run, commit+push on this branch.
8. Report absolute paths of discovery.json, results.json, and the four key figure/CSV artifacts.

Do NOT use smoke data. Asano PIPseq 1w only.
```

---

## Commands (host)

```bash
# 1) Checkout
export ISP_ROOT="${HOME}/20260916AsanoISP/ISP-Platform"
# fallback: export ISP_ROOT="${HOME}/ISP-Platform"
cd "$ISP_ROOT"
git fetch origin cursor/isp-umap-celltype-tracking-d63e
git checkout cursor/isp-umap-celltype-tracking-d63e
git pull --ff-only origin cursor/isp-umap-celltype-tracking-d63e

# 2) GPU env (pick what exists on this machine)
# source "$ISP_ROOT/H100/env.sh" && h100_activate
# or: conda activate isp
nvidia-smi -L

# 3) Discover Asano layout
python3 scripts/validate_asano_igfbp2_celltype_tracking.py --discover-only

# 4a) Fast path — reuse existing tokenize/FT/ISP runs
python3 scripts/validate_asano_igfbp2_celltype_tracking.py \
  --umap-only --models mouse human

# 4b) Slow path — no prior runs (tokenize → FT 1 epoch → ISP Igfbp2 delete → UMAP postprocess)
python3 scripts/validate_asano_igfbp2_celltype_tracking.py \
  --models mouse human --max-cells 5000 --epochs 1
```

Outputs default to:

```text
$ISP_ROOT/output/asano_igfbp2_celltype_validation/<UTC_STAMP>/
  discovery.json
  results.json
  figures/mouse__umap_celltype_trajectories.png
  figures/human__umap_celltype_trajectories.png
  figures/mouse__celltype_shift_summary.csv
  figures/human__celltype_shift_summary.csv
  isp_umap_mouse_Igfbp2/   # full UMAP run dir
  isp_umap_human_Igfbp2/
```

Success prints `VALIDATION_OK`.

---

## Manual equivalent (if script discover fails)

```bash
# After locating RUN_MOUSE / RUN_HUMAN (pipeline_* with stage_configs/isp.yaml)
python3 core/run_isp_umap.py \
  --run-dir "$RUN_MOUSE" --gene Igfbp2 \
  --enable-postprocess --postprocess-celltype \
  --output-dir "$ISP_ROOT/output/manual_umap_mouse_Igfbp2"

python3 core/run_isp_umap.py \
  --run-dir "$RUN_HUMAN" --gene Igfbp2 \
  --enable-postprocess --postprocess-celltype \
  --output-dir "$ISP_ROOT/output/manual_umap_human_Igfbp2"
```

Expected under each `--output-dir/cluster_coexpr_analysis/`:

- `umap_celltype_trajectories.png`
- `celltype_shift_summary.csv`
- `l2_mean_by_pred_celltype.png`

---

## Config knobs (already in branch)

```yaml
postprocess:
  enabled: true
  celltype_prediction: true
  prefer_metadata_celltype: false  # ignore user-filled cell_type metadata
  celltype_rank_weights: true      # rank-weighted marker scores
```

Species blocks:

```yaml
# mouse native
species: { model_organism: mouse, model: mouse_geneformer, mouse_variant: base }

# human model on mouse PIPseq
species:
  model_organism: mouse
  model: human_geneformer
  human_variant: v2_104m
  ortholog_policy: one2one
```

Perturbation: `type: delete`, `genes_to_perturb: [Igfbp2]`, `state_key: disease`, `start_state: AD`, `end_state: WT` (match folder names).

---

## Done when

- [ ] `nvidia-smi` OK on spark
- [ ] `discovery.json` shows `data_1w` and/or both pipeline runs
- [ ] mouse + human trajectory PNGs + shift CSVs exist
- [ ] human run is not all `Unknown`/`Ambiguous` solely due to mouse-only markers (species-aware panel should apply)
- [ ] Artifacts copied for the walkthrough; any fixes pushed to `cursor/isp-umap-celltype-tracking-d63e`
