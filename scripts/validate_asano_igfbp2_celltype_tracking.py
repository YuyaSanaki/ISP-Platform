#!/usr/bin/env python3
"""Validate ISP UMAP cell-type trajectory tracking on Asano PIPseq (1w).

Runs Igfbp2 delete ISP UMAP with postprocess for:
  - mouse Geneformer (native)
  - human Geneformer V2-104M (cross-species ortholog path)

Designed for spark-943a / H100 hosts that already have Asano ``data/1w`` and
pretrained weights. Prefer reusing an existing Pipeline run's dataset+checkpoint
when found; otherwise tokenize→finetune→ISP for each model (expensive).

Examples::

  python3 scripts/validate_asano_igfbp2_celltype_tracking.py --discover-only
  python3 scripts/validate_asano_igfbp2_celltype_tracking.py --umap-only
  python3 scripts/validate_asano_igfbp2_celltype_tracking.py --max-cells 500
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _first_existing(*candidates: Path) -> Path | None:
    for p in candidates:
        if p is not None and p.exists():
            return p
    return None


def discover_layout(isp_root: Path) -> dict[str, Any]:
    """Locate Asano 1w data, models, and prior pipeline runs under isp_root / siblings."""
    home = Path.home()
    roots = [
        isp_root,
        home / "ISP-Platform",
        home / "20260916AsanoISP" / "ISP-Platform",
        home / "20260916AsanoISP",
    ]
    roots = [r.resolve() for r in roots if r.exists()]

    data_1w = None
    for r in roots:
        for cand in (
            r / "data" / "1w",
            r / "data" / "Asano" / "1w",
            r / "1w",
        ):
            if cand.is_dir() and any(cand.glob("1w-*")):
                data_1w = cand
                break
        if data_1w:
            break

    mouse_model = _first_existing(
        *(r / "models" / "mouse-Geneformer" for r in roots),
        isp_root / "models" / "mouse-Geneformer",
    )
    human_model = _first_existing(
        *(r / "models" / "human-Geneformer-V2-104M" for r in roots),
        isp_root / "models" / "human-Geneformer-V2-104M",
    )

    def _find_pipeline_runs(output_roots: list[Path]) -> list[Path]:
        found: list[Path] = []
        for out in output_roots:
            if not out.is_dir():
                continue
            for p in out.glob("**/pipeline_*/stage_configs/isp.yaml"):
                found.append(p.parent.parent)
        # Newest first
        found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return found

    output_roots = []
    for r in roots:
        for name in ("output", "outputs"):
            cand = r / name
            if cand.is_dir():
                output_roots.append(cand)
        for cand in r.glob("output*"):
            if cand.is_dir() and cand not in output_roots:
                output_roots.append(cand)

    runs = _find_pipeline_runs(output_roots)

    def _classify_run(run_dir: Path) -> str | None:
        isp_yaml = run_dir / "stage_configs" / "isp.yaml"
        if not isp_yaml.exists():
            return None
        text = isp_yaml.read_text(encoding="utf-8", errors="ignore").lower()
        # Prefer explicit model id
        if "human_geneformer" in text or "human-geneformer" in text:
            return "human"
        if "mouse_geneformer" in text or "mouse-geneformer" in text:
            return "mouse"
        # Fallback: path hints
        s = str(run_dir).lower()
        if "human" in s:
            return "human"
        if "mouse" in s:
            return "mouse"
        return None

    mouse_run = next((r for r in runs if _classify_run(r) == "mouse"), None)
    human_run = next((r for r in runs if _classify_run(r) == "human"), None)

    return {
        "isp_root": str(isp_root),
        "search_roots": [str(r) for r in roots],
        "data_1w": str(data_1w) if data_1w else None,
        "mouse_model": str(mouse_model) if mouse_model else None,
        "human_model": str(human_model) if human_model else None,
        "mouse_pipeline_run": str(mouse_run) if mouse_run else None,
        "human_pipeline_run": str(human_run) if human_run else None,
        "pipeline_runs_found": len(runs),
        "gpu": _gpu_ok(),
    }


def _gpu_ok() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return shutil.which("nvidia-smi") is not None


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def run_umap_for_pipeline(
    run_dir: Path,
    *,
    gene: str,
    out_dir: Path,
    max_cells: int | None,
    enable_postprocess: bool = True,
) -> Path:
    """Invoke run_isp_umap.py against an existing pipeline run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(CORE / "run_isp_umap.py"),
        "--run-dir",
        str(run_dir),
        "--gene",
        gene,
        "--output-dir",
        str(out_dir),
    ]
    if enable_postprocess:
        cmd.extend(["--enable-postprocess", "--postprocess-celltype"])
    if max_cells is not None:
        # CLI may not expose max-cells; patch via env-less YAML override if present.
        # Prefer editing a temp config when --max-cells unsupported.
        pass
    _run(cmd, cwd=ROOT)

    # Soft-fail: if trajectory plot missing because joint overlays lacked labels,
    # try annotate + refresh.
    traj = out_dir / "cluster_coexpr_analysis" / "umap_celltype_trajectories.png"
    summary = out_dir / "cluster_coexpr_analysis" / "celltype_shift_summary.csv"
    if not traj.exists() or not summary.exists():
        cfg = run_dir / "stage_configs" / "isp_umap.yaml"
        if not cfg.exists():
            cfg = run_dir / "stage_configs" / "isp.yaml"
        if cfg.exists() and (out_dir / "per_cell_isp_shift.csv").exists():
            try:
                _run(
                    [
                        sys.executable,
                        str(CORE / "annotate_isp_umap_celltypes.py"),
                        "--run-dir",
                        str(out_dir),
                        "--config",
                        str(cfg),
                        "--refresh-overlays",
                    ],
                    cwd=ROOT,
                )
            except subprocess.CalledProcessError as exc:
                print(f"WARN: annotate/refresh failed: {exc}", flush=True)
        # Direct trajectory plot from whatever CSV we have.
        if not traj.exists():
            try:
                _run(
                    [
                        sys.executable,
                        str(CORE / "plot_isp_umap_celltype_trajectories.py"),
                        "--run-dir",
                        str(out_dir),
                    ],
                    cwd=ROOT,
                )
            except subprocess.CalledProcessError as exc:
                print(f"WARN: trajectory plot failed: {exc}", flush=True)
    return out_dir


def _write_pipeline_yaml(
    dest: Path,
    *,
    data_1w: Path,
    output_root: Path,
    model: str,
    max_cells: int,
    epochs: int,
) -> Path:
    import yaml

    if model == "mouse":
        species = {
            "model_organism": "mouse",
            "model": "mouse_geneformer",
            "mouse_variant": "base",
        }
        ft_batch = 4
        isp_batch = 50
    else:
        species = {
            "model_organism": "mouse",
            "model": "human_geneformer",
            "human_variant": "v2_104m",
            "ortholog_policy": "one2one",
        }
        ft_batch = 2
        isp_batch = 25

    cfg: dict[str, Any] = {
        "data": {
            "input_type": "single-cell",
            "input_dir": str(data_1w),
            "output_prefix": "Asano_1w",
        },
        "paths": {"output_root": str(output_root)},
        "species": species,
        "runtime": {
            "nproc": max(1, min(8, (os.cpu_count() or 4))),
            "max_cells": int(max_cells),
            "forward_batch_size": isp_batch,
        },
        "perturbation": {
            "type": "delete",
            "state_key": "disease",
            "start_state": "AD",
            "end_state": "WT",
            "genes_to_perturb": ["Igfbp2"],
        },
        "stages": {
            "finetune": {
                "training": {
                    "epochs": int(epochs),
                    "warmup_ratio": None,
                    "warmup_steps": 500,
                    "batch_size": ft_batch,
                },
                "runtime": {"dataloader_num_workers": 0, "batch_size": ft_batch},
                "metadata": {"add_columns": {"organ_major": "brain"}},
            },
            "isp": {
                "runtime": {"forward_batch_size": isp_batch},
                "umap": {
                    "enabled": True,
                    "max_cells_per_state": min(2000, int(max_cells)),
                },
                "postprocess": {
                    "enabled": True,
                    "n_clusters": 4,
                    "celltype_prediction": True,
                    "prefer_metadata_celltype": False,
                    "celltype_rank_weights": True,
                },
            },
        },
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return dest


def run_full_pipeline(cfg_path: Path) -> Path:
    """Run tokenize→finetune→ISP; return newest pipeline_* under output_root."""
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    output_root = Path(cfg["paths"]["output_root"])
    before = {p.resolve() for p in output_root.glob("**/pipeline_*") if p.is_dir()}
    _run([sys.executable, str(CORE / "run_pipeline.py"), "--config", str(cfg_path)], cwd=ROOT)
    after = [p for p in output_root.glob("**/pipeline_*") if p.is_dir() and p.resolve() not in before]
    if after:
        return max(after, key=lambda p: p.stat().st_mtime)
    # Fallback: newest overall
    all_runs = list(output_root.glob("**/pipeline_*"))
    if not all_runs:
        raise FileNotFoundError(f"No pipeline_* under {output_root}")
    return max(all_runs, key=lambda p: p.stat().st_mtime)


def collect_artifacts(umap_dir: Path, dest: Path, label: str) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    wanted = [
        "umap_*.png",
        "per_cell_isp_shift.csv",
        "cluster_coexpr_analysis/umap_celltype_trajectories.png",
        "cluster_coexpr_analysis/celltype_shift_summary.csv",
        "cluster_coexpr_analysis/l2_by_coarse_celltype.png",
        "cluster_coexpr_analysis/l2_mean_by_coarse_celltype.png",
        "cluster_coexpr_analysis/l2_mean_by_pred_celltype.png",
        "cluster_coexpr_analysis/umap_joint_l2_cluster_celltype.png",
    ]
    copied: list[Path] = []
    for pattern in wanted:
        for src in umap_dir.glob(pattern):
            target = dest / f"{label}__{src.name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            copied.append(target)
            print(f"copied {src} -> {target}", flush=True)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isp-root",
        type=Path,
        default=ROOT,
        help="ISP-Platform checkout (default: repo root)",
    )
    parser.add_argument("--gene", default="Igfbp2")
    parser.add_argument("--max-cells", type=int, default=5000, help="Cap for full pipeline max_cells")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Print discovered paths as JSON and exit",
    )
    parser.add_argument(
        "--umap-only",
        action="store_true",
        help="Only run ISP UMAP+postprocess on existing pipeline runs (no FT)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mouse", "human"],
        choices=["mouse", "human"],
        help="Which Geneformer backends to validate",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Where to copy figures/CSVs (default: output/asano_igfbp2_celltype_validation/<stamp>)",
    )
    args = parser.parse_args()

    isp_root = args.isp_root.resolve()
    info = discover_layout(isp_root)
    print(json.dumps(info, indent=2), flush=True)
    if args.discover_only:
        return

    stamp = _utc_stamp()
    artifact_dir = args.artifact_dir or (
        isp_root / "output" / "asano_igfbp2_celltype_validation" / stamp
    )
    artifact_dir = artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "discovery.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    results: dict[str, Any] = {"artifact_dir": str(artifact_dir), "models": {}}

    for model in args.models:
        run_key = f"{model}_pipeline_run"
        run_dir_s = info.get(run_key)
        run_dir = Path(run_dir_s) if run_dir_s else None

        if run_dir is None or not run_dir.exists():
            if args.umap_only:
                print(f"SKIP {model}: no existing pipeline run and --umap-only set", flush=True)
                results["models"][model] = {"status": "skipped_no_run"}
                continue
            if not info.get("data_1w"):
                raise SystemExit(
                    f"No Asano data/1w found and no existing {model} pipeline run. "
                    "Place 1w-AD-* / 1w-WT-* under data/1w or pass a host with prior runs."
                )
            if model == "mouse" and not info.get("mouse_model"):
                raise SystemExit("mouse-Geneformer weights not found under models/")
            if model == "human" and not info.get("human_model"):
                raise SystemExit("human-Geneformer-V2-104M weights not found under models/")
            if not info.get("gpu"):
                print("WARN: CUDA not detected; full pipeline may be very slow or fail", flush=True)

            out_root = isp_root / "output" / f"asano_1w_{model}_igfbp2_{stamp}"
            cfg_path = artifact_dir / f"pipeline_asano_{model}.yaml"
            _write_pipeline_yaml(
                cfg_path,
                data_1w=Path(info["data_1w"]),
                output_root=out_root,
                model=model,
                max_cells=args.max_cells,
                epochs=args.epochs,
            )
            print(f"=== Full pipeline ({model}) ===", flush=True)
            run_dir = run_full_pipeline(cfg_path)

        umap_out = artifact_dir / f"isp_umap_{model}_{args.gene}"
        print(f"=== ISP UMAP + cell-type tracking ({model}) run_dir={run_dir} ===", flush=True)
        run_umap_for_pipeline(
            run_dir,
            gene=args.gene,
            out_dir=umap_out,
            max_cells=args.max_cells,
            enable_postprocess=True,
        )
        copied = collect_artifacts(umap_out, artifact_dir / "figures", label=model)
        results["models"][model] = {
            "status": "ok",
            "pipeline_run": str(run_dir),
            "umap_dir": str(umap_out),
            "artifacts": [str(p) for p in copied],
        }

    (artifact_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)

    # Require trajectory artifacts for success when model ran.
    missing = []
    for model, meta in results["models"].items():
        if meta.get("status") != "ok":
            continue
        figs = artifact_dir / "figures"
        if not list(figs.glob(f"{model}__umap_celltype_trajectories.png")):
            missing.append(f"{model}:umap_celltype_trajectories.png")
        if not list(figs.glob(f"{model}__celltype_shift_summary.csv")):
            missing.append(f"{model}:celltype_shift_summary.csv")
    if missing:
        raise SystemExit("Validation incomplete; missing: " + ", ".join(missing))
    print("VALIDATION_OK", flush=True)


if __name__ == "__main__":
    main()
