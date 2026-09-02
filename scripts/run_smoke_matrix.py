#!/usr/bin/env python3
"""Run the smoke matrix: {mouse,human,fly} data × key Geneformer backends.

Each case runs tokenize → finetune → ISP via run_pipeline.py inside Docker
(or locally if --local). Results + failures are written under output/smoke_matrix/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from geneformer.gene_converter import conversion_pair

ROOT = Path(__file__).resolve().parent.parent

# FBgn0036990 = fly ortholog of mouse Mrpl15 (present in smoke_fly features).
_FLY_PERTURB = ["FBgn0036990"]

CASES = [
    {
        "id": "mouse_data__mouse_base",
        "input_dir": "/app/data/smoke_mouse",
        "output_root": "/app/output/smoke_matrix/mouse_data__mouse_base",
        "species": {
            "model_organism": "mouse",
            "model": "mouse_geneformer",
            "mouse_variant": "base",
        },
        "genes_to_perturb": ["Igfbp2"],
        "forward_batch_size": 8,
        "finetune_batch_size": 4,
        "max_input_size": 2048,
    },
    {
        "id": "mouse_data__mouse_12l",
        "input_dir": "/app/data/smoke_mouse",
        "output_root": "/app/output/smoke_matrix/mouse_data__mouse_12l",
        "species": {
            "model_organism": "mouse",
            "model": "mouse_geneformer",
            "mouse_variant": "12l_e20",
        },
        "genes_to_perturb": ["Igfbp2"],
        "forward_batch_size": 8,
        "finetune_batch_size": 4,
        "max_input_size": 2048,
    },
    {
        "id": "mouse_data__human_v2",
        "input_dir": "/app/data/smoke_mouse",
        "output_root": "/app/output/smoke_matrix/mouse_data__human_v2",
        "species": {
            "model_organism": "mouse",
            "model": "human_geneformer",
            "human_variant": "v2_104m",
        },
        "genes_to_perturb": ["Igfbp2"],
        "forward_batch_size": 4,
        "finetune_batch_size": 2,
        "max_input_size": 4096,
    },
    {
        "id": "human_data__human_v2",
        "input_dir": "/app/data/smoke_human",
        "output_root": "/app/output/smoke_matrix/human_data__human_v2",
        "species": {
            "model_organism": "human",
            "model": "human_geneformer",
            "human_variant": "v2_104m",
        },
        "genes_to_perturb": ["IGFBP2"],
        "forward_batch_size": 4,
        "finetune_batch_size": 2,
        "max_input_size": 4096,
    },
    {
        "id": "human_data__mouse_base",
        "input_dir": "/app/data/smoke_human",
        "output_root": "/app/output/smoke_matrix/human_data__mouse_base",
        "species": {
            "model_organism": "human",
            "model": "mouse_geneformer",
            "mouse_variant": "base",
        },
        "genes_to_perturb": ["IGFBP2"],
        "forward_batch_size": 8,
        "finetune_batch_size": 4,
        "max_input_size": 2048,
    },
    {
        "id": "human_data__mouse_12l",
        "input_dir": "/app/data/smoke_human",
        "output_root": "/app/output/smoke_matrix/human_data__mouse_12l",
        "species": {
            "model_organism": "human",
            "model": "mouse_geneformer",
            "mouse_variant": "12l_e20",
        },
        "genes_to_perturb": ["IGFBP2"],
        "forward_batch_size": 8,
        "finetune_batch_size": 4,
        "max_input_size": 2048,
    },
    {
        "id": "fly_data__mouse_base",
        "input_dir": "/app/data/smoke_fly",
        "output_root": "/app/output/smoke_matrix/fly_data__mouse_base",
        "species": {
            "model_organism": "drosophila",
            "model": "mouse_geneformer",
            "mouse_variant": "base",
        },
        "genes_to_perturb": list(_FLY_PERTURB),
        "forward_batch_size": 8,
        "finetune_batch_size": 4,
        "max_input_size": 2048,
    },
    {
        "id": "fly_data__human_v2",
        "input_dir": "/app/data/smoke_fly",
        "output_root": "/app/output/smoke_matrix/fly_data__human_v2",
        "species": {
            "model_organism": "drosophila",
            "model": "human_geneformer",
            "human_variant": "v2_104m",
        },
        "genes_to_perturb": list(_FLY_PERTURB),
        "forward_batch_size": 4,
        "finetune_batch_size": 2,
        "max_input_size": 4096,
    },
]


def _smoke_needs_conversion_report(species: dict) -> bool:
    pair = conversion_pair(species["model_organism"], species["model"])
    return pair is not None


def _write_case_config(case: dict, cfg_path: Path) -> None:
    stages_ft: dict = {
        "training": {
            "epochs": 1,
            "warmup_ratio": 0.05,
            "batch_size": case["finetune_batch_size"],
            "eval_batch_size": 1,
            "num_runs": 1,
        },
        "runtime": {"dataloader_num_workers": 0},
        "metadata": {"add_columns": {"organ_major": "brain"}},
        "umap": {"enabled": False},
    }
    if case["max_input_size"] != 2048:
        stages_ft["model"] = {"max_input_size": case["max_input_size"]}

    cfg = {
        "data": {
            "input_type": "single-cell",
            "input_dir": case["input_dir"],
            "output_prefix": "smoke",
        },
        "paths": {"output_root": case["output_root"]},
        "species": case["species"],
        "runtime": {
            "nproc": 2,
            "max_cells": 5000,
            "forward_batch_size": case["forward_batch_size"],
        },
        "perturbation": {
            "type": "delete",
            "state_key": "disease",
            "start_state": "AD",
            "end_state": "WT",
            "genes_to_perturb": case["genes_to_perturb"],
        },
        "stages": {
            "tokenize": (
                {"report_conversion": True}
                if _smoke_needs_conversion_report(case["species"])
                else {}
            ),
            "finetune": stages_ft,
            "isp": {
                "runtime": {"forward_batch_size": case["forward_batch_size"]},
                "umap": {"enabled": False},
            },
        },
    }
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)


def _run_docker(cfg_path: Path, case_id: str, log_path: Path) -> int:
    env = os.environ.copy()
    env["PIPELINE_CONFIG"] = f"/app/{cfg_path.relative_to(ROOT).as_posix()}"
    env["WANDB_DISABLED"] = "true"
    env["GENEFORMER_MODELS_ROOT"] = "/app/models"
    cmd = [
        "docker",
        "compose",
        "run",
        "--rm",
        "-e",
        f"PIPELINE_CONFIG={env['PIPELINE_CONFIG']}",
        "-e",
        "WANDB_DISABLED=true",
        "-e",
        "GENEFORMER_MODELS_ROOT=/app/models",
        "pipeline",
        "python3",
        "/app/core/run_pipeline.py",
        "--config",
        env["PIPELINE_CONFIG"],
    ]
    print(f"\n===== CASE {case_id} =====", flush=True)
    print(" ".join(cmd), flush=True)
    with log_path.open("w") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return proc.returncode


def _run_local(cfg_path: Path, case_id: str, log_path: Path) -> int:
    env = os.environ.copy()
    env["WANDB_DISABLED"] = "true"
    env.setdefault("GENEFORMER_MODELS_ROOT", str(ROOT / "models"))
    cmd = [sys.executable, str(ROOT / "core" / "run_pipeline.py"), "--config", str(cfg_path)]
    print(f"\n===== CASE {case_id} (local) =====", flush=True)
    with log_path.open("w") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return proc.returncode


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local", action="store_true", help="Run without docker compose")
    p.add_argument(
        "--only",
        action="append",
        default=[],
        help="Case id substring filter (repeatable)",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Keep going after a failed case (default)",
    )
    p.add_argument("--stop-on-error", action="store_true")
    args = p.parse_args()
    continue_on_error = not args.stop_on_error

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "output" / "smoke_matrix"
    cfg_dir = ROOT / "core" / "config" / "smoke_matrix"
    log_dir = out_dir / "logs" / stamp
    log_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for case in CASES:
        if args.only and not any(s in case["id"] for s in args.only):
            continue
        cfg_path = cfg_dir / f"{case['id']}.yaml"
        _write_case_config(case, cfg_path)
        log_path = log_dir / f"{case['id']}.log"
        t0 = time.time()
        try:
            rc = (
                _run_local(cfg_path, case["id"], log_path)
                if args.local
                else _run_docker(cfg_path, case["id"], log_path)
            )
        except Exception as exc:  # noqa: BLE001
            rc = 1
            log_path.write_text(f"Launcher exception: {exc}\n")
        elapsed = round(time.time() - t0, 1)
        status = "PASS" if rc == 0 else "FAIL"
        print(f">>> {case['id']}: {status} ({elapsed}s) log={log_path}", flush=True)
        results.append(
            {
                "id": case["id"],
                "status": status,
                "returncode": rc,
                "elapsed_s": elapsed,
                "config": str(cfg_path.relative_to(ROOT)),
                "log": str(log_path.relative_to(ROOT)),
            }
        )
        if rc != 0 and not continue_on_error:
            break

    summary_path = log_dir / "summary.json"
    summary_path.write_text(json.dumps({"stamp": stamp, "results": results}, indent=2))
    print("\n===== SUMMARY =====")
    for r in results:
        print(f"  {r['status']:4}  {r['id']}  ({r['elapsed_s']}s)")
    print(f"Wrote {summary_path}")
    if any(r["status"] != "PASS" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
