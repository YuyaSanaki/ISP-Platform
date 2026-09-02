#!/usr/bin/env python3
"""End-to-end: serve a deposited zip URL → import → tokenize/FT/ISP smoke run.

Usage (from repo root):
  python3 scripts/test_url_data_pipeline.py
"""
from __future__ import annotations

import http.server
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "core", ROOT / "contracts", ROOT / "webui"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

FIXTURE = ROOT / "data" / "streamlit_workspace" / "_fixtures" / "smoke_mouse_url_test.zip"
OUT_ROOT = ROOT / "output" / "url_data_smoke"


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A003
        return


def _serve(directory: Path) -> tuple[str, http.server.HTTPServer]:
    directory = directory.resolve()

    class Handler(_QuietHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", server


def main() -> int:
    if not FIXTURE.is_file():
        print(f"FATAL: missing fixture {FIXTURE}", file=sys.stderr)
        return 2

    from data_input_layout import discover_sample_dirs, unique_states_from_samples
    from streamlit_remote_data import import_study_from_url

    bugs: list[dict] = []
    results: dict = {"download": None, "run": None, "bugs": bugs}

    base, server = _serve(FIXTURE.parent)
    url = f"{base}/{FIXTURE.name}"
    upload_dir = ROOT / "data" / "streamlit_workspace" / "_url_e2e"
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== DOWNLOAD TEST ===\nURL: {url}")
    t0 = time.time()
    try:
        tokenize_dir, summary, filename = import_study_from_url(
            url, upload_dir, "url_e2e_smoke"
        )
        samples = discover_sample_dirs(tokenize_dir)
        states = unique_states_from_samples(tokenize_dir)
        elapsed = round(time.time() - t0, 2)
        results["download"] = {
            "status": "PASS",
            "elapsed_s": elapsed,
            "filename": filename,
            "tokenize_dir": str(tokenize_dir),
            "n_samples": len(samples),
            "states": states,
        }
        print(f"PASS download ({elapsed}s) samples={len(samples)} states={states}")
        print(summary)
    except Exception as e:  # noqa: BLE001
        results["download"] = {"status": "FAIL", "error": f"{type(e).__name__}: {e}"}
        bugs.append(
            {
                "id": "BUG-URL-1",
                "area": "download/import",
                "detail": str(e),
            }
        )
        print(f"FAIL download: {e}")
        server.shutdown()
        _write_results(results)
        return 1
    finally:
        # Keep server up through run test (config embeds host URL only for provenance).
        pass

    print("\n=== RUN TEST (tokenize → finetune → ISP) ===")
    start_state, end_state = (states + ["AD", "WT"])[:2]
    if len(states) >= 2:
        start_state, end_state = states[0], states[1]
    cfg = {
        "data": {
            "input_type": "single-cell",
            "input_dir": str(tokenize_dir),
            "output_prefix": "url_e2e_smoke",
        },
        "paths": {"output_root": str(OUT_ROOT)},
        "species": {
            "model_organism": "mouse",
            "model": "mouse_geneformer",
            "mouse_variant": "base",
        },
        "runtime": {"nproc": 2, "max_cells": 5000, "forward_batch_size": 8},
        "perturbation": {
            "type": "delete",
            "state_key": "disease",
            "start_state": start_state,
            "end_state": end_state,
            "genes_to_perturb": ["Igfbp2"],
        },
        "stages": {
            "tokenize": {"report_conversion": False},
            "finetune": {
                "training": {
                    "epochs": 1,
                    "warmup_ratio": 0.05,
                    "batch_size": 4,
                    "eval_batch_size": 1,
                    "num_runs": 1,
                },
                "runtime": {"dataloader_num_workers": 0},
                "metadata": {"add_columns": {"organ_major": "brain"}},
                "umap": {"enabled": False},
            },
            "isp": {
                "runtime": {"forward_batch_size": 8},
                "umap": {"enabled": False},
            },
        },
    }
    cfg_dir = ROOT / "core" / "config" / "smoke_matrix"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "url_e2e_smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # Paths inside the container are /app/...
    container_cfg = "/app/core/config/smoke_matrix/url_e2e_smoke.yaml"
    # Rewrite host paths to /app for docker bind mount
    cfg_docker = json.loads(json.dumps(cfg))
    cfg_docker["data"]["input_dir"] = str(tokenize_dir).replace(str(ROOT), "/app")
    cfg_docker["paths"]["output_root"] = str(OUT_ROOT).replace(str(ROOT), "/app")
    cfg_path.write_text(yaml.safe_dump(cfg_docker, sort_keys=False), encoding="utf-8")

    t1 = time.time()
    cmd = [
        "docker",
        "compose",
        "run",
        "--rm",
        "-e",
        "WANDB_DISABLED=true",
        "-e",
        "GENEFORMER_MODELS_ROOT=/app/models",
        "pipeline",
        "python3",
        "/app/core/run_pipeline.py",
        "--config",
        container_cfg,
    ]
    log_path = OUT_ROOT / "url_e2e_run.log"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(" ".join(cmd))
    with log_path.open("w") as log:
        proc = subprocess.run(
            cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, text=True
        )
    elapsed = round(time.time() - t1, 2)
    if proc.returncode == 0:
        results["run"] = {"status": "PASS", "elapsed_s": elapsed, "log": str(log_path)}
        print(f"PASS run ({elapsed}s) log={log_path}")
    else:
        results["run"] = {
            "status": "FAIL",
            "elapsed_s": elapsed,
            "returncode": proc.returncode,
            "log": str(log_path),
        }
        # Capture tail for handoff
        tail = log_path.read_text(errors="replace")[-2000:]
        bugs.append(
            {
                "id": "BUG-URL-2",
                "area": "pipeline run after URL import",
                "detail": f"exit={proc.returncode}",
                "log_tail": tail,
            }
        )
        print(f"FAIL run exit={proc.returncode} ({elapsed}s)")
        print(tail)

    server.shutdown()
    _write_results(results)
    return 0 if results["run"]["status"] == "PASS" else 1


def _write_results(results: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / "url_e2e_summary.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
