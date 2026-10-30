"""Run one ISP inference case and record timing, memory, and the exit status.

The process is ``core/run_isp.py`` with an integer forward batch size.
``forward_batch_size`` is the number of deleted-gene sequences in one forward,
not the number of original cells.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import yaml


def _nvidia_used_mib() -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.strip().splitlines()[0] if out.strip() else ""
    if not line or "n/a" in line.lower():
        return None
    try:
        return int(float(line.split()[0]))
    except ValueError:
        return None


def _sample_memory(stop: threading.Event, samples: list[dict], interval: float) -> None:
    while not stop.wait(interval):
        row = {"t": time.time()}
        try:
            mem = {}
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:")):
                        mem[line.split()[0][:-1]] = int(line.split()[1]) * 1024
            row["host"] = mem
        except OSError:
            pass
        used = _nvidia_used_mib()
        if used is not None:
            row["gpu_used_mib"] = used
        samples.append(row)


def _mark(text: str, marks: dict[str, float], clock: float) -> None:
    keys = (
        ("Computing state centroids", "centroid_start"),
        ("ISP phase 1/2", "phase1_start"),
        ("ISP phase 2/2", "phase2_start"),
        ("ISP: perturb cells", "phase2_start"),
    )
    for needle, key in keys:
        if needle in text and key not in marks:
            marks[key] = clock


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "isp.yaml"
    cfg = {
        "paths": {
            "dataset": str(args.dataset),
            "geneformer_model": str(args.model),
            "output_root": str(out / "isp_out"),
            "output_time_subdir": False,
        },
        "species": {
            "model_organism": "human",
            "model": "human_geneformer",
            "human_variant": "v2_104m",
            "ortholog_policy": "one2one",
        },
        "perturbation": {
            "type": "delete",
            "organ_data": "bench",
            "state_key": "disease",
            "start_state": "Matrigel",
            "end_state": "No_Matrigel",
            "alt_states": [],
            "genes_to_perturb": [],
        },
        "model": {"type": "CellClassifier", "num_classes": 2, "max_input_size": 4096},
        "isp": {
            "max_ncells": 100000,
            "emb_layer": 0,
            "emb_mode": "cell",
            "cell_emb_style": "mean_pool",
            "combos": 0,
            "anchor_gene": None,
            "filter_data": None,
            "legacy_padding_mean": False,
            "max_ncells_per_state": None,
        },
        "stats": {"mode": "goal_state_shift"},
        "analysis": {"enabled": False},
        "umap": {"enabled": False},
        "postprocess": {"enabled": False},
        "provenance": {"enable_input_fingerprint": False},
        "runtime": {"forward_batch_size": int(args.batch_size), "nproc": 4},
    }
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["WANDB_DISABLED"] = "true"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    repo = str(args.repo)
    env["PYTHONPATH"] = os.pathsep.join(
        [f"{repo}/core", f"{repo}/contracts", env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    command = [
        args.python,
        f"{repo}/core/run_isp.py",
        "--config",
        str(cfg_path),
        "--forward-batch-size",
        str(args.batch_size),
        "--nproc",
        "4",
        "--skip-analysis",
        "--skip-isp-umap",
        "--no-output-date-subdir",
        "--no-output-time-subdir",
        "--skip-postprocess",
    ]
    log_path = out / "run.log"
    samples: list[dict] = []
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_memory, args=(stop, samples, 2.0), daemon=True)
    sampler.start()
    marks: dict[str, float] = {}
    started = time.time()
    clock0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=repo,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            now = time.perf_counter()
            log.write(f"{now - clock0:.3f} {line}")
            log.flush()
            _mark(line, marks, now - clock0)
        code = proc.wait()
    stop.set()
    elapsed = time.perf_counter() - clock0
    text = log_path.read_text(encoding="utf-8", errors="replace")
    oom = code in (137, -9, 9) or "out of memory" in text.lower() or "CUDA out of memory" in text
    phase2 = None
    if "phase2_start" in marks:
        phase2 = elapsed - marks["phase2_start"]
    avail = [row["host"]["MemAvailable"] for row in samples if row.get("host")]
    gpu_used = [row["gpu_used_mib"] for row in samples if "gpu_used_mib" in row]
    result = {
        "machine": args.machine,
        "subset": args.subset,
        "forward_batch_size": args.batch_size,
        "pairs": args.pairs,
        "exit_code": code,
        "oom": bool(oom),
        "wall_seconds": round(elapsed, 3),
        "phase2_seconds": None if phase2 is None else round(phase2, 3),
        "pairs_per_second_wall": None if elapsed <= 0 else round(args.pairs / elapsed, 4),
        "pairs_per_second_phase2": None
        if not phase2
        else round(args.pairs / phase2, 4),
        "marks": {key: round(value, 3) for key, value in marks.items()},
        "host_mem_available_min_bytes": min(avail) if avail else None,
        "host_mem_available_start_bytes": avail[0] if avail else None,
        "gpu_memory_used_max_mib": max(gpu_used) if gpu_used else None,
        "started_unix": started,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if code == 0 else code


if __name__ == "__main__":
    raise SystemExit(main())
