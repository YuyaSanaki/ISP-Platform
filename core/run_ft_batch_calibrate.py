"""Calibrate fine-tune train_batch_size on this GPU (no training).

Loads the pretrained Geneformer for the configured species/model, runs the same
throughput/memory probe as fine-tune `training.batch_size: auto`, then writes
`recommended_train_batch_size.json` next to the config and under output_root.

Usage:
  python3 /app/core/run_ft_batch_calibrate.py --config /path/to/ft_batch_calibrate.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import BertForSequenceClassification

sys.path.append(os.getcwd())
from geneformer.auto_batch_size import (  # noqa: E402
    BusyGPUError,
    is_auto,
    train_candidates_for_seq_len,
    train_memory_fraction_for_seq_len,
)
from geneformer.backends import resolve_pretrained_path  # noqa: E402
from geneformer.species_context import (  # noqa: E402
    backend_from_config,
    default_finetune_batch_size,
    log_species_banner,
    species_from_config,
)
from run_finetune import resolve_train_batch_size  # noqa: E402

RESULT_FILENAME = "recommended_train_batch_size.json"
RESULT_MARKER = "RECOMMENDED_TRAIN_BATCH_SIZE"


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return cfg


def resolve_model_dir(cfg: dict[str, Any]) -> Path:
    """Resolve checkpoint dir: paths.geneformer_model, else species → pretrained path."""
    paths = cfg.get("paths") or {}
    raw = paths.get("geneformer_model")
    if raw is not None and str(raw).strip() and str(raw).strip().lower() != "null":
        return Path(str(raw))
    # resolve_pretrained_path expects (species mapping, paths mapping), not a path string.
    return Path(resolve_pretrained_path(cfg.get("species"), paths))


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=os.environ.get(
            "FT_BATCH_CALIBRATE_CONFIG",
            "/app/core/config/ft_batch_calibrate.yaml",
        ),
        help="YAML config path (or FT_BATCH_CALIBRATE_CONFIG)",
    )
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help=(
            "Allow calibration while another job holds the GPU "
            "(not recommended — yields a too-small batch size)."
        ),
    )
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    # Always measure fresh for this dedicated run type (ignore stale cache).
    os.environ["GENEFORMER_BATCH_SIZE_CACHE"] = "off"

    print("=" * 72)
    print("Fine-tune train_batch_size calibration (no training)")
    print("=" * 72)
    print(f"  config: {config_path}")
    log_species_banner(species_from_config(cfg), prefix="  ")

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required for FT batch calibration. "
            "Run this job on a GPU host (e.g. the webui / pipeline container)."
        )

    model_dir = resolve_model_dir(cfg)
    backend = backend_from_config(cfg)
    tr_cfg = cfg.get("training") or {}
    fp16 = bool(tr_cfg.get("fp16", True))
    num_labels = max(2, int(tr_cfg.get("probe_num_labels", 2)))
    default_bs = default_finetune_batch_size(int(backend.max_input_size))

    print(f"  pretrained model: {model_dir}")
    print(f"  probe_num_labels: {num_labels}")
    print(f"  fp16:             {fp16}")
    print()

    model = BertForSequenceClassification.from_pretrained(
        str(model_dir),
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False,
        ignore_mismatched_sizes=True,
    ).to("cuda")

    max_len = int(model.config.max_position_embeddings)
    pad_id = int(model.config.pad_token_id or 0)

    print(f"  seq_len (max_position_embeddings): {max_len}")
    print(f"  train candidates: {list(train_candidates_for_seq_len(max_len))}")
    print(f"  train memory_fraction: {train_memory_fraction_for_seq_len(max_len)}")
    if max_len > 2048:
        print(
            "  NOTE: long-sequence model (e.g. human Geneformer V2 / Mouse→Human) — "
            "calibration uses a stricter memory budget and starts candidates at 2 "
            "so the recommendation matches real Trainer memory."
        )
    print()

    try:
        chosen = resolve_train_batch_size(
            "auto",
            model,
            max_len=max_len,
            pad_id=pad_id,
            fp16=fp16,
            model_dir=str(model_dir),
            num_labels=num_labels,
            default=default_bs,
            require_idle=not args.allow_busy_gpu,
            skip_cache=True,
        )
    except BusyGPUError as e:
        print()
        print("=" * 72)
        print(f"ERROR: {e}")
        print("=" * 72)
        return 2

    if is_auto(chosen):
        raise RuntimeError("Calibration returned 'auto'; expected an int.")

    chosen_i = int(chosen)
    payload: dict[str, Any] = {
        "recommended_train_batch_size": chosen_i,
        "gpu": torch.cuda.get_device_name(),
        "model_dir": str(model_dir),
        "seq_len": max_len,
        "num_labels": num_labels,
        "fp16": fp16,
        "config": str(config_path),
        "calibrated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Copy this integer into Pipeline → Fine-tune train_batch_size "
            "(runtime.train_batch_size). Keep it fixed for comparable FT results. "
            "Long-seq models (max_position_embeddings>2048) use a stricter "
            "calibration budget automatically."
        ),
    }

    next_to_config = config_path.parent / RESULT_FILENAME
    write_result(next_to_config, payload)

    out_root = Path(str((cfg.get("paths") or {}).get("output_root") or "/app/output"))
    date_dir = out_root / datetime.now(timezone.utc).strftime("%Y%m%d")
    run_dir = date_dir / (
        "ft_batch_calibrate_" + datetime.now(timezone.utc).strftime("%H%M%S_%f")[:-3] + "Z"
    )
    write_result(run_dir / RESULT_FILENAME, payload)
    with open(run_dir / "ft_batch_calibrate_config_used.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print()
    print("=" * 72)
    print(f"  Recommended fine-tune train_batch_size: {chosen_i}")
    print(f"  Wrote: {next_to_config}")
    print(f"  Wrote: {run_dir / RESULT_FILENAME}")
    print("=" * 72)
    print(f"{RESULT_MARKER}={chosen_i}")
    print(
        "Paste this value into Pipeline (E2E) → Fine-tune train_batch_size, "
        "then keep it unchanged for subsequent comparable runs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
