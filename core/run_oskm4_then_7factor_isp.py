#!/usr/bin/env python3
"""Sequential group OE: Yamanaka OSKM4, then 7-factor cocktail on the same cells.

Step 1 applies OSKM4 (OCT4+SOX2+KLF4+MYC). Step 2 applies the 7-factor cocktail
(NANOG+OCT4+SOX2+ESRRB+LIN28A+DPPA4+TERT) onto those perturbed encodings.
Scores goal_state_shift after each step. No OSN arm.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))

from geneformer import in_silico_perturber as isp
from geneformer.auto_batch_size import coerce_batch_size
from geneformer.species_context import (
    backend_from_config,
    default_isp_forward_batch_size,
    log_species_banner,
    species_from_config,
)
from sequential_oe import (
    COCKTAIL_FACTOR_KEYS,
    COCKTAIL_FACTORS,
    OSKM_FACTOR_KEYS,
    OSKM_FACTORS,
)

import run_sequential_isp as seq


STAGES = (
    ("oskm4", "Yamanaka OSKM4", OSKM_FACTOR_KEYS, OSKM_FACTORS),
    ("7factor", "OSKM4 → 7-factor", COCKTAIL_FACTOR_KEYS, COCKTAIL_FACTORS),
)


def _tokens_for(keys: Sequence[str], table: dict[str, dict[str, str]], species_key: str, token_dict: dict) -> list[int]:
    out: list[int] = []
    for key in keys:
        ens = table[key][species_key]
        if ens not in token_dict:
            raise KeyError(f"Factor {key} Ensembl {ens} not in token dictionary")
        out.append(int(token_dict[ens]))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OSKM4 then 7-factor sequential OE")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-ncells", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--no-save-datasets", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--forward-batch-size", default=None, metavar="N|auto")
    parser.add_argument("--nproc", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.resolve().read_text())
    paths = cfg.get("paths") or {}
    pert = cfg.get("perturbation") or {}
    isp_cfg = cfg.get("isp") or {}
    mdl = cfg.get("model") or {}
    seq_cfg = cfg.get("sequential") or {}
    runtime = cfg.get("runtime") or {}

    dataset_path = seq._app_path(str(paths["dataset"]))
    model_path = seq._app_path(str(paths["geneformer_model"]))
    output_root = args.output_root or seq._app_path(str(paths["output_root"]))
    if not output_root.is_absolute():
        output_root = ROOT / output_root

    state_key = str(pert["state_key"])
    start_state = str(pert["start_state"])
    goal_state = str(pert["end_state"])
    max_ncells = args.max_ncells if args.max_ncells is not None else isp_cfg.get("max_ncells")
    emb_layer = int(isp_cfg.get("emb_layer", 0))
    nproc = args.nproc if args.nproc is not None else int(runtime.get("nproc", 4))
    save_datasets = not args.no_save_datasets and bool(seq_cfg.get("save_intermediate_datasets", True))
    resume = not args.no_resume

    species_key = (
        "human"
        if "human" in str((cfg.get("species") or {}).get("model_organism", "human")).lower()
        else "mouse"
    )
    log_species_banner(species_from_config(cfg))

    from datasets import load_from_disk

    print(f"Loading dataset: {dataset_path}", flush=True)
    dataset = load_from_disk(str(dataset_path))
    start_ds = seq._select_start_cells(dataset, state_key, start_state, max_ncells)
    print(f"Start cells ({start_state}): n={len(start_ds)}", flush=True)

    model = isp.load_model(mdl.get("type", "CellClassifier"), int(mdl.get("num_classes", 2)), str(model_path))
    layer_to_quant = isp.quant_layers(model) + emb_layer
    model_input_size = isp.get_model_input_size(model)

    import pickle

    backend = backend_from_config(cfg)
    with open(backend.token_dictionary, "rb") as fh:
        token_dict = pickle.load(fh)
    pad_token_id = token_dict.get("<pad>")

    fbs_raw = args.forward_batch_size or runtime.get("forward_batch_size", isp_cfg.get("forward_batch_size", "auto"))
    forward_batch_size = seq.resolve_sequential_forward_batch_size(
        coerce_batch_size(fbs_raw, default=default_isp_forward_batch_size(backend.max_input_size)),
        model,
        start_ds,
        pad_token_id,
        str(model_path),
    )
    print(f"forward_batch_size={forward_batch_size}", flush=True)

    cell_states = {
        "state_key": state_key,
        "start_state": start_state,
        "goal_state": goal_state,
        "alt_states": list(pert.get("alt_states") or []),
    }
    centroid_states = [start_state, goal_state, *list(pert.get("alt_states") or [])]
    centroid_ds = seq._centroid_dataset(dataset, state_key, centroid_states, max_ncells, nproc)
    state_embs = isp.get_cell_state_avg_embs(
        model,
        centroid_ds,
        cell_states,
        layer_to_quant,
        pad_token_id,
        forward_batch_size,
        nproc,
    )

    stages: list[tuple[str, str, list[int]]] = []
    for tag, label, keys, table in STAGES:
        stages.append((tag, label, _tokens_for(keys, table, species_key, token_dict)))

    output_root.mkdir(parents=True, exist_ok=True)
    seq._save_dataset(start_ds, output_root / "start.dataset")
    run_meta = {
        "config": str(args.config.resolve()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "start_state": start_state,
        "goal_state": goal_state,
        "n_cells": len(start_ds),
        "stages": [s[0] for s in stages],
        "mode": "oskm4_then_7factor",
        "rank_convention": "later_step_leftmost",
    }
    (output_root / "run_manifest.json").write_text(json.dumps(run_meta, indent=2) + "\n")

    working = start_ds
    try:
        working.reset_format()
    except Exception:
        pass

    step_rows: list[dict[str, Any]] = []
    for step_idx, (tag, label, tokens) in enumerate(stages, start=1):
        step_dir = output_root / "steps" / f"step{step_idx:02d}_{tag}"
        csv_path = step_dir / "single_gene_per_cell_shifts.csv"
        ds_path = step_dir / "perturbed.dataset"
        if resume and csv_path.is_file() and (not save_datasets or ds_path.exists()):
            print(f"Resume: skip {label} ({tag})", flush=True)
            s = pd.to_numeric(pd.read_csv(csv_path)["Shift_to_goal_end"], errors="coerce").dropna()
            stats = seq._summarize_series(s)
            if save_datasets and ds_path.exists():
                from datasets import load_from_disk as _load

                working = _load(str(ds_path))
        else:
            print(f"Step {step_idx}: {label} ({len(tokens)} genes)...", flush=True)
            working = working.map(
                seq._apply_step_example,
                fn_kwargs={"tokens": list(tokens)},
                num_proc=seq._gpu_resident_map_workers(nproc),
            )
            df = seq.compute_goal_state_shifts(
                model,
                start_ds,
                working,
                list(tokens),
                goal_state,
                cell_states,
                state_embs,
                layer_to_quant,
                pad_token_id,
                model_input_size,
                forward_batch_size,
                nproc,
            )
            seq._write_shifts(df, csv_path)
            if save_datasets:
                seq._save_dataset(working, ds_path)
            stats = seq._summarize_series(df["Shift_to_goal_end"])
            print(f"  median shift={stats['median']:.6f} n={stats['n']} frac>0={stats['frac_positive']:.3f}", flush=True)
            seq._empty_cuda_cache()

        step_rows.append(
            {
                "step": step_idx,
                "tag": tag,
                "label": label,
                "n": stats["n"],
                "median": stats["median"],
                "mean": stats["mean"],
                "p25": stats["p25"],
                "p75": stats["p75"],
                "frac_positive": stats["frac_positive"],
            }
        )

    pd.DataFrame(step_rows).to_csv(output_root / "step_summary.csv", index=False)
    pd.DataFrame(step_rows).to_csv(output_root / "order_summary.csv", index=False)
    print(f"Wrote {output_root / 'step_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
