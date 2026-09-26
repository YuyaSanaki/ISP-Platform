#!/usr/bin/env python3
"""State-feedback Phase 0 — Oracle fixed-gene-set feedback (go/no-go).

Compares, on the same start cells / FT model / goal centroids:

1. **ordered_rank_edit** — current paper sequential OE chain (insert-0).
2. **oracle_mid** — after step-1 OE, permute to observed endpoint ranks
   (fixed gene set), then continue OE for remaining factors.
3. **oracle_endpoint** — permute start cells to observed endpoint ranks
   (no OE); ceiling if the goal state ranks alone explain the shift.

Hard gate: if oracle_mid (and/or oracle_endpoint) does not improve median
``goal_state_shift`` over ordered_rank_edit in a meaningful way, stop —
do not implement Phase 1–2 decoders.

Usage:
  python3 core/run_state_feedback_oracle.py --config core/config/state_feedback_oracle.yaml
  python3 core/run_state_feedback_oracle.py --config ... --max-ncells 50
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
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
    OSKM_FACTOR_KEYS,
    OSKM_FACTORS,
    apply_single_step_overexpress,
    order_label,
    tokens_for_order,
)
from state_feedback.oracle_rerank import (
    build_pseudobulk_rank_priority,
    rerank_fixed_gene_set,
    spearman_rank_correlation,
)

# Reuse sequential ISP helpers without importing __main__ side effects.
import run_sequential_isp as seq


def _resolve_factor_tokens(species_key: str, cfg: Mapping[str, Any]) -> dict[str, int]:
    backend = backend_from_config(cfg)
    with open(backend.token_dictionary, "rb") as fh:
        token_dict = pickle.load(fh)
    out: dict[str, int] = {}
    for key in OSKM_FACTOR_KEYS:
        ens = OSKM_FACTORS[key][species_key]
        if ens not in token_dict:
            raise KeyError(f"OSKM factor {key} ({ens}) missing from token dictionary")
        out[key] = int(token_dict[ens])
    return out


def _apply_oe(example: dict[str, Any], tokens: Sequence[int]) -> dict[str, Any]:
    return apply_single_step_overexpress(example, tokens)


def _apply_oracle(example: dict[str, Any], priority: Mapping[int, float]) -> dict[str, Any]:
    ex = dict(example)
    ids = list(ex["input_ids"])
    new_ids = rerank_fixed_gene_set(ids, priority)
    ex["input_ids"] = new_ids
    ex["length"] = len(new_ids)
    ex["attention_mask"] = [1] * len(new_ids)
    return ex


def _summarize(df: pd.DataFrame) -> dict[str, float]:
    s = pd.to_numeric(df["Shift_to_goal_end"], errors="coerce").dropna()
    if s.empty:
        return {"n": 0.0, "median": float("nan"), "mean": float("nan"), "frac_positive": float("nan")}
    return {
        "n": float(len(s)),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "frac_positive": float((s > 0).mean()),
    }


def _score(
    model,
    start_ds,
    working,
    oe_tokens: Sequence[int],
    goal_state: str,
    cell_states: dict[str, Any],
    state_embs: dict[str, torch.Tensor],
    *,
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    nproc: int,
    batch_state: list[int],
) -> pd.DataFrame:
    return seq.compute_goal_state_shifts(
        model,
        start_ds,
        working,
        list(oe_tokens),
        goal_state,
        cell_states,
        state_embs,
        layer_to_quant,
        pad_token_id,
        model_input_size,
        batch_state[0],
        nproc,
        batch_state=batch_state,
    )


def run_ordered_rank_edit(
    model,
    start_ds,
    order: tuple[str, ...],
    token_by_factor: Mapping[str, int],
    goal_state: str,
    cell_states: dict[str, Any],
    state_embs: dict[str, torch.Tensor],
    out_dir: Path,
    *,
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    nproc: int,
) -> list[dict[str, Any]]:
    """Baseline: cumulative length-preserving OE (Ordered rank-edit ISP)."""
    map_workers = seq._gpu_resident_map_workers(nproc)
    batch_state = [int(forward_batch_size)]
    working = start_ds
    try:
        working.reset_format()
    except Exception:
        pass
    step_tokens = tokens_for_order(order, token_by_factor)
    cumulative: list[int] = []
    rows: list[dict[str, Any]] = []
    for step_idx, (fk, toks) in enumerate(zip(order, step_tokens), start=1):
        cumulative.extend(toks)
        working = working.map(
            lambda ex, t=list(toks): _apply_oe(ex, t),
            num_proc=map_workers,
        )
        df = _score(
            model,
            start_ds,
            working,
            cumulative,
            goal_state,
            cell_states,
            state_embs,
            layer_to_quant=layer_to_quant,
            pad_token_id=pad_token_id,
            model_input_size=model_input_size,
            forward_batch_size=forward_batch_size,
            nproc=nproc,
            batch_state=batch_state,
        )
        step_dir = out_dir / f"step{step_idx:02d}_{fk}"
        step_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(step_dir / "single_gene_per_cell_shifts.csv", index=False)
        stats = _summarize(df)
        rows.append(
            {
                "condition": "ordered_rank_edit",
                "order": order_label(order),
                "step": step_idx,
                "factor": fk,
                "event": f"oe_{fk}",
                **stats,
            }
        )
        print(
            f"  [ordered_rank_edit] step{step_idx} {fk}: "
            f"median={stats['median']:.6f} n={int(stats['n'])}",
            flush=True,
        )
        seq._empty_cuda_cache()
    return rows


def run_oracle_mid(
    model,
    start_ds,
    order: tuple[str, ...],
    token_by_factor: Mapping[str, int],
    oracle_priority: Mapping[int, float],
    goal_state: str,
    cell_states: dict[str, Any],
    state_embs: dict[str, torch.Tensor],
    out_dir: Path,
    *,
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    nproc: int,
    oracle_after_step: int = 1,
) -> list[dict[str, Any]]:
    """OE chain with an oracle fixed-gene-set permute after ``oracle_after_step``."""
    map_workers = seq._gpu_resident_map_workers(nproc)
    batch_state = [int(forward_batch_size)]
    working = start_ds
    try:
        working.reset_format()
    except Exception:
        pass
    step_tokens = tokens_for_order(order, token_by_factor)
    cumulative: list[int] = []
    rows: list[dict[str, Any]] = []
    for step_idx, (fk, toks) in enumerate(zip(order, step_tokens), start=1):
        cumulative.extend(toks)
        working = working.map(
            lambda ex, t=list(toks): _apply_oe(ex, t),
            num_proc=map_workers,
        )
        df = _score(
            model,
            start_ds,
            working,
            cumulative,
            goal_state,
            cell_states,
            state_embs,
            layer_to_quant=layer_to_quant,
            pad_token_id=pad_token_id,
            model_input_size=model_input_size,
            forward_batch_size=forward_batch_size,
            nproc=nproc,
            batch_state=batch_state,
        )
        step_dir = out_dir / f"step{step_idx:02d}_{fk}"
        step_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(step_dir / "single_gene_per_cell_shifts.csv", index=False)
        stats = _summarize(df)
        rows.append(
            {
                "condition": "oracle_mid",
                "order": order_label(order),
                "step": step_idx,
                "factor": fk,
                "event": f"oe_{fk}",
                **stats,
            }
        )
        print(
            f"  [oracle_mid] step{step_idx} oe {fk}: "
            f"median={stats['median']:.6f} n={int(stats['n'])}",
            flush=True,
        )
        seq._empty_cuda_cache()

        if step_idx == int(oracle_after_step) and step_idx < len(order):
            # Spearman vs pre-oracle on a few cells for diagnostics
            sample_n = min(32, len(working))
            rho_vals = []
            for i in range(sample_n):
                before = list(working[i]["input_ids"])
                after = rerank_fixed_gene_set(before, oracle_priority)
                rho_vals.append(spearman_rank_correlation(before, after))
            mean_rho = float(sum(rho_vals) / len(rho_vals)) if rho_vals else float("nan")

            working = working.map(
                lambda ex, p=dict(oracle_priority): _apply_oracle(ex, p),
                num_proc=map_workers,
            )
            df_o = _score(
                model,
                start_ds,
                working,
                cumulative,
                goal_state,
                cell_states,
                state_embs,
                layer_to_quant=layer_to_quant,
                pad_token_id=pad_token_id,
                model_input_size=model_input_size,
                forward_batch_size=forward_batch_size,
                nproc=nproc,
                batch_state=batch_state,
            )
            o_dir = out_dir / f"step{step_idx:02d}_oracle_permute"
            o_dir.mkdir(parents=True, exist_ok=True)
            df_o.to_csv(o_dir / "single_gene_per_cell_shifts.csv", index=False)
            stats_o = _summarize(df_o)
            rows.append(
                {
                    "condition": "oracle_mid",
                    "order": order_label(order),
                    "step": step_idx,
                    "factor": "ORACLE",
                    "event": "oracle_permute",
                    "spearman_vs_pre_oracle_mean": mean_rho,
                    **stats_o,
                }
            )
            print(
                f"  [oracle_mid] oracle permute after step{step_idx}: "
                f"median={stats_o['median']:.6f} "
                f"spearman_vs_pre≈{mean_rho:.4f}",
                flush=True,
            )
            seq._empty_cuda_cache()
    return rows


def run_oracle_endpoint(
    model,
    start_ds,
    oracle_priority: Mapping[int, float],
    goal_state: str,
    cell_states: dict[str, Any],
    state_embs: dict[str, torch.Tensor],
    out_dir: Path,
    *,
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    nproc: int,
) -> list[dict[str, Any]]:
    """Ceiling: permute start → observed endpoint ranks; score with empty OE list via cell means."""
    map_workers = seq._gpu_resident_map_workers(nproc)
    batch_state = [int(forward_batch_size)]
    working = start_ds.map(
        lambda ex, p=dict(oracle_priority): _apply_oracle(ex, p),
        num_proc=map_workers,
    )
    # No OE tokens — use cell-mean goal shift path
    df = seq.compute_cell_mean_goal_state_shifts(
        model,
        start_ds,
        working,
        goal_state,
        state_embs,
        layer_to_quant,
        pad_token_id,
        model_input_size,
        batch_state[0],
        nproc,
        strip_leading=0,
        batch_state=batch_state,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "single_gene_per_cell_shifts.csv", index=False)
    stats = _summarize(df)
    print(
        f"  [oracle_endpoint] permute only: median={stats['median']:.6f} n={int(stats['n'])}",
        flush=True,
    )
    return [
        {
            "condition": "oracle_endpoint",
            "order": "",
            "step": 0,
            "factor": "ORACLE",
            "event": "endpoint_permute",
            **stats,
        }
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-ncells", type=int, default=None)
    parser.add_argument("--forward-batch-size", default=None)
    parser.add_argument("--nproc", type=int, default=None)
    parser.add_argument(
        "--orders",
        nargs="+",
        default=None,
        help="Factor orders e.g. K-M-S-O (default: config oracle.orders)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    paths = cfg.get("paths") or {}
    pert = cfg.get("perturbation") or {}
    mdl = cfg.get("model") or {}
    isp_cfg = cfg.get("isp") or {}
    runtime = cfg.get("runtime") or {}
    oracle_cfg = cfg.get("oracle") or {}

    dataset_path = Path(paths["dataset"]).expanduser()
    model_path = Path(paths["geneformer_model"]).expanduser()
    output_root = Path(paths["output_root"]).expanduser()
    if paths.get("output_date_subdir", True):
        output_root = output_root / datetime.now(timezone.utc).strftime("%Y%m%d")
    if paths.get("output_time_subdir", True):
        output_root = output_root / f"state_feedback_oracle_{datetime.now(timezone.utc).strftime('%H%M%S')}"

    state_key = str(pert["state_key"])
    start_state = str(pert["start_state"])
    goal_state = str(pert["end_state"])
    observed_state = str(oracle_cfg.get("observed_state") or goal_state)
    oracle_after_step = int(oracle_cfg.get("oracle_after_step", 1))

    max_ncells = args.max_ncells if args.max_ncells is not None else isp_cfg.get("max_ncells")
    emb_layer = int(isp_cfg.get("emb_layer", 0))
    nproc = args.nproc if args.nproc is not None else int(runtime.get("nproc", 1))

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

    # Observed ranks from endpoint / intermediate state (pseudobulk positions)
    obs_cap = int(oracle_cfg.get("observed_max_ncells") or max_ncells or 3000)
    obs_ds = seq._select_start_cells(dataset, state_key, observed_state, obs_cap)
    print(f"Observed-rank cells ({observed_state}): n={len(obs_ds)}", flush=True)
    oracle_priority = build_pseudobulk_rank_priority(obs_ds["input_ids"])
    print(f"Oracle priority tokens: {len(oracle_priority)}", flush=True)

    model = isp.load_model(mdl.get("type", "CellClassifier"), int(mdl.get("num_classes", 2)), str(model_path))
    layer_to_quant = isp.quant_layers(model) + emb_layer
    model_input_size = isp.get_model_input_size(model)

    backend = backend_from_config(cfg)
    with open(backend.token_dictionary, "rb") as fh:
        token_dict = pickle.load(fh)
    pad_token_id = token_dict.get("<pad>")

    fbs_raw = args.forward_batch_size or runtime.get("forward_batch_size", "auto")
    species_default = default_isp_forward_batch_size(backend.max_input_size)
    forward_batch_size = seq.resolve_sequential_forward_batch_size(
        coerce_batch_size(fbs_raw, default=species_default),
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
        seq._gpu_resident_map_workers(nproc),
    )
    seq._empty_cuda_cache()

    token_by_factor = _resolve_factor_tokens(species_key, cfg)
    raw_orders = args.orders or oracle_cfg.get("orders") or ["K-M-S-O"]
    orders: list[tuple[str, ...]] = []
    for label in raw_orders:
        parts = tuple(p.strip() for p in str(label).replace("_", "-").split("-") if p.strip())
        for p in parts:
            if p not in OSKM_FACTOR_KEYS:
                raise ValueError(f"Unknown factor {p!r} in order {label!r}")
        orders.append(parts)

    output_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []

    print("=== oracle_endpoint (permute start → observed ranks) ===", flush=True)
    all_rows.extend(
        run_oracle_endpoint(
            model,
            start_ds,
            oracle_priority,
            goal_state,
            cell_states,
            state_embs,
            output_root / "oracle_endpoint",
            layer_to_quant=layer_to_quant,
            pad_token_id=pad_token_id,
            model_input_size=model_input_size,
            forward_batch_size=forward_batch_size,
            nproc=nproc,
        )
    )

    for order in orders:
        tag = order_label(order)
        print(f"=== ordered_rank_edit {tag} ===", flush=True)
        all_rows.extend(
            run_ordered_rank_edit(
                model,
                start_ds,
                order,
                token_by_factor,
                goal_state,
                cell_states,
                state_embs,
                output_root / "ordered_rank_edit" / tag,
                layer_to_quant=layer_to_quant,
                pad_token_id=pad_token_id,
                model_input_size=model_input_size,
                forward_batch_size=forward_batch_size,
                nproc=nproc,
            )
        )
        print(f"=== oracle_mid {tag} (oracle after step {oracle_after_step}) ===", flush=True)
        all_rows.extend(
            run_oracle_mid(
                model,
                start_ds,
                order,
                token_by_factor,
                oracle_priority,
                goal_state,
                cell_states,
                state_embs,
                output_root / "oracle_mid" / tag,
                layer_to_quant=layer_to_quant,
                pad_token_id=pad_token_id,
                model_input_size=model_input_size,
                forward_batch_size=forward_batch_size,
                nproc=nproc,
                oracle_after_step=oracle_after_step,
            )
        )

    summary = pd.DataFrame(all_rows)
    summary_path = output_root / "phase0_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Go/no-go: final-step median oracle_mid vs ordered_rank_edit
    gate_rows = []
    for order in orders:
        tag = order_label(order)
        base = summary[
            (summary["condition"] == "ordered_rank_edit")
            & (summary["order"] == tag)
            & (summary["event"].astype(str).str.startswith("oe_"))
        ]
        ora = summary[
            (summary["condition"] == "oracle_mid")
            & (summary["order"] == tag)
            & (summary["event"].astype(str).str.startswith("oe_"))
        ]
        if base.empty or ora.empty:
            continue
        base_final = float(base.sort_values("step").iloc[-1]["median"])
        ora_final = float(ora.sort_values("step").iloc[-1]["median"])
        delta = ora_final - base_final
        gate_rows.append(
            {
                "order": tag,
                "ordered_rank_edit_final_median": base_final,
                "oracle_mid_final_median": ora_final,
                "delta_oracle_minus_ordered": delta,
                "oracle_improves": bool(delta > 0),
            }
        )
    gate = pd.DataFrame(gate_rows)
    gate_path = output_root / "phase0_go_nogo.csv"
    gate.to_csv(gate_path, index=False)

    ep = summary[summary["condition"] == "oracle_endpoint"]
    ep_med = float(ep.iloc[0]["median"]) if not ep.empty else float("nan")

    manifest = {
        "phase": 0,
        "mode": "state_feedback_oracle",
        "config": str(args.config.resolve()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "model": str(model_path),
        "n_start_cells": len(start_ds),
        "observed_state": observed_state,
        "oracle_after_step": oracle_after_step,
        "orders": [order_label(o) for o in orders],
        "oracle_endpoint_median": ep_med,
        "gate": gate_rows,
        "hard_gate": (
            "GO if any order has oracle_mid final median > ordered_rank_edit; "
            "also inspect oracle_endpoint ceiling. STOP Phase 1–2 if no improvement."
        ),
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {gate_path}", flush=True)
    print(json.dumps({"gate": gate_rows, "oracle_endpoint_median": ep_med}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
