#!/usr/bin/env python3
"""Stability of multi-step State-feedback ISP (feedback after every step).

Pre-registered criteria S1-S4 and the decision rule are in
docs/state_feedback_decode_methods.md ("Multi-step stability evaluation").

For each decoder seed (decoder retrained; the train/val gene split stays fixed):

| Chain | What it measures |
|-------|------------------|
| configured, multi-step + ``idle_events`` | per-event change (S1); extra reranks with no new perturbation must converge, not drift (S2) |
| ``n_random_chains`` random chains, multi-step | endpoint gain over Ordered rank-edit must be larger for the configured chain than for every random chain (S3); a decoder trained on the endpoint teacher could pull any chain toward the goal |

Every chain gets feedback after every step (``state_feedback.multistep``). Random chains
are drawn by ``state_feedback.random_chains`` (structure- and position-matched; the
draw is written to ``random_chains.json``). The configured gain beyond the mean
random-chain gain (specific gain) is reported per seed with a cell-bootstrap CI.

Ordered rank-edit runs once per chain (no decoder) and is the reference path.
Across seeds, the final configured encodings and endpoint shifts must agree (S4).

With ``state_feedback.placebo_contrast.enabled`` the decoder is trained on, and every
chain is reranked with, the placebo contrast of ``state_feedback.placebo_contrast``
(estimation placebos drawn separately from the random chains; written to
``estimation_placebos.json``). The configured schedule's decoder is used for its random
chains.

Usage:
  python3 core/run_state_feedback_stability.py --config core/config/state_feedback_isp.yaml
  python3 core/run_state_feedback_stability.py --config ... --max-ncells 5 --seeds 0 \
      --idle-events 2 --n-random-chains 1
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from ordered_rank_edit import config_block, parse_steps
import run_ordered_rank_edit_isp as ore
import run_state_feedback_isp as rsf

from state_feedback import feedback as fb
from state_feedback import gene_states as gs
from state_feedback import placebo_contrast as pc
from state_feedback import random_chains
from state_feedback.multistep import check_feedback_config
from state_feedback.stability import (
    DEFAULT_TOPK,
    ChainTracker,
    cross_seed_agreement,
    running_random_mean,
    specific_gain,
    specific_gain_conditional,
    stability_verdict,
)
from state_feedback.teacher import observed_delta_rank, split_tokens

CONFIGURED = "configured"


def _final_overexpressed(steps, token_by_step) -> frozenset[int]:
    pinned: set[int] = set()
    for step, tokens in zip(steps, token_by_step):
        if ore.normalize_step_type(str(step["type"])) == ore.PERTURB_OVEREXPRESS:
            pinned.update(int(t) for t in tokens)
        else:
            pinned.difference_update(int(t) for t in tokens)
    return frozenset(pinned)


def _row_medians(rows: list[dict[str, Any]]) -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    for r in rows:
        kind = "feedback" if r["step_name"] == "feedback" else "step"
        out[(kind, int(r["step"]))] = float(r["median"])
    return out


def _endpoint(rows: list[dict[str, Any]]) -> float:
    last = max(int(r["step"]) for r in rows)
    at_last = [r for r in rows if int(r["step"]) == last]
    return float(at_last[-1]["median"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-ncells", type=int, default=None)
    parser.add_argument("--forward-batch-size", default=None)
    parser.add_argument("--nproc", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--idle-events", type=int, default=None)
    parser.add_argument("--n-random-chains", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--random-seeds",
        type=int,
        nargs="+",
        default=None,
        help="One independent draw of --n-random-chains chains per seed.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Subdirectory under the output root (default: state_feedback_stability_<HHMMSS>).",
    )
    parser.add_argument(
        "--aggregate",
        type=Path,
        nargs="+",
        default=None,
        help="Combine finished run directories (e.g. one per seed) into one verdict, "
        "written to --output-root.",
    )
    args = parser.parse_args()

    if args.aggregate:
        if args.output_root is None:
            parser.error("--aggregate needs --output-root")
        aggregate(list(args.aggregate), args.output_root)
        return 0
    if args.config is None:
        parser.error("--config is required")

    t0 = time.time()
    cfg = yaml.safe_load(args.config.read_text())
    paths = cfg.get("paths") or {}
    pert = cfg.get("perturbation") or {}
    mdl = cfg.get("model") or {}
    isp_cfg = cfg.get("isp") or {}
    runtime = cfg.get("runtime") or {}
    sf_cfg = cfg.get("state_feedback") or {}
    dec_cfg = sf_cfg.get("decoder") or {}
    st_cfg = sf_cfg.get("stability") or {}

    seeds = [int(s) for s in (args.seeds or st_cfg.get("seeds") or [0, 1, 2])]
    idle_events = int(
        args.idle_events if args.idle_events is not None else st_cfg.get("idle_events", 5)
    )
    n_random = int(
        args.n_random_chains
        if args.n_random_chains is not None
        else st_cfg.get("n_random_chains", 5)
    )
    random_seeds = [
        int(s) for s in (args.random_seeds or st_cfg.get("random_seeds") or [0])
    ]
    min_stratum = int(st_cfg.get("min_stratum", 50))
    pc_cfg = sf_cfg.get("placebo_contrast") or {}
    use_contrast = bool(pc_cfg.get("enabled", False))
    n_estimation = int(pc_cfg.get("n_estimation", 10))
    estimation_seed = int(pc_cfg.get("estimation_draw_seed", 1))
    if use_contrast and estimation_seed in random_seeds:
        raise ValueError(
            "state_feedback.placebo_contrast.estimation_draw_seed must differ from "
            "stability.random_seeds (estimation and comparison placebos are separate draws)"
        )
    if "random_seed" in st_cfg:
        raise ValueError("state_feedback.stability.random_seed is now random_seeds (a list)")
    topk = tuple(int(k) for k in (st_cfg.get("topk") or DEFAULT_TOPK))
    for key in ("modes", "single_event_at", "single_event_random"):
        if key in st_cfg:
            raise ValueError(
                f"state_feedback.stability.{key} was removed: only feedback after every step is run"
            )

    dataset_path = Path(paths["dataset"]).expanduser()
    model_path = Path(paths["geneformer_model"]).expanduser()
    if args.output_root is not None:
        output_root = args.output_root
    else:
        output_root = Path(paths["output_root"]).expanduser()
        if paths.get("output_date_subdir", True):
            output_root = output_root / datetime.now(timezone.utc).strftime("%Y%m%d")
    output_root = output_root / (
        args.run_name
        or "state_feedback_stability_" + datetime.now(timezone.utc).strftime("%H%M%S")
    )

    state_key = str(pert["state_key"])
    start_state = str(pert["start_state"])
    goal_state = str(pert["end_state"])
    observed_state = str(sf_cfg.get("observed_state") or goal_state)
    max_ncells = args.max_ncells if args.max_ncells is not None else isp_cfg.get("max_ncells")
    emb_layer = int(isp_cfg.get("emb_layer", 0))
    nproc = args.nproc if args.nproc is not None else int(runtime.get("nproc", 1))
    hysteresis = float(sf_cfg.get("hysteresis", 0.0))
    pin_overexpressed = bool(sf_cfg.get("pin_overexpressed", True))
    ctrl_reference = str(sf_cfg.get("ctrl_reference", "start"))
    if ctrl_reference not in {"start", "previous"}:
        raise ValueError("state_feedback.ctrl_reference must be 'start' or 'previous'")

    log_species_banner(species_from_config(cfg))
    steps = parse_steps(sf_cfg) or parse_steps(config_block(cfg))
    if not steps:
        raise ValueError("stability evaluation needs explicit state_feedback.steps")
    check_feedback_config(sf_cfg, len(steps))
    delete_steps = pc.delete_step_indices(steps)
    if use_contrast and delete_steps and ctrl_reference != "start":
        raise ValueError("the placebo contrast of delete steps needs ctrl_reference: start")
    random_steps = st_cfg.get("random_steps")
    require_detected = list(sf_cfg.get("require_detected") or [])

    from datasets import load_from_disk

    print(f"Loading dataset: {dataset_path}", flush=True)
    dataset = load_from_disk(str(dataset_path))
    if require_detected:
        req_tokens = set(ore._resolve_gene_tokens(cfg, require_detected)[0])
        start_ds = ore._select_start_cells(dataset, state_key, start_state, None)
        start_ds = start_ds.filter(lambda x: req_tokens <= set(x["input_ids"]))
        if max_ncells is not None and len(start_ds) > int(max_ncells):
            start_ds = start_ds.select(range(int(max_ncells)))
        print(f"Start cells restricted to cells with {require_detected} detected", flush=True)
    else:
        start_ds = ore._select_start_cells(dataset, state_key, start_state, max_ncells)
    print(f"Start cells ({start_state}): n={len(start_ds)}", flush=True)
    obs_cap = int(sf_cfg.get("observed_max_ncells") or max_ncells or 3000)
    obs_ds = ore._select_start_cells(dataset, state_key, observed_state, obs_cap)
    teacher_ctrl_ds = ore._select_start_cells(dataset, state_key, start_state, obs_cap)
    teacher = observed_delta_rank(
        teacher_ctrl_ds["input_ids"],
        obs_ds["input_ids"],
        min_detection_count=int(sf_cfg.get("min_detection_count", 5)),
    )
    print(f"Teacher delta-rank tokens: {len(teacher)}", flush=True)

    model = isp.load_model(
        mdl.get("type", "CellClassifier"), int(mdl.get("num_classes", 2)), str(model_path)
    )
    layer_to_quant = isp.quant_layers(model) + emb_layer
    model_input_size = isp.get_model_input_size(model)
    backend = backend_from_config(cfg)
    with open(backend.token_dictionary, "rb") as fh:
        pad_token_id = pickle.load(fh).get("<pad>")
    fbs_raw = args.forward_batch_size or runtime.get("forward_batch_size", "auto")
    forward_batch_size = ore.resolve_dual_forward_batch_size(
        coerce_batch_size(fbs_raw, default=default_isp_forward_batch_size(backend.max_input_size)),
        model,
        start_ds,
        pad_token_id,
        str(model_path),
    )
    print(f"forward_batch_size={forward_batch_size}", flush=True)

    centroid_states = [start_state, goal_state, *list(pert.get("alt_states") or [])]
    centroid_ds = ore._centroid_dataset(dataset, state_key, centroid_states, max_ncells, nproc)
    state_embs = isp.get_cell_state_avg_embs(
        model,
        centroid_ds,
        {
            "state_key": state_key,
            "start_state": start_state,
            "goal_state": goal_state,
            "alt_states": list(pert.get("alt_states") or []),
        },
        layer_to_quant,
        pad_token_id,
        forward_batch_size,
        ore._gpu_resident_map_workers(nproc),
    )
    ore._empty_cuda_cache()

    token_by_step: list[list[int]] = []
    for step in steps:
        tokens, resolved = ore._resolve_gene_tokens(cfg, step["genes"])
        token_by_step.append(list(tokens))
        print(f"Step {step['index']} {step['type']} {step['name']}: {', '.join(resolved)}",
              flush=True)

    with open(backend.token_dictionary, "rb") as fh:
        vocab = pickle.load(fh)
    population = sorted(int(t) for k, t in vocab.items() if not str(k).startswith("<"))
    tok_names = {int(t): str(k) for k, t in vocab.items()}
    if getattr(backend, "gene_symbol_to_ensembl", None):
        with open(backend.gene_symbol_to_ensembl, "rb") as fh:
            ens_to_symbol = {v: k for k, v in pickle.load(fh).items()}
        tok_names = {t: ens_to_symbol.get(e, e) for t, e in tok_names.items()}
    start_detection, start_rank = random_chains.start_profiles(
        gs.raw_input_ids(start_ds, 0, len(start_ds), model_input_size)
    )
    goal_detection, _ = random_chains.start_profiles(obs_ds["input_ids"])
    covariates = {
        "start_detection": {t: start_detection.get(t, 0.0) for t in population},
        "start_rank": start_rank,
        "goal_detection": {t: goal_detection.get(t, 0.0) for t in population},
        "teacher_delta": {int(t): float(v) for t, v in teacher.items()},
    }
    chains: dict[str, tuple[list[dict[str, Any]], list[list[int]]]] = {
        CONFIGURED: (list(steps), token_by_step),
    }
    draws: dict[str, Any] = {}
    strata_pools: dict[str, Any] = {}
    for rs in random_seeds:
        drawn, record = random_chains.draw_random_chains(
            steps, token_by_step, population, start_detection, start_rank,
            n_chains=n_random, seed=rs, min_stratum=min_stratum, prefix=f"random_s{rs}_",
        )
        if random_steps:
            drawn = random_chains.randomize_steps_only(drawn, steps, token_by_step, random_steps)
            record["random_steps"] = [int(i) for i in random_steps]
        chains.update(drawn)
        draws[str(rs)] = {
            **{k: v for k, v in record.items() if k != "strata"},
            "strata": {
                str(t): {k: v for k, v in info.items() if k != "pool"}
                for t, info in record["strata"].items()
            },
            **random_chains.describe(record, covariates, tok_names),
        }
        strata_pools[str(rs)] = {str(t): info["pool"] for t, info in record["strata"].items()}
        for row in draws[str(rs)]["balance"]:
            print(
                f"Random set s{rs} / {row['configured_gene']}: stratum={row['size']} "
                f"({row['rule']}, rank_window={row['rank_window']}) "
                f"goal_detection SMD={row['goal_detection_smd']:+.2f} "
                f"teacher_delta SMD={row['teacher_delta_smd']:+.2f}",
                flush=True,
            )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "random_chains.json").write_text(json.dumps(draws, indent=2) + "\n")
    (output_root / "random_strata.json").write_text(json.dumps(strata_pools) + "\n")

    if delete_steps:
        start_sets = [set(map(int, ids)) for ids in gs.raw_input_ids(
            start_ds, 0, len(start_ds), model_input_size)]
        pd.DataFrame([
            {"chain": chain, "cell_index": i,
             "treated": {int(t) for k in delete_steps for t in tbs[k]} <= cell}
            for chain, (_sl, tbs) in chains.items() for i, cell in enumerate(start_sets)
        ]).to_csv(output_root / "delete_cells.csv", index=False)

    chain_subs: dict[str, list[Any]] = {}
    if use_contrast:
        est_placebos, _ = pc.draw_estimation_placebos(
            steps, token_by_step, population, start_detection, start_rank,
            n=n_estimation, seed=estimation_seed, min_stratum=min_stratum,
        )
        for chain, (_sl, tbs) in chains.items():
            chain_subs[chain] = pc.slot_substitutions(tbs, est_placebos, delete_steps)
        left_out = {c: n_estimation - len(s) for c, s in chain_subs.items()
                    if len(s) < n_estimation}
        print(
            f"Placebo contrast: {n_estimation} estimation placebos (draw seed "
            f"{estimation_seed}); left out for sharing a gene: {left_out or 'none'}",
            flush=True,
        )
        (output_root / "estimation_placebos.json").write_text(json.dumps({
            "draw_seed": estimation_seed,
            "n_estimation": n_estimation,
            "placebos": [[[tok_names.get(int(t), str(t)) for t in ts] for ts in p]
                         for p in est_placebos],
            "placebo_tokens": est_placebos,
            "left_out_by_chain": left_out,
        }, indent=2) + "\n")

    split_seed = int(dec_cfg.get("seed", 0))
    train_tokens, val_tokens = split_tokens(
        teacher.keys(), val_fraction=float(dec_cfg.get("val_fraction", 0.2)), seed=split_seed
    )
    output_root.mkdir(parents=True, exist_ok=True)
    n_steps = len(steps)
    start_ids = gs.raw_input_ids(start_ds, 0, len(start_ds), model_input_size)
    workers = ore._gpu_resident_map_workers(nproc)
    common = dict(
        layer_to_quant=layer_to_quant,
        pad_token_id=pad_token_id,
        model_input_size=model_input_size,
        forward_batch_size=forward_batch_size,
        nproc=nproc,
    )
    timings: dict[str, float] = {"setup_s": time.time() - t0}

    def _ids(ds) -> list[list[int]]:
        return gs.raw_input_ids(ds, 0, len(ds), model_input_size)

    def run_tracked(label, step_list, tbs, out_dir, rerank_fn, *, reference=None):
        tracker = ChainTracker(label, start_ids, reference=reference, topk=topk)
        captured: dict[str, Any] = {"ref": {}}

        def on_encoding(kind, step, ds):
            ids = _ids(ds)
            if rerank_fn is None:
                captured["ref"][int(step)] = ids
            tracker.observe(kind, step, ids)
            captured["ds"] = ds

        rows = rsf.run_condition(
            label, model, start_ds, step_list, tbs, goal_state, state_embs, out_dir,
            rerank_fn=rerank_fn,
            pin_overexpressed=pin_overexpressed,
            ctrl_reference=ctrl_reference,
            on_encoding=on_encoding,
            **common,
        )
        medians = _row_medians(rows)
        for row in tracker.rows:
            row["shift_median"] = medians.get((row["kind"], row["step"]), float("nan"))
        return tracker, rows, captured

    # Reference path: Ordered rank-edit, once per chain.
    event_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    references: dict[str, dict[int, list[list[int]]]] = {}
    ore_endpoint: dict[str, float] = {}
    t1 = time.time()
    for chain, (step_list, tbs) in chains.items():
        print(f"=== ordered_rank_edit / {chain} ===", flush=True)
        tracker, rows, captured = run_tracked(
            f"ore_{chain}", step_list, tbs,
            output_root / "ordered_rank_edit" / chain, None,
        )
        references[chain] = captured["ref"]
        ore_endpoint[chain] = _endpoint(rows)
        for row in tracker.rows:
            event_rows.append({"seed": None, "mode": "ordered_rank_edit", **row})
        endpoint_rows.append({
            "seed": None, "chain": chain, "mode": "ordered_rank_edit",
            "endpoint_median": ore_endpoint[chain],
        })
    timings["ordered_rank_edit_s"] = time.time() - t1

    def _flush() -> None:
        pd.DataFrame(event_rows).to_csv(output_root / "events.csv", index=False)
        pd.DataFrame(endpoint_rows).to_csv(output_root / "endpoints.csv", index=False)

    _flush()
    decoders: dict[int, Any] = {}
    pinned = _final_overexpressed(steps, token_by_step)

    for seed in seeds:
        ts = time.time()
        seed_dir = output_root / f"seed{seed}"
        print(f"=== seed {seed}: training decoder ===", flush=True)
        decoder, info = rsf._build_decoder(
            model, start_ds, steps, token_by_step, teacher, {**dec_cfg, "seed": seed},
            train_tokens, val_tokens, out_dir=seed_dir / "decoder", **common,
            contrast_subs=chain_subs.get(CONFIGURED),
        )
        decoders[seed] = info.get("selected", {})
        ore._empty_cuda_cache()
        timings[f"seed{seed}_decoder_s"] = time.time() - ts
        fwd = {k: common[k] for k in (
            "layer_to_quant", "pad_token_id", "model_input_size", "forward_batch_size",
        )}

        def make_rerank(chain, _decoder=decoder):
            if not use_contrast:
                return lambda ctrl_ds, pert_ds: fb.rerank_linear_deltarank(
                    model, _decoder, ctrl_ds, pert_ds, hysteresis=hysteresis, **fwd
                )
            return lambda ctrl_ds, pert_ds: pc.rerank_placebo_contrast(
                model, _decoder, ctrl_ds, pert_ds, chain_subs[chain],
                hysteresis=hysteresis, **fwd,
            )

        gains: dict[str, float] = {}
        for chain, (step_list, tbs) in chains.items():
            tc = time.time()
            print(f"=== seed {seed} / multi-step / {chain} ===", flush=True)
            rerank_fn = make_rerank(chain)
            tracker, rows, captured = run_tracked(
                f"multi_{chain}", step_list, tbs, seed_dir / f"multi_{chain}",
                rerank_fn, reference=references[chain],
            )
            end = _endpoint(rows)
            gains[chain] = end - ore_endpoint[chain]
            endpoint_rows.append({
                "seed": seed, "chain": chain, "mode": "multi_step",
                "endpoint_median": end, "ordered_rank_edit_endpoint": ore_endpoint[chain],
                "gain_over_ordered_rank_edit": gains[chain],
            })
            if chain == CONFIGURED:
                with open(seed_dir / "final_configured.pkl", "wb") as fh:
                    pickle.dump([list(map(int, x)) for x in tracker.last], fh)
                working = captured["ds"]
                batch_state = [int(forward_batch_size)]
                for k in range(1, idle_events + 1):
                    ctrl_ds = start_ds if ctrl_reference == "start" else working
                    new_ids, _diag = rerank_fn(ctrl_ds, working)
                    if pin_overexpressed and pinned:
                        new_ids, _diag = fb.pin_tokens_front(_ids(working), new_ids, pinned)
                    working = fb.replace_input_ids(working, new_ids, num_proc=workers)
                    df = rsf._score_cell_mean(
                        model, start_ds, working, goal_state, state_embs,
                        layer_to_quant=layer_to_quant, pad_token_id=pad_token_id,
                        model_input_size=model_input_size, nproc=nproc,
                        batch_state=batch_state,
                    )
                    idle_dir = seed_dir / f"multi_{chain}" / f"idle{k:02d}"
                    idle_dir.mkdir(parents=True, exist_ok=True)
                    df.to_csv(idle_dir / "per_cell_shifts.csv", index=False)
                    shift = rsf._summarize(df)["median"]
                    row = tracker.observe(
                        "idle", k, new_ids, ref_step=n_steps, shift_median=shift
                    )
                    print(
                        f"  [idle {k}] shift={shift:.6f} "
                        f"rho={row['event_spearman_median']:.4f} "
                        f"disp={row['event_disp_median_median']:.1f} "
                        f"vs_ore_rho={row.get('vs_ref_spearman_median', float('nan')):.4f}",
                        flush=True,
                    )
                    ore._empty_cuda_cache()
            for row in tracker.rows:
                event_rows.append({"seed": seed, "mode": "multi_step", **row})
            timings[f"seed{seed}_multi_{chain}_s"] = time.time() - tc

        _flush()
        del decoder
        ore._empty_cuda_cache()

    timings["total_s"] = time.time() - t0
    _flush()
    manifest = {
        "mode": "state_feedback_stability",
        "config": str(args.config.resolve()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "model": str(model_path),
        "shift_metric": rsf.SHIFT_METRIC,
        "n_start_cells": len(start_ds),
        "steps": [{"name": s["name"], "type": s["type"], "genes": s["genes"]} for s in steps],
        "chains": {c: [s["genes"] for s in sl] for c, (sl, _) in chains.items()},
        "seeds": seeds,
        "split_seed": split_seed,
        "idle_events": idle_events,
        "n_random_chains": n_random,
        "random_seeds": random_seeds,
        "random_population": "model vocabulary minus special tokens and configured genes",
        "random_min_stratum": min_stratum,
        "topk": list(topk),
        "feedback": {"policy": "after_every_step", "events_per_chain": n_steps},
        "pin_overexpressed": pin_overexpressed,
        "ctrl_reference": ctrl_reference,
        "hysteresis": hysteresis,
        "delete_steps": sorted(i + 1 for i in delete_steps),
        "random_steps": [int(i) for i in random_steps] if random_steps else None,
        "require_detected": require_detected,
        "placebo_contrast": {
            "enabled": use_contrast,
            "n_estimation": n_estimation if use_contrast else 0,
            "estimation_draw_seed": estimation_seed if use_contrast else None,
        },
        "decoders": decoders,
        "forward_batch_size": forward_batch_size,
        "timings": timings,
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"timings": timings}, indent=2), flush=True)
    aggregate([output_root], output_root)
    return 0


def aggregate(run_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    """Cross-seed agreement and the S1-S4 verdict over one or more finished runs.

    Seed jobs run separately (one decoder seed each) are combined here; a seed may
    appear in only one run.
    """
    events = pd.concat([pd.read_csv(d / "events.csv") for d in run_dirs], ignore_index=True)
    endpoints = pd.concat(
        [pd.read_csv(d / "endpoints.csv") for d in run_dirs], ignore_index=True
    )
    topk = tuple(json.loads((run_dirs[0] / "run_manifest.json").read_text())["topk"])
    finals = _load_finals(run_dirs, "final_configured.pkl")

    fed = events[(events["mode"] == "multi_step") & (events["chain"] == f"multi_{CONFIGURED}")]
    rows_by_seed = {
        int(s): g.sort_values("event_index").to_dict("records") for s, g in fed.groupby("seed")
    }
    multi = endpoints[endpoints["mode"] == "multi_step"]
    gain_by_seed = {
        int(s): dict(zip(g["chain"], g["gain_over_ordered_rank_edit"]))
        for s, g in multi.groupby("seed")
    }
    endpoint_by_seed = {
        int(r["seed"]): float(r["endpoint_median"])
        for _, r in multi[multi["chain"] == CONFIGURED].iterrows()
    }
    agreement = cross_seed_agreement(finals, topk)
    verdict = stability_verdict(
        rows_by_seed, gain_by_seed, endpoint_by_seed, agreement, configured=CONFIGURED
    )
    verdict["runs"] = [str(d) for d in run_dirs]
    verdict["seeds"] = sorted(finals)

    out_dir.mkdir(parents=True, exist_ok=True)
    if [out_dir] != list(run_dirs):
        events.to_csv(out_dir / "events.csv", index=False)
        endpoints.to_csv(out_dir / "endpoints.csv", index=False)
    pd.DataFrame(agreement).to_csv(out_dir / "seed_agreement.csv", index=False)
    (out_dir / "stability_verdict.json").write_text(
        json.dumps(verdict, indent=2, default=str) + "\n"
    )
    print(json.dumps({"seeds": verdict["seeds"], "passed": verdict["passed"],
                      "multi_step_supported": verdict["multi_step_supported"]}, indent=2),
          flush=True)
    print(f"Wrote {out_dir / 'stability_verdict.json'}", flush=True)

    spec_rows: list[dict[str, Any]] = []
    running_rows: list[dict[str, Any]] = []
    for d in run_dirs:
        for seed_dir in sorted(d.glob("seed*")):
            if not seed_dir.is_dir():
                continue
            seed = int(seed_dir.name.removeprefix("seed"))
            spec = _specific_gain(d, seed_dir, seed=seed)
            if spec is not None:
                spec_rows += [{"seed": seed, **r} for r in spec[0]]
                running_rows += [{"seed": seed, **r} for r in spec[1]]
    if spec_rows:
        pd.DataFrame(spec_rows).to_csv(out_dir / "specificity.csv", index=False)
        pd.DataFrame(running_rows).to_csv(out_dir / "random_running_mean.csv", index=False)
        for r in spec_rows:
            print(
                f"  seed {r['seed']} set={r['set']:<4} n={int(r['n_random_chains'])} "
                f"gain={r['configured_gain_mean']:.4f} "
                f"random={r['random_gain_mean']:.4f}±{r['random_gain_se_chains']:.4f} "
                f"specific={r['specific_gain_mean']:+.4f} "
                f"[{r['specific_gain_ci_low']:+.4f}, {r['specific_gain_ci_high']:+.4f}] "
                f"null_p={r.get('null_empirical_p', float('nan')):.3f} "
                f"z={r.get('z_vs_random', float('nan')):.2f} "
                f"rank={int(r['rank_among_random'])}",
                flush=True,
            )
    return verdict


def _load_finals(run_dirs: list[Path], name: str) -> dict[int, list]:
    finals: dict[int, list] = {}
    for d in run_dirs:
        for path in sorted(d.glob(f"seed*/{name}")):
            seed = int(path.parent.name.removeprefix("seed"))
            if seed in finals:
                raise ValueError(f"seed {seed} appears in more than one run: {path}")
            with open(path, "rb") as fh:
                finals[seed] = pickle.load(fh)
    return finals


def _final_cells(chain_dir: Path) -> pd.Series | None:
    """Per-cell end-point shift of one chain: last feedback if any, else last step."""
    steps = sorted(p for p in chain_dir.glob("step[0-9][0-9]_*") if p.is_dir())
    if not steps:
        return None
    fed = [p for p in steps if p.name.endswith("_feedback")]
    last = max(fed or steps, key=lambda p: (int(p.name[4:6]), p.name.endswith("_feedback")))
    df = pd.read_csv(last / "per_cell_shifts.csv")
    return df.set_index("cell_index")["Shift_to_goal_end"].sort_index()


def _random_set(chain: str) -> tuple[str, int]:
    """``random_s1_7`` -> ("s1", 7); older ``random7`` -> ("all", 7)."""
    body = chain.removeprefix("random")
    if body.startswith("_s"):
        rs, _, idx = body[2:].partition("_")
        return f"s{rs}", int(idx)
    return "all", int(body)


def _specific_gain(run_dir: Path, seed_dir: Path, *, seed: int, prefix: str = "multi_"):
    """Specific gain over all random chains and per independent random draw.

    Returns ``(rows, running)``: one ``specific_gain`` row per random set (``set``
    = ``all`` or ``s<random_seed>``) and the running random mean per set in draw order.
    Chains with delete steps (``delete_cells.csv``) use ``specific_gain_conditional``.
    """
    conf = _final_cells(seed_dir / f"{prefix}{CONFIGURED}")
    conf_ref = _final_cells(run_dir / "ordered_rank_edit" / CONFIGURED)
    if conf is None or conf_ref is None:
        return None
    treated = None
    if (run_dir / "delete_cells.csv").exists():
        cells = pd.read_csv(run_dir / "delete_cells.csv")
        treated = {
            chain: g.set_index("cell_index")["treated"].astype(bool).loc[conf.index].to_numpy()
            for chain, g in cells.groupby("chain")
        }
    by_set: dict[str, list[tuple[int, str, Any, Any]]] = {}
    for chain_dir in seed_dir.glob(f"{prefix}random*"):
        chain = chain_dir.name.removeprefix(prefix)
        r = _final_cells(chain_dir)
        rr = _final_cells(run_dir / "ordered_rank_edit" / chain)
        if r is None or rr is None:
            continue
        name, idx = _random_set(chain)
        by_set.setdefault(name, []).append(
            (idx, chain, r.loc[conf.index].to_numpy(), rr.loc[conf.index].to_numpy())
        )
    if not by_set:
        return None
    sets = {k: sorted(v, key=lambda x: x[0]) for k, v in sorted(by_set.items())}
    if "all" not in sets:
        sets = {"all": [c for v in sets.values() for c in v], **sets}
    rows, running = [], []
    c, cr = conf.to_numpy(), conf_ref.loc[conf.index].to_numpy()
    for name, chains in sets.items():
        randoms = [x[2] for x in chains]
        refs = [x[3] for x in chains]
        if treated is None:
            spec = specific_gain(c, cr, randoms, refs, seed=seed)
        else:
            spec = specific_gain_conditional(
                c, cr, randoms, refs, treated[CONFIGURED], [treated[x[1]] for x in chains],
                seed=seed,
            )
        rows.append({"set": name, **spec})
        if name != "all" or len(sets) == 1:
            running += [{"set": name, **r} for r in running_random_mean(randoms, refs)]
    return rows, running


if __name__ == "__main__":
    raise SystemExit(main())
