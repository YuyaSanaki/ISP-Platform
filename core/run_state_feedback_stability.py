#!/usr/bin/env python3
"""Stability of multi-step State-feedback ISP (feedback after every step).

Pre-registered criteria S1-S4 and the decision rule are in
docs/state_feedback_decode_methods.md ("Multi-step stability evaluation").

For each decoder seed (decoder retrained; the train/val gene split stays fixed):

| Chain | What it measures |
|-------|------------------|
| configured, multi-step + ``idle_events`` | per-event change (S1); extra reranks with no new perturbation must converge, not drift (S2) |
| ``n_random_chains`` random chains, multi-step | endpoint gain over Ordered rank-edit must be larger for the configured chain than for every random chain (S3); a decoder trained on the endpoint teacher could pull any chain toward the goal |
| configured and random chains, single event | one feedback event after ``single_event_at`` (first or last step); the configured gain beyond the random-chain gain is compared with multi-step (D1, D2) |

Ordered rank-edit runs once per chain (no decoder) and is the reference path.
Across seeds, the final configured encodings and endpoint shifts must agree (S4).

Usage:
  python3 core/run_state_feedback_stability.py --config core/config/state_feedback_isp.yaml
  python3 core/run_state_feedback_stability.py --config ... --max-ncells 5 --seeds 0 \
      --idle-events 2 --n-random-chains 1
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
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
from state_feedback.multistep import MultiStepConfig
from state_feedback.stability import (
    DEFAULT_TOPK,
    ChainTracker,
    cross_seed_agreement,
    mode_comparison_verdict,
    specific_gain,
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
        "--modes",
        nargs="+",
        default=None,
        help="multi and/or single (default: state_feedback.stability.modes, else both).",
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
    random_seed = int(st_cfg.get("random_seed", 0))
    topk = tuple(int(k) for k in (st_cfg.get("topk") or DEFAULT_TOPK))
    modes = list(args.modes or st_cfg.get("modes") or ["multi", "single"])
    unknown_modes = sorted(set(modes) - {"multi", "single"})
    if unknown_modes:
        raise ValueError(f"unknown stability modes {unknown_modes}; valid: multi, single")
    single_at = str(st_cfg.get("single_event_at", "last"))
    if single_at not in {"first", "last"}:
        raise ValueError("state_feedback.stability.single_event_at must be 'first' or 'last'")
    single_random = bool(st_cfg.get("single_event_random", True))

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
    feedback_after_step = int(sf_cfg.get("feedback_after_step", 1))
    pin_overexpressed = bool(sf_cfg.get("pin_overexpressed", True))
    ctrl_reference = str(sf_cfg.get("ctrl_reference", "start"))
    if ctrl_reference not in {"start", "previous"}:
        raise ValueError("state_feedback.ctrl_reference must be 'start' or 'previous'")
    multi_step = MultiStepConfig.from_mapping(sf_cfg.get("multi_step"))

    log_species_banner(species_from_config(cfg))
    steps = parse_steps(sf_cfg) or parse_steps(config_block(cfg))
    if not steps:
        raise ValueError("stability evaluation needs explicit state_feedback.steps")

    from datasets import load_from_disk

    print(f"Loading dataset: {dataset_path}", flush=True)
    dataset = load_from_disk(str(dataset_path))
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

    configured_tokens = {t for ts in token_by_step for t in ts}
    pool = sorted(set(int(t) for t in teacher) - configured_tokens)
    rng = random.Random(random_seed)
    chains: dict[str, tuple[list[dict[str, Any]], list[list[int]]]] = {
        CONFIGURED: (list(steps), token_by_step),
    }
    for r in range(n_random):
        picks = rng.sample(pool, len(steps))
        chains[f"random{r}"] = (
            [
                {"index": i + 1, "name": f"random{r}_{t}", "type": s["type"], "genes": [str(t)]}
                for i, (s, t) in enumerate(zip(steps, picks))
            ],
            [[t] for t in picks],
        )

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

    def run_tracked(label, step_list, tbs, out_dir, rerank_fn, *, every_step, reference=None,
                    after_step=feedback_after_step):
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
            feedback_after_step=after_step,
            feedback_every_step=every_step,
            feedback_after_last_step=True,
            pin_overexpressed=pin_overexpressed,
            ctrl_reference=ctrl_reference,
            multi_step=multi_step,
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
            output_root / "ordered_rank_edit" / chain, None, every_step=False,
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
        )
        decoders[seed] = info.get("selected", {})
        ore._empty_cuda_cache()
        timings[f"seed{seed}_decoder_s"] = time.time() - ts

        def rerank_fn(ctrl_ds, pert_ds, _decoder=decoder):
            return fb.rerank_linear_deltarank(
                model, _decoder, ctrl_ds, pert_ds, hysteresis=hysteresis, **{
                    k: common[k] for k in (
                        "layer_to_quant", "pad_token_id", "model_input_size",
                        "forward_batch_size",
                    )
                },
            )

        gains: dict[str, float] = {}
        for chain, (step_list, tbs) in (chains.items() if "multi" in modes else ()):
            tc = time.time()
            print(f"=== seed {seed} / multi-step / {chain} ===", flush=True)
            tracker, rows, captured = run_tracked(
                f"multi_{chain}", step_list, tbs, seed_dir / f"multi_{chain}",
                rerank_fn, every_step=True, reference=references[chain],
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

        single_mode = f"single_{single_at}"
        single_chains = list(chains) if single_random else [CONFIGURED]
        for chain in (single_chains if "single" in modes else ()):
            tc = time.time()
            print(f"=== seed {seed} / single event ({single_at} step) / {chain} ===", flush=True)
            step_list, tbs = chains[chain]
            tracker, rows, _ = run_tracked(
                f"{single_mode}_{chain}", step_list, tbs,
                seed_dir / f"{single_mode}_{chain}", rerank_fn, every_step=False,
                reference=references[chain],
                after_step=n_steps if single_at == "last" else feedback_after_step,
            )
            end = _endpoint(rows)
            endpoint_rows.append({
                "seed": seed, "chain": chain, "mode": single_mode,
                "endpoint_median": end,
                "ordered_rank_edit_endpoint": ore_endpoint[chain],
                "gain_over_ordered_rank_edit": end - ore_endpoint[chain],
            })
            if chain == CONFIGURED:
                with open(seed_dir / f"final_{single_mode}_configured.pkl", "wb") as fh:
                    pickle.dump([list(map(int, x)) for x in tracker.last], fh)
            for row in tracker.rows:
                event_rows.append({"seed": seed, "mode": single_mode, **row})
            timings[f"seed{seed}_{single_mode}_{chain}_s"] = time.time() - tc

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
        "modes": modes,
        "single_event_at": single_at,
        "single_event_random": single_random,
        "idle_events": idle_events,
        "n_random_chains": n_random,
        "random_seed": random_seed,
        "topk": list(topk),
        "feedback_after_step": feedback_after_step,
        "pin_overexpressed": pin_overexpressed,
        "ctrl_reference": ctrl_reference,
        "multi_step": multi_step.to_dict(),
        "hysteresis": hysteresis,
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
    spec_by_seed: dict[int, dict[str, dict[str, float]]] = {}
    for d in run_dirs:
        for seed_dir in sorted(d.glob("seed*")):
            if not seed_dir.is_dir():
                continue
            seed = int(seed_dir.name.removeprefix("seed"))
            for mode, prefix in (("multi_step", "multi_"), (SINGLE_LAST, f"{SINGLE_LAST}_")):
                spec = _mode_specific_gain(d, seed_dir, prefix, seed=seed)
                if spec is None:
                    continue
                spec_by_seed.setdefault(seed, {})[mode] = spec
                spec_rows.append({"seed": seed, "mode": mode, **spec})
    if spec_rows:
        comparison = mode_comparison_verdict(spec_by_seed, multi="multi_step", single=SINGLE_LAST)
        single_finals = _load_finals(run_dirs, f"final_{SINGLE_LAST}_configured.pkl")
        single_agreement = cross_seed_agreement(single_finals, topk)
        comparison["single_seed_spearman"] = [r["spearman_median"] for r in single_agreement]
        comparison["multi_seed_spearman"] = [r["spearman_median"] for r in agreement]
        pd.DataFrame(spec_rows).to_csv(out_dir / "mode_specificity.csv", index=False)
        pd.DataFrame(single_agreement).to_csv(
            out_dir / f"seed_agreement_{SINGLE_LAST}.csv", index=False
        )
        (out_dir / "mode_comparison.json").write_text(
            json.dumps(comparison, indent=2, default=str) + "\n"
        )
        for r in spec_rows:
            print(
                f"  seed {r['seed']} {r['mode']:<12} gain={r['configured_gain_mean']:.4f} "
                f"random={r['random_gain_mean']:.4f} specific={r['specific_gain_mean']:+.4f} "
                f"[{r['specific_gain_ci_low']:+.4f}, {r['specific_gain_ci_high']:+.4f}] "
                f"fraction={r['specific_fraction']:.3f}",
                flush=True,
            )
        print(json.dumps({k: comparison[k] for k in ("D1_multi", "D1_single", "D2", "primary")},
                         indent=2), flush=True)
    return verdict


SINGLE_LAST = "single_last"


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


def _mode_specific_gain(run_dir: Path, seed_dir: Path, prefix: str, *, seed: int):
    conf = _final_cells(seed_dir / f"{prefix}{CONFIGURED}")
    conf_ref = _final_cells(run_dir / "ordered_rank_edit" / CONFIGURED)
    if conf is None or conf_ref is None:
        return None
    randoms, refs = [], []
    for chain_dir in sorted(seed_dir.glob(f"{prefix}random*")):
        chain = chain_dir.name.removeprefix(prefix)
        r = _final_cells(chain_dir)
        rr = _final_cells(run_dir / "ordered_rank_edit" / chain)
        if r is not None and rr is not None:
            randoms.append(r.loc[conf.index].to_numpy())
            refs.append(rr.loc[conf.index].to_numpy())
    if not randoms:
        return None
    out = specific_gain(
        conf.to_numpy(), conf_ref.loc[conf.index].to_numpy(), randoms, refs, seed=seed
    )
    out["n_random_chains"] = float(len(randoms))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
