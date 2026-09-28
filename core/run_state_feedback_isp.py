#!/usr/bin/env python3
"""State-feedback ISP Phase 1-2 — rerank baselines and the Delta-rank decoder.

Generic: the perturbation is whatever ``state_feedback.steps`` lists (any genes,
overexpress or knockdown), the states are whatever ``perturbation.state_key`` /
``start_state`` / ``end_state`` name in the user's fine-tuned dataset. Nothing
here is specific to reprogramming factors.

Conditions compared in one run:

| Condition | What it does |
|-----------|--------------|
| ``ordered_rank_edit`` | paper baseline: OE/KD chain, no feedback |
| ``norm`` | Phase 1 null baseline: rerank by hidden-state norm change |
| ``delta_mlm`` | Phase 1 comparator: pretrained MLM self-logit change + inertia |
| ``linear_deltarank`` | Phase 2 primary: trained residual Delta-rank decoder |
| ``oracle`` | ceiling: observed ``observed_state`` ranks |
| ``null_feedback`` | guardrail: decoder feedback with zero perturbation |

All conditions are scored with the **cell-mean cosine** goal-state shift
(``strip_leading=0``) so a permuted encoding stays comparable with an
unpermuted one. That metric differs from the group-rank-aligned score used by
``run_ordered_rank_edit_isp.py``; compare within this run, not against paper tables.

Usage:
  python3 core/run_state_feedback_isp.py --config core/config/state_feedback_isp.yaml
  python3 core/run_state_feedback_isp.py --config ... --max-ncells 50 \
      --conditions ordered_rank_edit linear_deltarank oracle
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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

from state_feedback import controls
from state_feedback import feedback as fb
from state_feedback import gene_states as gs
from state_feedback.decoder import (
    load_decoder,
    null_drift,
    predict_delta_rank,
    save_decoder,
    token_mask,
    train_delta_rank_decoder,
)
from state_feedback.evaluate import compare_methods, direction_fidelity_verdict
from state_feedback.metrics import gap_closed_fraction, spearman_values
from state_feedback.multistep import FeedbackGuard, MultiStepConfig
from state_feedback.oracle_rerank import build_pseudobulk_rank_priority
from state_feedback.samples import collect_eval_samples
from state_feedback.teacher import observed_delta_rank, split_tokens

ALL_CONDITIONS = (
    "ordered_rank_edit",
    "norm",
    "delta_mlm",
    "linear_deltarank",
    "oracle",
    "null_feedback",
)
SHIFT_METRIC = "cell_mean_cosine"


def _summarize(df: pd.DataFrame) -> dict[str, float]:
    s = pd.to_numeric(df["Shift_to_goal_end"], errors="coerce").dropna()
    if s.empty:
        return {
            "n": 0.0,
            "median": float("nan"),
            "mean": float("nan"),
            "frac_positive": float("nan"),
        }
    return {
        "n": float(len(s)),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "frac_positive": float((s > 0).mean()),
    }


def _score_cell_mean(
    model,
    start_ds,
    working,
    goal_state: str,
    state_embs: Mapping[str, torch.Tensor],
    *,
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    nproc: int,
    batch_state: list[int],
) -> pd.DataFrame:
    """Goal-state shift that does not assume perturbed genes sit at the front."""
    return ore.compute_cell_mean_goal_state_shifts(
        model,
        start_ds,
        working,
        goal_state,
        dict(state_embs),
        layer_to_quant,
        pad_token_id,
        model_input_size,
        batch_state[0],
        nproc,
        strip_leading=0,
        batch_state=batch_state,
    )


def _apply_steps_at_once(
    dataset,
    steps: Sequence[Mapping[str, Any]],
    token_by_step: Sequence[Sequence[int]],
    *,
    nproc: int,
    n_steps: int | None = None,
):
    """Apply the first ``n_steps`` perturbations cumulatively (default: all)."""
    working = dataset
    workers = ore._gpu_resident_map_workers(nproc)
    limit = len(steps) if n_steps is None else max(0, int(n_steps))
    for step, tokens in list(zip(steps, token_by_step))[:limit]:
        working = working.map(
            ore._apply_typed_step,
            fn_kwargs={"tokens": list(tokens), "perturb_type": str(step["type"])},
            num_proc=workers,
        )
    return working


def run_condition(
    condition: str,
    model,
    start_ds,
    steps: Sequence[Mapping[str, Any]],
    token_by_step: Sequence[Sequence[int]],
    goal_state: str,
    state_embs: Mapping[str, torch.Tensor],
    out_dir: Path,
    *,
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    nproc: int,
    rerank_fn: Callable[[Any, Any], fb.RerankResult] | None = None,
    feedback_after_step: int = 1,
    feedback_every_step: bool = False,
    ctrl_reference: str = "start",
    multi_step: MultiStepConfig | None = None,
    guard_summaries: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run one condition's perturbation chain, optionally with state feedback.

    With ``feedback_every_step`` the chain runs under a ``FeedbackGuard`` (cycle halt,
    convergence stop, event cap); per-cell halts go to ``feedback_guard.csv``.
    """
    workers = ore._gpu_resident_map_workers(nproc)
    guard = (
        FeedbackGuard(len(start_ds), multi_step)
        if rerank_fn is not None and feedback_every_step
        else None
    )
    cap_logged = False
    batch_state = [int(forward_batch_size)]
    working = start_ds
    try:
        working.reset_format()
    except Exception:
        pass
    rows: list[dict[str, Any]] = []
    n_steps = len(steps)

    for step_idx, (step, tokens) in enumerate(zip(steps, token_by_step), start=1):
        name = str(step["name"])
        ptype = str(step["type"])
        pre_step = working
        working = working.map(
            ore._apply_typed_step,
            fn_kwargs={"tokens": list(tokens), "perturb_type": ptype},
            num_proc=workers,
        )
        df = _score_cell_mean(
            model,
            start_ds,
            working,
            goal_state,
            state_embs,
            layer_to_quant=layer_to_quant,
            pad_token_id=pad_token_id,
            model_input_size=model_input_size,
            nproc=nproc,
            batch_state=batch_state,
        )
        step_dir = out_dir / f"step{step_idx:02d}_{name}"
        step_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(step_dir / "per_cell_shifts.csv", index=False)
        stats = _summarize(df)
        rows.append(
            {
                "condition": condition,
                "step": step_idx,
                "step_name": name,
                "perturb_type": ptype,
                "event": f"{ptype}_{name}",
                "shift_metric": SHIFT_METRIC,
                **stats,
            }
        )
        print(
            f"  [{condition}] step{step_idx} {ptype} {name}: "
            f"median={stats['median']:.6f} n={int(stats['n'])}",
            flush=True,
        )
        ore._empty_cuda_cache()

        do_feedback = (
            rerank_fn is not None
            and step_idx >= int(feedback_after_step)
            and step_idx < n_steps
            and (feedback_every_step or step_idx == int(feedback_after_step))
        )
        if not do_feedback:
            continue
        if guard is not None and guard.exhausted:
            if guard.cap_reached and not cap_logged:
                n_capped = guard.mark_cap(step_idx)
                cap_logged = True
                print(
                    f"  [{condition}] feedback cap ({guard.cfg.max_feedback_events} events) "
                    f"reached before step{step_idx}; {n_capped} cells stop here",
                    flush=True,
                )
            continue

        ctrl_ds = start_ds if ctrl_reference == "start" else pre_step
        new_ids, diag = rerank_fn(ctrl_ds, working)
        guard_stats: dict[str, float] = {}
        if guard is not None:
            before_ids = gs.raw_input_ids(working, 0, len(working), model_input_size)
            new_ids, guard_stats = guard.apply(step_idx, before_ids, new_ids)
            diag = fb.rerank_diagnostics(before_ids, new_ids)
        working = fb.replace_input_ids(working, new_ids, num_proc=workers)
        df_fb = _score_cell_mean(
            model,
            start_ds,
            working,
            goal_state,
            state_embs,
            layer_to_quant=layer_to_quant,
            pad_token_id=pad_token_id,
            model_input_size=model_input_size,
            nproc=nproc,
            batch_state=batch_state,
        )
        fb_dir = out_dir / f"step{step_idx:02d}_feedback"
        fb_dir.mkdir(parents=True, exist_ok=True)
        df_fb.to_csv(fb_dir / "per_cell_shifts.csv", index=False)
        stats_fb = _summarize(df_fb)
        rows.append(
            {
                "condition": condition,
                "step": step_idx,
                "step_name": "feedback",
                "perturb_type": "rerank",
                "event": "state_feedback",
                "shift_metric": SHIFT_METRIC,
                **stats_fb,
                **diag,
                **guard_stats,
            }
        )
        guard_note = (
            f" active={int(guard_stats['guard_active_after'])}/{guard.n_cells}"
            f" cycle+={int(guard_stats['guard_new_cycle_halts'])}"
            f" converged+={int(guard_stats['guard_new_converged'])}"
            if guard is not None
            else ""
        )
        print(
            f"  [{condition}] feedback after step{step_idx}: "
            f"median={stats_fb['median']:.6f} "
            f"moved={diag.get('rerank_frac_moved', float('nan')):.3f} "
            f"rho={diag.get('rerank_spearman_before_after', float('nan')):.4f}"
            f"{guard_note}",
            flush=True,
        )
        ore._empty_cuda_cache()

    if guard is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(guard.cell_rows()).to_csv(out_dir / "feedback_guard.csv", index=False)
        summary = guard.summary()
        print(f"  [{condition}] feedback guard: {summary}", flush=True)
        if guard_summaries is not None:
            guard_summaries[condition] = summary

    return rows


def run_null_feedback(
    model,
    decoder,
    start_ds,
    goal_state: str,
    state_embs: Mapping[str, torch.Tensor],
    out_dir: Path,
    *,
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    nproc: int,
    hysteresis: float,
) -> list[dict[str, Any]]:
    """Guardrail: decoder feedback with no perturbation must not move genes."""
    batch_state = [int(forward_batch_size)]
    new_ids, diag = fb.rerank_linear_deltarank(
        model,
        decoder,
        start_ds,
        start_ds,
        layer_to_quant=layer_to_quant,
        pad_token_id=pad_token_id,
        model_input_size=model_input_size,
        forward_batch_size=forward_batch_size,
        hysteresis=hysteresis,
    )
    working = fb.replace_input_ids(
        start_ds, new_ids, num_proc=ore._gpu_resident_map_workers(nproc)
    )
    df = _score_cell_mean(
        model,
        start_ds,
        working,
        goal_state,
        state_embs,
        layer_to_quant=layer_to_quant,
        pad_token_id=pad_token_id,
        model_input_size=model_input_size,
        nproc=nproc,
        batch_state=batch_state,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "per_cell_shifts.csv", index=False)
    stats = _summarize(df)
    print(
        f"  [null_feedback] median={stats['median']:.6f} "
        f"moved={diag.get('rerank_frac_moved', float('nan')):.4f}",
        flush=True,
    )
    return [
        {
            "condition": "null_feedback",
            "step": 0,
            "step_name": "null",
            "perturb_type": "rerank",
            "event": "null_feedback",
            "shift_metric": SHIFT_METRIC,
            **stats,
            **diag,
        }
    ]


def _build_decoder(
    model,
    start_ds,
    steps,
    token_by_step,
    teacher: Mapping[int, float],
    dec_cfg: Mapping[str, Any],
    train_tokens: set[int],
    val_tokens: set[int],
    *,
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    nproc: int,
    out_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    """Collect training samples, grid-search ``max_shift``, return best decoder."""
    seed = int(dec_cfg.get("seed", 0))

    train_pert = _apply_steps_at_once(start_ds, steps, token_by_step, nproc=nproc)
    data = fb.collect_training_samples(
        model,
        start_ds,
        train_pert,
        teacher,
        layer_to_quant=layer_to_quant,
        pad_token_id=pad_token_id,
        model_input_size=model_input_size,
        forward_batch_size=forward_batch_size,
        keep_tokens=None,
        max_genes_per_cell=int(dec_cfg.get("max_genes_per_cell", 256)),
        max_cells=dec_cfg.get("train_max_ncells"),
        seed=seed,
    )
    print(f"Decoder training samples: {len(data)}", flush=True)
    if len(data) == 0:
        raise RuntimeError(
            "No decoder training samples. Check that observed_state cells exist and "
            "that teacher tokens overlap the start-state encodings."
        )

    tr_mask = token_mask(data.tokens, train_tokens)
    va_mask = token_mask(data.tokens, val_tokens)
    train_set = data.subset(tr_mask)
    val_set = data.subset(va_mask)
    print(f"  train pairs={len(train_set)} val pairs={len(val_set)}", flush=True)

    grid = dec_cfg.get("max_shift_grid") or [dec_cfg.get("max_shift", 0.1)]
    results: list[dict[str, Any]] = []
    best = None
    for max_shift in grid:
        decoder, hist = train_delta_rank_decoder(
            train_set,
            d_model=gs.hidden_size(model),
            max_shift=float(max_shift),
            epochs=int(dec_cfg.get("epochs", 20)),
            lr=float(dec_cfg.get("lr", 1e-3)),
            batch_size=int(dec_cfg.get("batch_size", 4096)),
            lam_huber=float(dec_cfg.get("lam_huber", 1.0)),
            lam_pair=float(dec_cfg.get("lam_pair", 0.5)),
            lam_identity=float(dec_cfg.get("lam_identity", 1.0)),
            lam_smooth=float(dec_cfg.get("lam_smooth", 0.01)),
            seed=seed,
        )
        metrics = {"max_shift": float(max_shift), "final_loss": hist["final_loss"]}
        for label, subset in (("train", train_set), ("val", val_set)):
            if len(subset) >= 2:
                pred = predict_delta_rank(decoder, subset.delta_h, subset.base_rank)
                metrics[f"spearman_{label}"] = spearman_values(
                    pred.tolist(), subset.target.tolist()
                )
            else:
                metrics[f"spearman_{label}"] = float("nan")
        metrics.update(null_drift(decoder))
        results.append(metrics)
        print(
            f"  max_shift={max_shift}: spearman_val={metrics['spearman_val']:.4f} "
            f"spearman_train={metrics['spearman_train']:.4f} "
            f"null_drift={metrics['null_drift_mean_abs']:.2e}",
            flush=True,
        )
        score = metrics["spearman_val"]
        if best is None or (score == score and score > best[0]):
            best = (score if score == score else float("-inf"), decoder, metrics)

    assert best is not None
    _, decoder, best_metrics = best
    out_dir.mkdir(parents=True, exist_ok=True)
    save_decoder(decoder, out_dir / "delta_rank_decoder.pt")
    info = {
        "n_teacher_tokens": len(teacher),
        "n_train_tokens": len(train_tokens),
        "n_val_tokens": len(val_tokens),
        "n_train_pairs": len(train_set),
        "n_val_pairs": len(val_set),
        "grid": results,
        "selected": best_metrics,
    }
    (out_dir / "decoder_metrics.json").write_text(json.dumps(info, indent=2) + "\n")
    return decoder, info


def run_direction_fidelity(
    model,
    decoder,
    start_ds,
    steps,
    token_by_step,
    teacher: Mapping[int, float],
    val_tokens: set[int],
    eval_cfg: Mapping[str, Any],
    out_dir: Path,
    *,
    mlm_model=None,
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    nproc: int,
    alpha: float,
    max_shift: float,
    feedback_after_step: int,
) -> dict[str, Any]:
    """Score every method against the observed Δrank on the held-out gene split.

    This is the direction-sensitive counterpart to the endpoint shift: it asks
    whether a method recovers the *direction* of observed rank displacement, which
    the classifier-space shift cannot distinguish from mere encoding disturbance.
    """
    which = str(eval_cfg.get("perturbation", "full_chain"))
    if which == "feedback_point":
        n_steps = int(feedback_after_step)
    elif which == "full_chain":
        n_steps = None
    else:
        raise ValueError("state_feedback.eval.perturbation must be full_chain or feedback_point")
    pert_ds = _apply_steps_at_once(
        start_ds, steps, token_by_step, nproc=nproc, n_steps=n_steps
    )

    seed = int(eval_cfg.get("seed", 0))
    samples = collect_eval_samples(
        model,
        start_ds,
        pert_ds,
        teacher,
        layer_to_quant=layer_to_quant,
        pad_token_id=pad_token_id,
        model_input_size=model_input_size,
        forward_batch_size=forward_batch_size,
        mlm_model=mlm_model,
        keep_tokens=val_tokens,
        max_genes_per_cell=int(eval_cfg.get("max_genes_per_cell", 512)),
        rows=None,
        seed=seed,
    )
    print(
        f"Direction-fidelity samples: {len(samples)} "
        f"(held-out genes only, perturbation={which})",
        flush=True,
    )
    if len(samples) == 0:
        raise RuntimeError("No held-out-gene samples collected for direction fidelity")

    result = compare_methods(
        samples,
        alpha=alpha,
        max_shift=max_shift,
        decoder=decoder,
        topk=[int(k) for k in (eval_cfg.get("topk") or [50, 100, 500])],
        n_boot=int(eval_cfg.get("n_boot", 1000)),
        n_perm=int(eval_cfg.get("n_perm", 10000)),
        seed=seed,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    calib_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    for row in result["methods"]:
        for b in row.pop("_calibration_bins", []):
            calib_rows.append({"method": row["method"], **b})
        for rho in row.pop("_cellwise_spearman", []):
            cell_rows.append({"method": row["method"], "cellwise_spearman": rho})
        method_rows.append(row)

    pd.DataFrame(method_rows).to_csv(out_dir / "direction_fidelity.csv", index=False)
    pd.DataFrame(result["contrasts"]).to_csv(out_dir / "contrasts.csv", index=False)
    pd.DataFrame(calib_rows).to_csv(out_dir / "calibration_bins.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(out_dir / "cellwise_spearman.csv", index=False)

    base_control = result.get("base_rank_control") or {}
    if base_control:
        pd.DataFrame(
            [{"method": m, **{k: v for k, v in r.items() if k != "stratified_by_base"},
              **{f"stratum_{i}_spearman": x for i, x in enumerate(r["stratified_by_base"])}}
             for m, r in base_control["methods"].items()]
        ).to_csv(out_dir / "base_rank_control.csv", index=False)

    verdict = direction_fidelity_verdict(
        {"methods": method_rows, "contrasts": result["contrasts"],
         "base_rank_control": base_control}
    )
    summary = {
        "perturbation": which,
        "n_samples": len(samples),
        "n_held_out_genes_available": len(val_tokens),
        "skipped_methods": result["skipped"],
        "methods": method_rows,
        "contrasts": result["contrasts"],
        "base_rank_control": base_control,
        "verdict": verdict,
        "not_computable_here": {
            "within_gene_across_cells_spearman":
                "undefined: the pseudobulk teacher is one constant per gene",
            "perturbation_wise":
                "see perturbation_specificity/ (state_feedback.specificity) when enabled",
            "donor_batch_stratified":
                "dataset has a single replicate and sample_id is collinear with state_key",
            "matched_perturbation_oracle":
                "no observed intermediate post-perturbation time point in this dataset",
        },
    }
    (out_dir / "direction_fidelity.json").write_text(json.dumps(summary, indent=2) + "\n")

    for row in method_rows:
        print(
            f"  {row['method']:<18} pooled_rho={row['pooled_spearman']:+.4f} "
            f"partial_rho={row.get('partial_spearman_given_base', float('nan')):+.4f} "
            f"cellwise_median={row['cellwise_spearman_median']:+.4f} "
            f"gene_rho={row['gene_aggregated_spearman']:+.4f} "
            f"P@100_up={row.get('precision_at_100_up', float('nan')):.3f}",
            flush=True,
        )
    nan = float("nan")
    for c in result["contrasts"]:
        print(
            f"  Δρ vs {c['baseline']:<12} = {c.get('boot_delta_rho', nan):+.4f} "
            f"[{c.get('boot_ci_low', nan):+.4f}, {c.get('boot_ci_high', nan):+.4f}] "
            f"excludes0={c.get('boot_excludes_zero', nan)} "
            f"p={c.get('perm_p_value', nan):.4g}",
            flush=True,
        )
    if base_control:
        lp = base_control.get("linear_partial", {})
        print(
            f"  base-rank control: base-only rho={base_control['base_only_pooled_spearman']:+.4f} "
            f"linear partial rho={lp.get('partial', nan):+.4f} "
            f"[{lp.get('ci_low', nan):+.4f}, {lp.get('ci_high', nan):+.4f}]",
            flush=True,
        )
    print(f"Wrote {out_dir / 'direction_fidelity.csv'}", flush=True)
    return summary


def run_perturbation_specificity(
    cfg: Mapping[str, Any],
    spec_cfg: Mapping[str, Any],
    model,
    decoder,
    start_ds,
    steps,
    token_by_step,
    teacher: Mapping[int, float],
    val_tokens: set[int],
    eval_cfg: Mapping[str, Any],
    out_dir: Path,
    *,
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    nproc: int,
) -> dict[str, Any]:
    """Feed the same decoder Δh from other perturbations; does its signal beyond
    base rank depend on the perturbation?

    Conditions: ``configured`` (the config's steps, cumulative), each named set in
    ``specificity.sets``, and ``n_random`` random draws of ``random_size`` teacher
    genes perturbed with ``random_type``. ``random_match_detection`` restricts the
    random pool to genes detected in 0.5-2x as many start cells as that gene (a
    delete of an undetected gene changes nothing). Every perturbed gene of every
    condition is removed from the scored set, so all conditions share one gene set.
    """
    seed = int(eval_cfg.get("seed", 0))
    n_boot = int(spec_cfg.get("n_boot", eval_cfg.get("n_boot", 1000)))
    max_genes = int(eval_cfg.get("max_genes_per_cell", 512))

    plans: dict[str, tuple[list[dict[str, Any]], list[list[int]], list[str]]] = {
        "configured": (
            list(steps),
            [list(t) for t in token_by_step],
            [f"{s['type']}:{g}" for s in steps for g in s["genes"]],
        ),
    }
    named_tokens: set[int] = {t for ts in token_by_step for t in ts}
    for name, spec in (spec_cfg.get("sets") or {}).items():
        genes = [str(g) for g in spec["genes"]]
        ptype = str(spec.get("type", "overexpress"))
        tokens, resolved = ore._resolve_gene_tokens(cfg, genes)
        if len(tokens) != len(genes):
            raise ValueError(f"specificity set {name!r}: resolved {resolved} for {genes}")
        named_tokens.update(tokens)
        plans[str(name)] = (
            [{"index": 1, "name": str(name), "type": ptype, "genes": genes}],
            [list(tokens)],
            [f"{ptype}:{g}" for g in genes],
        )

    n_random = int(spec_cfg.get("n_random", 0))
    random_size = int(spec_cfg.get("random_size", 1))
    random_type = str(spec_cfg.get("random_type", "overexpress"))
    pool = sorted(set(int(t) for t in teacher) - named_tokens)
    match_info: dict[str, Any] | None = None
    match_gene = spec_cfg.get("random_match_detection")
    if match_gene:
        ref_token = ore._resolve_gene_tokens(cfg, [str(match_gene)])[0][0]
        detection = controls.detection_rate(start_ds["input_ids"])
        ref = detection.get(int(ref_token), 0.0)
        pool = controls.detection_matched_pool(pool, detection, ref)
        match_info = {"gene": str(match_gene), "detection_rate": ref, "pool_size": len(pool)}
        print(f"Random pool matched to {match_gene} (detection {ref:.3f}): {len(pool)} genes",
              flush=True)
    if n_random and len(pool) < random_size:
        raise ValueError(f"random pool has {len(pool)} genes; need {random_size}")
    rng = random.Random(seed)
    for r in range(n_random):
        tokens = rng.sample(pool, random_size)
        plans[f"random{r}"] = (
            [{"index": 1, "name": f"random{r}", "type": random_type,
              "genes": [str(t) for t in tokens]}],
            [tokens],
            [f"{random_type}:token:{t}" for t in tokens],
        )

    excluded = {t for _, tbs, _ in plans.values() for ts in tbs for t in ts}
    keep = set(int(t) for t in val_tokens) - excluded
    print(f"Specificity: {len(plans)} conditions, {len(keep)} scored held-out genes",
          flush=True)

    rows: list[dict[str, Any]] = []
    kept_samples: dict[str, Any] = {}
    for name, (st, tbs, genes) in plans.items():
        pert = _apply_steps_at_once(start_ds, st, tbs, nproc=nproc)
        samples = collect_eval_samples(
            model, start_ds, pert, teacher,
            layer_to_quant=layer_to_quant, pad_token_id=pad_token_id,
            model_input_size=model_input_size, forward_batch_size=forward_batch_size,
            mlm_model=None, keep_tokens=keep, max_genes_per_cell=max_genes, seed=seed,
        )
        ore._empty_cuda_cache()
        rep = controls.base_rank_control(samples, decoder, n_boot=n_boot, seed=seed)
        lp = rep["linear_partial"]
        row = {
            "condition": name,
            "genes": " ".join(genes),
            "n_samples": rep["n_samples"],
            "mean_delta_h_norm": rep["mean_delta_h_norm"],
            "linear_pooled_spearman": rep["methods"]["linear_deltarank"]["pooled_spearman"],
            "linear_partial": lp["partial"],
            "linear_partial_ci_low": lp["ci_low"],
            "linear_partial_ci_high": lp["ci_high"],
            "delta_h_only_partial":
                rep["methods"]["delta_h_only"]["partial_spearman_given_base"],
            "delta_h_shuffled_partial":
                rep["methods"]["delta_h_shuffled"]["partial_spearman_given_base"],
            "base_only_pooled_spearman": rep["base_only_pooled_spearman"],
        }
        rows.append(row)
        print(
            f"  [{name}] samples={row['n_samples']} |dh|={row['mean_delta_h_norm']:.3f} "
            f"partial={row['linear_partial']:+.3f} "
            f"[{row['linear_partial_ci_low']:+.3f}, {row['linear_partial_ci_high']:+.3f}] "
            f"dh_only={row['delta_h_only_partial']:+.3f}",
            flush=True,
        )
        if not name.startswith("random"):
            kept_samples[name] = samples

    random_rows = [r for r in rows if r["condition"].startswith("random")]
    vs_random = [
        {"condition": r["condition"], "linear_partial": r["linear_partial"],
         "delta_h_only_partial": r["delta_h_only_partial"],
         **controls.versus_random(r["linear_partial"], [x["linear_partial"] for x in random_rows]),
         **{f"delta_h_only_{k}": v for k, v in controls.versus_random(
             r["delta_h_only_partial"], [x["delta_h_only_partial"] for x in random_rows]).items()
            if k in ("random_mean", "random_max", "n_random_ge", "empirical_p")}}
        for r in rows if not r["condition"].startswith("random")
    ] if random_rows else []

    contrasts: list[dict[str, Any]] = []
    names = list(kept_samples)
    linear = {n: controls.decoder_variants(decoder, s, seed=seed)["linear_deltarank"]
              for n, s in kept_samples.items()}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa, sb = kept_samples[a], kept_samples[b]
            ia = {k: j for j, k in enumerate(zip(sa.cell_index, sa.tokens))}
            ib = {k: j for j, k in enumerate(zip(sb.cell_index, sb.tokens))}
            common = sorted(set(ia) & set(ib))
            xa = [ia[k] for k in common]
            xb = [ib[k] for k in common]
            diff = controls.bootstrap_partial_diff(
                linear[a][xa], linear[b][xb],
                sa.target.cpu().numpy()[xa], sa.base_rank.cpu().numpy()[xa],
                [sa.cell_index[j] for j in xa], n_boot=n_boot, seed=seed,
            )
            contrasts.append({"a": a, "b": b, "n_common": len(common), **diff})

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "perturbation_specificity.csv", index=False)
    pd.DataFrame(vs_random).to_csv(out_dir / "vs_random.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out_dir / "specificity_contrasts.csv", index=False)
    summary = {
        "conditions": {n: p[2] for n, p in plans.items()},
        "n_scored_genes": len(keep),
        "n_start_cells": len(start_ds),
        "random": {"n": n_random, "size": random_size, "type": random_type,
                   "match": match_info},
        "rows": rows,
        "vs_random": vs_random,
        "contrasts": contrasts,
    }
    (out_dir / "perturbation_specificity.json").write_text(json.dumps(summary, indent=2) + "\n")
    for v in vs_random:
        print(
            f"  {v['condition']} vs {v['random_n']} random: partial={v['linear_partial']:+.3f} "
            f"random mean={v['random_mean']:+.3f} max={v['random_max']:+.3f} "
            f"p={v['empirical_p']:.3f}",
            flush=True,
        )
    for c in contrasts:
        print(f"  {c['a']} - {c['b']}: {c['diff']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]",
              flush=True)
    print(f"Wrote {out_dir / 'perturbation_specificity.csv'}", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-ncells", type=int, default=None)
    parser.add_argument("--forward-batch-size", default=None)
    parser.add_argument("--nproc", type=int, default=None)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument(
        "--decoder-checkpoint",
        type=Path,
        default=None,
        help="Reuse a trained decoder instead of fitting a new one.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Train/load the decoder and run direction-fidelity scoring only, "
        "skipping the endpoint-shift conditions.",
    )
    parser.add_argument(
        "--skip-direction-fidelity",
        action="store_true",
        help="Skip the direction-fidelity comparison.",
    )
    parser.add_argument(
        "--specificity",
        action="store_true",
        help="Run the perturbation-specificity control even if "
        "state_feedback.specificity.enabled is false.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    paths = cfg.get("paths") or {}
    pert = cfg.get("perturbation") or {}
    mdl = cfg.get("model") or {}
    isp_cfg = cfg.get("isp") or {}
    runtime = cfg.get("runtime") or {}
    sf_cfg = cfg.get("state_feedback") or {}
    dec_cfg = sf_cfg.get("decoder") or {}

    dataset_path = Path(paths["dataset"]).expanduser()
    model_path = Path(paths["geneformer_model"]).expanduser()
    output_root = Path(paths["output_root"]).expanduser()
    if paths.get("output_date_subdir", True):
        output_root = output_root / datetime.now(timezone.utc).strftime("%Y%m%d")
    if paths.get("output_time_subdir", True):
        stamp = datetime.now(timezone.utc).strftime("%H%M%S")
        output_root = output_root / f"state_feedback_isp_{stamp}"

    state_key = str(pert["state_key"])
    start_state = str(pert["start_state"])
    goal_state = str(pert["end_state"])
    observed_state = str(sf_cfg.get("observed_state") or goal_state)

    conditions = [
        c for c in (args.conditions or sf_cfg.get("conditions") or ALL_CONDITIONS)
    ]
    unknown = [c for c in conditions if c not in ALL_CONDITIONS]
    if unknown:
        raise ValueError(f"Unknown conditions {unknown}; valid: {list(ALL_CONDITIONS)}")

    max_ncells = args.max_ncells if args.max_ncells is not None else isp_cfg.get("max_ncells")
    emb_layer = int(isp_cfg.get("emb_layer", 0))
    nproc = args.nproc if args.nproc is not None else int(runtime.get("nproc", 1))
    hysteresis = float(sf_cfg.get("hysteresis", 0.0))
    alpha = float(sf_cfg.get("alpha", 1.0))
    baseline_max_shift = float(sf_cfg.get("baseline_max_shift", 0.1))
    feedback_after_step = int(sf_cfg.get("feedback_after_step", 1))
    feedback_every_step = bool(sf_cfg.get("feedback_every_step", False))
    ctrl_reference = str(sf_cfg.get("ctrl_reference", "start"))
    if ctrl_reference not in {"start", "previous"}:
        raise ValueError("state_feedback.ctrl_reference must be 'start' or 'previous'")
    multi_step = MultiStepConfig.from_mapping(sf_cfg.get("multi_step"))

    log_species_banner(species_from_config(cfg))

    # Configs from before the rename list the steps under `sequential:`.
    steps = parse_steps(sf_cfg) or parse_steps(config_block(cfg))
    if not steps:
        raise ValueError(
            "state-feedback ISP needs explicit state_feedback.steps (genes + type per step)"
        )

    from datasets import load_from_disk

    print(f"Loading dataset: {dataset_path}", flush=True)
    dataset = load_from_disk(str(dataset_path))
    start_ds = ore._select_start_cells(dataset, state_key, start_state, max_ncells)
    print(f"Start cells ({start_state}): n={len(start_ds)}", flush=True)

    obs_cap = int(sf_cfg.get("observed_max_ncells") or max_ncells or 3000)
    obs_ds = ore._select_start_cells(dataset, state_key, observed_state, obs_cap)
    print(f"Observed-state cells ({observed_state}): n={len(obs_ds)}", flush=True)
    teacher_ctrl_ds = ore._select_start_cells(dataset, state_key, start_state, obs_cap)
    teacher = observed_delta_rank(
        teacher_ctrl_ds["input_ids"],
        obs_ds["input_ids"],
        min_detection_count=int(sf_cfg.get("min_detection_count", 5)),
    )
    oracle_priority = build_pseudobulk_rank_priority(obs_ds["input_ids"])
    print(
        f"Teacher delta-rank tokens: {len(teacher)}; oracle priority tokens: "
        f"{len(oracle_priority)}",
        flush=True,
    )

    model = isp.load_model(
        mdl.get("type", "CellClassifier"), int(mdl.get("num_classes", 2)), str(model_path)
    )
    layer_to_quant = isp.quant_layers(model) + emb_layer
    model_input_size = isp.get_model_input_size(model)

    backend = backend_from_config(cfg)
    with open(backend.token_dictionary, "rb") as fh:
        token_dict = pickle.load(fh)
    pad_token_id = token_dict.get("<pad>")

    fbs_raw = args.forward_batch_size or runtime.get("forward_batch_size", "auto")
    species_default = default_isp_forward_batch_size(backend.max_input_size)
    forward_batch_size = ore.resolve_dual_forward_batch_size(
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
    centroid_ds = ore._centroid_dataset(dataset, state_key, centroid_states, max_ncells, nproc)
    state_embs = isp.get_cell_state_avg_embs(
        model,
        centroid_ds,
        cell_states,
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
        print(
            f"Step {step['index']} {step['type']} {step['name']}: "
            f"{len(tokens)} token(s) ({', '.join(resolved)})",
            flush=True,
        )

    output_root.mkdir(parents=True, exist_ok=True)

    eval_cfg = sf_cfg.get("eval") or {}
    run_eval = not args.skip_direction_fidelity
    if args.eval_only:
        conditions = []

    train_tokens, val_tokens = split_tokens(
        teacher.keys(),
        val_fraction=float(dec_cfg.get("val_fraction", 0.2)),
        seed=int(dec_cfg.get("seed", 0)),
    )
    print(
        f"Teacher tokens: {len(teacher)} (train={len(train_tokens)} val={len(val_tokens)})",
        flush=True,
    )

    decoder = None
    decoder_info: dict[str, Any] = {}
    needs_decoder = run_eval or any(
        c in {"linear_deltarank", "null_feedback"} for c in conditions
    )
    if needs_decoder:
        if args.decoder_checkpoint is not None:
            decoder = load_decoder(
                args.decoder_checkpoint,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            decoder_info = {"loaded_from": str(args.decoder_checkpoint)}
            print(f"Loaded decoder: {args.decoder_checkpoint}", flush=True)
        else:
            print("=== training Delta-rank decoder ===", flush=True)
            decoder, decoder_info = _build_decoder(
                model,
                start_ds,
                steps,
                token_by_step,
                teacher,
                dec_cfg,
                train_tokens,
                val_tokens,
                layer_to_quant=layer_to_quant,
                pad_token_id=pad_token_id,
                model_input_size=model_input_size,
                forward_batch_size=forward_batch_size,
                nproc=nproc,
                out_dir=output_root / "decoder",
            )
        ore._empty_cuda_cache()

    mlm_model = None
    if "delta_mlm" in conditions or run_eval:
        print(f"Loading pretrained MLM head: {backend.pretrained_model}", flush=True)
        mlm_model = gs.load_mlm_model(backend.pretrained_model)

    fidelity: dict[str, Any] = {}
    if run_eval:
        print("=== direction fidelity (held-out genes, same teacher) ===", flush=True)
        fidelity = run_direction_fidelity(
            model,
            decoder,
            start_ds,
            steps,
            token_by_step,
            teacher,
            val_tokens,
            eval_cfg,
            output_root / "direction_fidelity",
            mlm_model=mlm_model,
            layer_to_quant=layer_to_quant,
            pad_token_id=pad_token_id,
            model_input_size=model_input_size,
            forward_batch_size=forward_batch_size,
            nproc=nproc,
            alpha=alpha,
            max_shift=baseline_max_shift,
            feedback_after_step=feedback_after_step,
        )
        ore._empty_cuda_cache()

    spec_cfg = sf_cfg.get("specificity") or {}
    specificity: dict[str, Any] = {}
    if args.specificity or bool(spec_cfg.get("enabled", False)):
        if decoder is None:
            raise ValueError("perturbation specificity needs a decoder (train or --decoder-checkpoint)")
        print("=== perturbation specificity (same decoder, other perturbations) ===", flush=True)
        specificity = run_perturbation_specificity(
            cfg,
            spec_cfg,
            model,
            decoder,
            start_ds,
            steps,
            token_by_step,
            teacher,
            val_tokens,
            eval_cfg,
            output_root / "perturbation_specificity",
            layer_to_quant=layer_to_quant,
            pad_token_id=pad_token_id,
            model_input_size=model_input_size,
            forward_batch_size=forward_batch_size,
            nproc=nproc,
        )
        ore._empty_cuda_cache()

    common = dict(
        layer_to_quant=layer_to_quant,
        pad_token_id=pad_token_id,
        model_input_size=model_input_size,
        forward_batch_size=forward_batch_size,
        nproc=nproc,
    )
    all_rows: list[dict[str, Any]] = []
    guard_summaries: dict[str, Any] = {}

    for condition in conditions:
        print(f"=== {condition} ===", flush=True)
        if condition == "null_feedback":
            all_rows.extend(
                run_null_feedback(
                    model,
                    decoder,
                    start_ds,
                    goal_state,
                    state_embs,
                    output_root / condition,
                    hysteresis=hysteresis,
                    **common,
                )
            )
            continue

        rerank_fn: Callable[[Any, Any], fb.RerankResult] | None = None
        if condition == "norm":
            def rerank_fn(ctrl_ds, pert_ds):  # noqa: F811
                return fb.rerank_norm(
                    model,
                    ctrl_ds,
                    pert_ds,
                    layer_to_quant=layer_to_quant,
                    pad_token_id=pad_token_id,
                    model_input_size=model_input_size,
                    forward_batch_size=forward_batch_size,
                    alpha=alpha,
                    max_shift=baseline_max_shift,
                    hysteresis=hysteresis,
                )
        elif condition == "delta_mlm":
            def rerank_fn(ctrl_ds, pert_ds):  # noqa: F811
                return fb.rerank_delta_mlm(
                    mlm_model,
                    ctrl_ds,
                    pert_ds,
                    pad_token_id=pad_token_id,
                    model_input_size=model_input_size,
                    forward_batch_size=forward_batch_size,
                    alpha=alpha,
                    max_shift=baseline_max_shift,
                    hysteresis=hysteresis,
                )
        elif condition == "linear_deltarank":
            def rerank_fn(ctrl_ds, pert_ds):  # noqa: F811
                return fb.rerank_linear_deltarank(
                    model,
                    decoder,
                    ctrl_ds,
                    pert_ds,
                    layer_to_quant=layer_to_quant,
                    pad_token_id=pad_token_id,
                    model_input_size=model_input_size,
                    forward_batch_size=forward_batch_size,
                    hysteresis=hysteresis,
                )
        elif condition == "oracle":
            def rerank_fn(ctrl_ds, pert_ds):  # noqa: F811
                return fb.rerank_oracle(
                    pert_ds,
                    oracle_priority,
                    hysteresis=hysteresis,
                    model_input_size=model_input_size,
                )

        all_rows.extend(
            run_condition(
                condition,
                model,
                start_ds,
                steps,
                token_by_step,
                goal_state,
                state_embs,
                output_root / condition,
                rerank_fn=rerank_fn,
                feedback_after_step=feedback_after_step,
                feedback_every_step=feedback_every_step,
                ctrl_reference=ctrl_reference,
                multi_step=multi_step,
                guard_summaries=guard_summaries,
                **common,
            )
        )

    summary = pd.DataFrame(all_rows)
    summary_path = output_root / "phase12_summary.csv"
    summary.to_csv(summary_path, index=False)

    def _final_median(condition: str) -> float:
        if summary.empty:
            return float("nan")
        sub = summary[(summary["condition"] == condition) & (summary["step"] > 0)]
        if sub.empty:
            return float("nan")
        return float(sub.sort_values(["step", "step_name"]).iloc[-1]["median"])

    baseline = _final_median("ordered_rank_edit")
    ceiling = _final_median("oracle")
    gate_rows: list[dict[str, Any]] = []
    for condition in conditions:
        if condition in {"ordered_rank_edit", "oracle", "null_feedback"}:
            continue
        value = _final_median(condition)
        gate_rows.append(
            {
                "condition": condition,
                "shift_metric": SHIFT_METRIC,
                "ordered_rank_edit_final_median": baseline,
                "oracle_final_median": ceiling,
                "method_final_median": value,
                "gap_closed_fraction": gap_closed_fraction(baseline, value, ceiling),
                "beats_ordered_rank_edit": bool(
                    value == value and baseline == baseline and value > baseline
                ),
            }
        )
    gate = pd.DataFrame(gate_rows)
    gate_path = output_root / "phase12_gate.csv"
    gate.to_csv(gate_path, index=False)

    manifest = {
        "phase": "1-2",
        "mode": "state_feedback_isp",
        "config": str(args.config.resolve()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "model": str(model_path),
        "shift_metric": SHIFT_METRIC,
        "n_start_cells": len(start_ds),
        "state_key": state_key,
        "start_state": start_state,
        "goal_state": goal_state,
        "observed_state": observed_state,
        "steps": [
            {"name": s["name"], "type": s["type"], "genes": s["genes"]} for s in steps
        ],
        "conditions": conditions,
        "feedback_after_step": feedback_after_step,
        "feedback_every_step": feedback_every_step,
        "multi_step": (
            {"config": multi_step.to_dict(), "guard": guard_summaries}
            if feedback_every_step
            else None
        ),
        "ctrl_reference": ctrl_reference,
        "hysteresis": hysteresis,
        "decoder": decoder_info,
        "gate": gate_rows,
        "direction_fidelity": fidelity.get("verdict", {}),
        "perturbation_specificity": specificity.get("vs_random", []),
        "hard_gate": (
            "Phase 2 gate is two-axis. Primary axis is direction fidelity: the "
            "linear decoder must beat norm and delta_mlm on held-out-gene Spearman "
            "against the observed Delta-rank, with a bootstrap CI on the difference "
            "that excludes zero, and its partial Spearman given base rank must have "
            "a CI above zero (a base-rank-only predictor can match the pooled "
            "Spearman). gap_closed_fraction against the oracle is a "
            "feasibility ceiling only - the oracle feeds observed goal-state ranks "
            "back in, so it carries endpoint leakage and is not a fidelity target. "
            "Phase 3 (MLP / multi-step) only after the primary axis is settled."
        ),
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {gate_path}", flush=True)
    print(
        json.dumps(
            {
                "gate_endpoint_shift": gate_rows,
                "decoder": decoder_info.get("selected", {}),
                "direction_fidelity": fidelity.get("verdict", {}),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
