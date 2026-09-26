#!/usr/bin/env python3
"""State-feedback ISP Phase 1-2 — rerank baselines and the Delta-rank decoder.

Generic: the perturbation is whatever ``sequential.steps`` lists (any genes,
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
``run_sequential_isp.py``; compare within this run, not against paper tables.

Usage:
  python3 core/run_state_feedback_isp.py --config core/config/state_feedback_isp.yaml
  python3 core/run_state_feedback_isp.py --config ... --max-ncells 50 \
      --conditions ordered_rank_edit linear_deltarank oracle
"""
from __future__ import annotations

import argparse
import json
import pickle
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
from sequential_oe import parse_sequential_steps
import run_sequential_isp as seq

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
from state_feedback.metrics import gap_closed_fraction, spearman_values
from state_feedback.oracle_rerank import build_pseudobulk_rank_priority
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
    return seq.compute_cell_mean_goal_state_shifts(
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
):
    """Apply every step's perturbation cumulatively (end state of the chain)."""
    working = dataset
    workers = seq._gpu_resident_map_workers(nproc)
    for step, tokens in zip(steps, token_by_step):
        working = working.map(
            seq._apply_typed_step,
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
) -> list[dict[str, Any]]:
    """Run one condition's perturbation chain, optionally with state feedback."""
    workers = seq._gpu_resident_map_workers(nproc)
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
            seq._apply_typed_step,
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
        seq._empty_cuda_cache()

        do_feedback = (
            rerank_fn is not None
            and step_idx >= int(feedback_after_step)
            and step_idx < n_steps
            and (feedback_every_step or step_idx == int(feedback_after_step))
        )
        if not do_feedback:
            continue

        ctrl_ds = start_ds if ctrl_reference == "start" else pre_step
        new_ids, diag = rerank_fn(ctrl_ds, working)
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
            }
        )
        print(
            f"  [{condition}] feedback after step{step_idx}: "
            f"median={stats_fb['median']:.6f} "
            f"moved={diag.get('rerank_frac_moved', float('nan')):.3f} "
            f"rho={diag.get('rerank_spearman_before_after', float('nan')):.4f}",
            flush=True,
        )
        seq._empty_cuda_cache()

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
        start_ds, new_ids, num_proc=seq._gpu_resident_map_workers(nproc)
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
    val_fraction = float(dec_cfg.get("val_fraction", 0.2))
    train_tokens, val_tokens = split_tokens(
        teacher.keys(), val_fraction=val_fraction, seed=seed
    )
    print(
        f"Teacher tokens: {len(teacher)} (train={len(train_tokens)} val={len(val_tokens)})",
        flush=True,
    )

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
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    paths = cfg.get("paths") or {}
    pert = cfg.get("perturbation") or {}
    mdl = cfg.get("model") or {}
    isp_cfg = cfg.get("isp") or {}
    runtime = cfg.get("runtime") or {}
    seq_cfg = cfg.get("sequential") or {}
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

    log_species_banner(species_from_config(cfg))

    steps = parse_sequential_steps(seq_cfg)
    if not steps:
        raise ValueError(
            "state-feedback ISP needs explicit sequential.steps (genes + type per step)"
        )

    from datasets import load_from_disk

    print(f"Loading dataset: {dataset_path}", flush=True)
    dataset = load_from_disk(str(dataset_path))
    start_ds = seq._select_start_cells(dataset, state_key, start_state, max_ncells)
    print(f"Start cells ({start_state}): n={len(start_ds)}", flush=True)

    obs_cap = int(sf_cfg.get("observed_max_ncells") or max_ncells or 3000)
    obs_ds = seq._select_start_cells(dataset, state_key, observed_state, obs_cap)
    print(f"Observed-state cells ({observed_state}): n={len(obs_ds)}", flush=True)
    teacher_ctrl_ds = seq._select_start_cells(dataset, state_key, start_state, obs_cap)
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

    token_by_step: list[list[int]] = []
    for step in steps:
        tokens, resolved = seq._resolve_gene_tokens(cfg, step["genes"])
        token_by_step.append(list(tokens))
        print(
            f"Step {step['index']} {step['type']} {step['name']}: "
            f"{len(tokens)} token(s) ({', '.join(resolved)})",
            flush=True,
        )

    output_root.mkdir(parents=True, exist_ok=True)

    decoder = None
    decoder_info: dict[str, Any] = {}
    needs_decoder = any(c in {"linear_deltarank", "null_feedback"} for c in conditions)
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
                layer_to_quant=layer_to_quant,
                pad_token_id=pad_token_id,
                model_input_size=model_input_size,
                forward_batch_size=forward_batch_size,
                nproc=nproc,
                out_dir=output_root / "decoder",
            )
        seq._empty_cuda_cache()

    mlm_model = None
    if "delta_mlm" in conditions:
        print(f"Loading pretrained MLM head: {backend.pretrained_model}", flush=True)
        mlm_model = gs.load_mlm_model(backend.pretrained_model)

    common = dict(
        layer_to_quant=layer_to_quant,
        pad_token_id=pad_token_id,
        model_input_size=model_input_size,
        forward_batch_size=forward_batch_size,
        nproc=nproc,
    )
    all_rows: list[dict[str, Any]] = []

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
                **common,
            )
        )

    summary = pd.DataFrame(all_rows)
    summary_path = output_root / "phase12_summary.csv"
    summary.to_csv(summary_path, index=False)

    def _final_median(condition: str) -> float:
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
        "ctrl_reference": ctrl_reference,
        "hysteresis": hysteresis,
        "decoder": decoder_info,
        "gate": gate_rows,
        "hard_gate": (
            "Phase 2 gate: inspect gap_closed_fraction for linear_deltarank against "
            "the oracle ceiling, plus decoder spearman_val and null drift. Phase 3 "
            "(MLP / multi-step) only if linear clearly saturates below the ceiling."
        ),
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {gate_path}", flush=True)
    print(json.dumps({"gate": gate_rows, "decoder": decoder_info.get("selected", {})}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
