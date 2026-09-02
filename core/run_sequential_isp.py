#!/usr/bin/env python3
"""Sequential ISP — chain OE and/or KD on rank-value encodings.

Platform mode (``sequential.steps``): apply an ordered list of group OE / KD
steps on start-state cells, scoring ``goal_state_shift`` after each step.

OSKM research mode (no ``steps``): 24 factor-order permutations of Yamanaka OE,
plus a simultaneous OSKM4 baseline.

Usage:
  python3 core/run_sequential_isp.py --config core/config/sequential_isp.yaml
  python3 core/run_sequential_isp.py --config ... --orders O-S-K-M K-M-O-S --max-ncells 500
"""
from __future__ import annotations

import argparse
import json
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
    OSKM_FACTORS,
    OSKM_FACTOR_KEYS,
    PERTURB_DELETE,
    PERTURB_OVEREXPRESS,
    all_oskm_orders,
    apply_step,
    apply_single_step_overexpress,
    normalize_step_type,
    order_label,
    parse_sequential_steps,
    perturb_index_for_tokens,
    tokens_for_order,
)


def _centroid_dataset(dataset, state_key: str, states: Sequence[str], max_ncells: int | None, nproc: int):
    """Keep up to max_ncells per state so centroids do not embed the full 6000-cell matrix."""
    if not max_ncells:
        return dataset
    from datasets import concatenate_datasets

    parts = []
    nproc_f = _gpu_resident_map_workers(nproc)
    for st in states:
        sub = dataset.filter(lambda ex, s=st: ex[state_key] == s, num_proc=nproc_f)
        n = min(len(sub), int(max_ncells))
        if n == 0:
            raise ValueError(f"No cells with {state_key}={st!r} for state centroids")
        parts.append(sub.select(range(n)))
        print(f"Centroid cells ({st}): n={n}", flush=True)
    return concatenate_datasets(parts)


def _gpu_resident_map_workers(nproc: int) -> int:
    """Keep dataset.map single-process while the model occupies the GPU.

    Multiprocessing duplicates the Arrow table in host RAM. On unified-memory
    boxes (GB10 / Grace-Blackwell) that RAM is the same pool as VRAM, so a
    4-worker map during sequential scoring is a common OOM trigger.
    """
    if torch.cuda.is_available():
        return 1
    return max(1, min(int(nproc), 4))


def _is_cuda_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_sequential_forward_batch_size(
    value,
    model,
    input_data,
    pad_token_id,
    model_directory,
) -> int:
    """Auto-batch sized for group scoring (perturbed + original forwards)."""
    return isp.resolve_forward_batch_size(
        value,
        model,
        input_data,
        pad_token_id,
        model_directory,
        n_forwards=2,
        task="sequential_isp_group",
        label="sequential ISP forward_batch_size",
    )


def _resolve_gene_tokens(cfg: Mapping[str, Any], genes: Sequence[str]) -> tuple[list[int], list[str]]:
    from run_isp_umap import resolve_perturbation_tokens

    return resolve_perturbation_tokens(cfg, list(genes))


def _apply_step_example(example: dict[str, Any], tokens: Sequence[int]) -> dict[str, Any]:
    return apply_single_step_overexpress(example, tokens)


def _apply_typed_step(
    example: dict[str, Any],
    tokens: Sequence[int],
    perturb_type: str,
) -> dict[str, Any]:
    return apply_step(example, tokens, perturb_type)


def _app_path(raw: str) -> Path:
    p = raw.strip()
    if p.startswith("/app/"):
        return ROOT / p[len("/app/") :]
    if p.startswith("/app"):
        return ROOT / p[len("/app") :].lstrip("/")
    return ROOT / p.lstrip("/")


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _resolve_orders(spec: Sequence[str] | None) -> list[tuple[str, ...]]:
    if not spec:
        return all_oskm_orders()
    out: list[tuple[str, ...]] = []
    for item in spec:
        keys = tuple(k.strip() for k in item.split("-"))
        if len(keys) != 4 or set(keys) != set(OSKM_FACTOR_KEYS):
            raise ValueError(f"Invalid order {item!r}; want four distinct keys from {OSKM_FACTOR_KEYS}")
        out.append(keys)
    return out


def _resolve_factor_tokens(species_key: str, cfg: Mapping[str, Any]) -> dict[str, int]:
    import pickle

    backend = backend_from_config(cfg)
    with open(backend.token_dictionary, "rb") as fh:
        token_dict = pickle.load(fh)

    overrides = (cfg.get("sequential") or {}).get("factor_ensembl") or {}
    token_by_factor: dict[str, int] = {}
    for key in OSKM_FACTOR_KEYS:
        ens = overrides.get(key) or OSKM_FACTORS[key][species_key]
        if ens not in token_dict:
            raise KeyError(f"Factor {key} Ensembl {ens} not in token dictionary")
        token_by_factor[key] = int(token_dict[ens])
    return token_by_factor


def _select_start_cells(dataset, state_key: str, start_state: str, max_ncells: int | None):
    data = dataset.filter(lambda x: x.get(state_key) == start_state)
    data = data.sort("length", reverse=True)
    if max_ncells is not None and len(data) > max_ncells:
        data = data.select(range(max_ncells))
    return data


def _summarize_series(s: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return {
        "n": int(len(s)),
        "median": float(s.median()) if len(s) else float("nan"),
        "mean": float(s.mean()) if len(s) else float("nan"),
        "p25": float(s.quantile(0.25)) if len(s) else float("nan"),
        "p75": float(s.quantile(0.75)) if len(s) else float("nan"),
        "frac_positive": float((s > 0).mean()) if len(s) else float("nan"),
    }


def _step_csv_path(out_root: Path, order: tuple[str, ...], step_idx: int, factor_key: str) -> Path:
    tag = order_label(order)
    return out_root / "orders" / tag / f"step{step_idx:02d}_{factor_key}" / "single_gene_per_cell_shifts.csv"


def _order_complete(out_root: Path, order: tuple[str, ...]) -> bool:
    last = order[-1]
    return _step_csv_path(out_root, order, len(order), last).is_file()


def _load_step_rows_from_disk(out_root: Path, order: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tag = order_label(order)
    for step_idx, factor_key in enumerate(order, start=1):
        path = _step_csv_path(out_root, order, step_idx, factor_key)
        if not path.is_file():
            break
        s = pd.to_numeric(pd.read_csv(path)["Shift_to_goal_end"], errors="coerce").dropna()
        stats = _summarize_series(s)
        rows.append(
            {
                "order": tag,
                "step": step_idx,
                "factor": factor_key,
                "factors_applied": order_label(order[:step_idx]),
                "n": stats["n"],
                "median_shift": stats["median"],
                "mean_shift": stats["mean"],
                "p25": stats["p25"],
                "p75": stats["p75"],
                "frac_positive": stats["frac_positive"],
            }
        )
    return rows


def compute_goal_state_shifts(
    model,
    original_ds,
    perturbed_ds,
    oe_tokens: Sequence[int],
    goal_state: str,
    cell_states_to_model: dict[str, Any],
    state_embs_dict: dict[str, torch.Tensor],
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    nproc: int,
    *,
    perturb_type: str = PERTURB_OVEREXPRESS,
    batch_state: list[int] | None = None,
) -> pd.DataFrame:
    """Per-cell ``goal_state_shift`` via ``quant_cos_sims`` (group aligned).

    ``batch_state`` is a one-element list of the live forward batch size. On CUDA
    OOM it is halved and the scoring call is retried (sequential-specific: the
    dual original+perturbed forward can exceed a 1-forward auto cache).
    """
    ptype = normalize_step_type(perturb_type)
    orig_input_ids = [list(x) for x in original_ds["input_ids"]]
    indices_to_perturb = [perturb_index_for_tokens(ids, oe_tokens) for ids in orig_input_ids]
    try:
        original_ds.reset_format()
    except Exception:
        pass
    try:
        perturbed_ds.reset_format()
    except Exception:
        pass

    batch = int(forward_batch_size)
    map_workers = _gpu_resident_map_workers(nproc)
    while True:
        try:
            cos_dict = isp.quant_cos_sims(
                model,
                ptype,
                perturbed_ds,
                None,
                None,
                batch,
                layer_to_quant,
                original_ds,
                list(oe_tokens),
                indices_to_perturb,
                True,
                cell_states_to_model,
                state_embs_dict,
                pad_token_id,
                model_input_size,
                map_workers,
            )
            break
        except Exception as exc:
            if not _is_cuda_oom(exc) or batch <= 1:
                raise
            new_batch = max(1, batch // 2)
            print(
                f"CUDA OOM during sequential scoring (batch={batch}); "
                f"retrying at {new_batch}",
                flush=True,
            )
            _empty_cuda_cache()
            batch = new_batch
            if batch_state is not None:
                batch_state[0] = batch

    shifts = cos_dict[goal_state].numpy()

    meta = {
        c: original_ds[c]
        for c in original_ds.column_names
        if c not in {"input_ids", "length", "attention_mask"}
    }
    meta["cell_index"] = list(range(len(original_ds)))
    meta["Shift_to_goal_end"] = shifts
    return pd.DataFrame(meta)


def _write_shifts(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_dataset(ds, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        import shutil

        shutil.rmtree(path)
    ds.save_to_disk(str(path))


def run_simultaneous_baseline(
    model,
    start_ds,
    all_tokens: Sequence[int],
    goal_state: str,
    cell_states_to_model: dict[str, Any],
    state_embs_dict: dict[str, torch.Tensor],
    out_dir: Path,
    *,
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    save_dataset: bool,
    nproc: int,
) -> dict[str, float]:

    def _map(example):
        ex = dict(example)
        ex["tokens_to_perturb"] = list(all_tokens)
        ids = list(ex["input_ids"])
        present = [ids.index(t) for t in all_tokens if t in ids]
        ex["perturb_index"] = present if present else [-100]
        ex = isp.overexpress_tokens(ex)
        ex["length"] = len(ex["input_ids"])
        ex["attention_mask"] = [1] * ex["length"]
        return ex

    perturbed = start_ds.map(_map, num_proc=_gpu_resident_map_workers(nproc))
    df = compute_goal_state_shifts(
        model,
        start_ds,
        perturbed,
        all_tokens,
        goal_state,
        cell_states_to_model,
        state_embs_dict,
        layer_to_quant,
        pad_token_id,
        model_input_size,
        forward_batch_size,
        nproc,
    )
    _write_shifts(df, out_dir / "single_gene_per_cell_shifts.csv")
    if save_dataset:
        _save_dataset(perturbed, out_dir / "perturbed.dataset")
    return _summarize_series(df["Shift_to_goal_end"])


def run_order(
    model,
    start_ds,
    order: tuple[str, ...],
    token_by_factor: Mapping[str, int],
    goal_state: str,
    cell_states_to_model: dict[str, Any],
    state_embs_dict: dict[str, torch.Tensor],
    out_root: Path,
    *,
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    save_datasets: bool,
    nproc: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    tag = order_label(order)
    order_dir = out_root / "orders" / tag
    step_tokens = tokens_for_order(order, token_by_factor)
    step_summaries: list[dict[str, Any]] = []
    cumulative_tokens: list[int] = []

    working = start_ds
    try:
        working.reset_format()
    except Exception:
        pass
    batch_state = [int(forward_batch_size)]
    map_workers = _gpu_resident_map_workers(nproc)
    for step_idx, (factor_key, step_tok) in enumerate(zip(order, step_tokens), start=1):
        cumulative_tokens.extend(step_tok)
        tokens = list(step_tok)
        working = working.map(
            _apply_step_example,
            fn_kwargs={"tokens": tokens},
            num_proc=map_workers,
        )
        oe_tokens = list(cumulative_tokens)
        step_dir = order_dir / f"step{step_idx:02d}_{factor_key}"
        df = compute_goal_state_shifts(
            model,
            start_ds,
            working,
            oe_tokens,
            goal_state,
            cell_states_to_model,
            state_embs_dict,
            layer_to_quant,
            pad_token_id,
            model_input_size,
            batch_state[0],
            nproc,
            perturb_type=PERTURB_OVEREXPRESS,
            batch_state=batch_state,
        )
        _write_shifts(df, step_dir / "single_gene_per_cell_shifts.csv")
        if save_datasets:
            _save_dataset(working, step_dir / "perturbed.dataset")
        _empty_cuda_cache()

        stats = _summarize_series(df["Shift_to_goal_end"])
        row = {
            "order": tag,
            "step": step_idx,
            "factor": factor_key,
            "factors_applied": order_label(order[:step_idx]),
            "n": stats["n"],
            "median_shift": stats["median"],
            "mean_shift": stats["mean"],
            "p25": stats["p25"],
            "p75": stats["p75"],
            "frac_positive": stats["frac_positive"],
        }
        step_summaries.append(row)

    final = step_summaries[-1]
    final_summary = {
        "order": tag,
        "n": final["n"],
        "median": final["median_shift"],
        "mean": final["mean_shift"],
        "p25": final["p25"],
        "p75": final["p75"],
        "frac_positive": final["frac_positive"],
        "steps": len(order),
    }
    return step_summaries, final_summary


def _forward_cell_means(
    model,
    dataset,
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    nproc: int,
    *,
    strip_leading: int = 0,
) -> torch.Tensor:
    """Mean-pooled cell embeddings [n_cells, hidden] for mixed OE+KD scoring."""
    n = len(dataset)
    try:
        dataset.reset_format()
    except Exception:
        pass
    map_workers = _gpu_resident_map_workers(nproc)
    chunks: list[torch.Tensor] = []
    batch = max(1, int(forward_batch_size))
    for i in range(0, n, batch):
        max_range = min(i + batch, n)
        minibatch = dataset.select(list(range(i, max_range)))
        lengths = [int(x) for x in minibatch["length"]]
        max_len = min(max(lengths), model_input_size)

        def _pad(example):
            example["input_ids"] = isp.pad_or_truncate_encoding(
                example["input_ids"], pad_token_id, max_len
            )
            return example

        if any(L != max_len for L in lengths) or max(lengths) > model_input_size:
            minibatch = minibatch.map(_pad, num_proc=map_workers)
        minibatch.set_format(type="torch")
        input_ids = isp._tensor_to_device(isp._force_tensor(minibatch["input_ids"]))
        attention_mask = isp.gen_attention_mask(minibatch, max_len)
        with torch.no_grad():
            outputs = isp._forward_with_oom_retry(
                lambda: model(input_ids=input_ids, attention_mask=attention_mask),
            )
        embs = isp.batched_layer_embs(outputs, layer_to_quant).clone()
        del outputs
        if strip_leading > 0:
            embs = embs[:, strip_leading:, :]
            adj = torch.clamp(
                torch.as_tensor(lengths, dtype=torch.long, device=embs.device) - strip_leading,
                min=1,
            )
        else:
            adj = torch.as_tensor(lengths, dtype=torch.long, device=embs.device)
        means = isp.mean_nonpadding_embs(embs, adj)
        chunks.append(means.detach().to("cpu"))
        del embs, means
    return torch.cat(chunks, dim=0)


def compute_cell_mean_goal_state_shifts(
    model,
    original_ds,
    perturbed_ds,
    goal_state: str,
    state_embs_dict: dict[str, torch.Tensor],
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    nproc: int,
    *,
    strip_leading: int = 0,
    batch_state: list[int] | None = None,
) -> pd.DataFrame:
    """Cell-level goal_state_shift without gene-rank alignment (mixed OE+KD)."""
    batch = int(forward_batch_size)
    while True:
        try:
            orig_means = _forward_cell_means(
                model,
                original_ds,
                layer_to_quant,
                pad_token_id,
                model_input_size,
                batch,
                nproc,
            )
            pert_means = _forward_cell_means(
                model,
                perturbed_ds,
                layer_to_quant,
                pad_token_id,
                model_input_size,
                batch,
                nproc,
                strip_leading=strip_leading,
            )
            break
        except Exception as exc:
            if not _is_cuda_oom(exc) or batch <= 1:
                raise
            new_batch = max(1, batch // 2)
            print(
                f"CUDA OOM during mixed sequential scoring (batch={batch}); "
                f"retrying at {new_batch}",
                flush=True,
            )
            _empty_cuda_cache()
            batch = new_batch
            if batch_state is not None:
                batch_state[0] = batch

    goal = state_embs_dict[goal_state]
    if goal.dim() == 1:
        goal = goal.unsqueeze(0)
    goal = goal.detach().float().to("cpu")
    if goal.dim() == 3:
        goal = goal.squeeze(1)
    orig_means = orig_means.float()
    pert_means = pert_means.float()
    if goal.size(-1) != orig_means.size(-1):
        raise RuntimeError(
            f"Centroid hidden size {tuple(goal.shape)} does not match "
            f"cell embeddings {tuple(orig_means.shape)}"
        )
    if goal.size(0) == 1 and orig_means.size(0) != 1:
        goal_b = goal.expand(orig_means.size(0), -1)
    else:
        goal_b = goal
    cos = torch.nn.CosineSimilarity(dim=1)
    origin_v_end = cos(orig_means, goal_b)
    perturb_v_end = cos(pert_means, goal_b)
    shifts = (perturb_v_end - origin_v_end).numpy()

    meta = {
        c: original_ds[c]
        for c in original_ds.column_names
        if c not in {"input_ids", "length", "attention_mask"}
    }
    meta["cell_index"] = list(range(len(original_ds)))
    meta["Shift_to_goal_end"] = shifts
    return pd.DataFrame(meta)


def run_typed_steps(
    model,
    start_ds,
    steps: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    goal_state: str,
    cell_states_to_model: dict[str, Any],
    state_embs_dict: dict[str, torch.Tensor],
    out_root: Path,
    *,
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    save_datasets: bool,
    nproc: int,
    resume: bool,
) -> list[dict[str, Any]]:
    """Generic sequential OE / KD chain (platform mode)."""
    batch_state = [int(forward_batch_size)]
    map_workers = _gpu_resident_map_workers(nproc)
    working = start_ds
    try:
        working.reset_format()
    except Exception:
        pass

    cumulative_oe: list[int] = []
    cumulative_kd: list[int] = []
    step_rows: list[dict[str, Any]] = []

    for spec in steps:
        step_idx = int(spec["index"])
        tag = str(spec["name"])
        ptype = str(spec["type"])
        genes = list(spec["genes"])
        tokens, resolved = _resolve_gene_tokens(cfg, genes)
        safe_tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tag)
        step_dir = out_root / "steps" / f"step{step_idx:02d}_{safe_tag}"
        csv_path = step_dir / "single_gene_per_cell_shifts.csv"
        ds_path = step_dir / "perturbed.dataset"

        if ptype == PERTURB_DELETE:
            cumulative_kd.extend(tokens)
        else:
            cumulative_oe.extend(tokens)

        if resume and csv_path.is_file() and (not save_datasets or ds_path.exists()):
            print(f"Resume: skip step {step_idx} {tag}", flush=True)
            s = pd.to_numeric(pd.read_csv(csv_path)["Shift_to_goal_end"], errors="coerce").dropna()
            stats = _summarize_series(s)
            if save_datasets and ds_path.exists():
                from datasets import load_from_disk as _load

                working = _load(str(ds_path))
        else:
            print(
                f"Step {step_idx}: {ptype} {tag} ({len(tokens)} genes: {', '.join(resolved)})...",
                flush=True,
            )
            working = working.map(
                _apply_typed_step,
                fn_kwargs={"tokens": list(tokens), "perturb_type": ptype},
                num_proc=map_workers,
            )
            mixed = bool(cumulative_oe) and bool(cumulative_kd)
            if mixed:
                df = compute_cell_mean_goal_state_shifts(
                    model,
                    start_ds,
                    working,
                    goal_state,
                    state_embs_dict,
                    layer_to_quant,
                    pad_token_id,
                    model_input_size,
                    batch_state[0],
                    nproc,
                    strip_leading=0,
                    batch_state=batch_state,
                )
            else:
                score_tokens = list(cumulative_kd if ptype == PERTURB_DELETE else cumulative_oe)
                df = compute_goal_state_shifts(
                    model,
                    start_ds,
                    working,
                    score_tokens,
                    goal_state,
                    cell_states_to_model,
                    state_embs_dict,
                    layer_to_quant,
                    pad_token_id,
                    model_input_size,
                    batch_state[0],
                    nproc,
                    perturb_type=ptype,
                    batch_state=batch_state,
                )
            _write_shifts(df, csv_path)
            if save_datasets:
                _save_dataset(working, ds_path)
            stats = _summarize_series(df["Shift_to_goal_end"])
            print(
                f"  median shift={stats['median']:.6f} n={stats['n']} "
                f"frac>0={stats['frac_positive']:.3f} batch={batch_state[0]}",
                flush=True,
            )
        _empty_cuda_cache()
        step_rows.append(
            {
                "step": step_idx,
                "name": tag,
                "type": ptype,
                "genes": ",".join(genes),
                "n": stats["n"],
                "median": stats["median"],
                "mean": stats["mean"],
                "p25": stats["p25"],
                "p75": stats["p75"],
                "frac_positive": stats["frac_positive"],
            }
        )
    return step_rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sequential ISP (OE and/or KD)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--orders",
        nargs="*",
        help="OSKM mode: subset of orders as O-S-K-M (default: all 24 permutations)",
    )
    parser.add_argument("--max-ncells", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--no-save-datasets", action="store_true")
    parser.add_argument("--skip-simultaneous", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--forward-batch-size",
        default=None,
        metavar="N|auto",
        help="Override runtime.forward_batch_size ('auto' measures this GPU for dual-forward scoring).",
    )
    parser.add_argument("--nproc", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = _load_config(args.config.resolve())
    paths = cfg.get("paths") or {}
    pert = cfg.get("perturbation") or {}
    isp_cfg = cfg.get("isp") or {}
    mdl = cfg.get("model") or {}
    seq_cfg = cfg.get("sequential") or {}
    runtime = cfg.get("runtime") or {}

    dataset_path = _app_path(str(paths["dataset"]))
    model_path = _app_path(str(paths["geneformer_model"]))
    output_root = args.output_root or _app_path(str(paths["output_root"]))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    if args.output_root is None:
        date_on = bool(paths.get("output_date_subdir", True))
        time_on = bool(paths.get("output_time_subdir", True))
        if date_on:
            output_root = output_root / datetime.now(timezone.utc).strftime("%Y%m%d")
        if time_on:
            stamp = datetime.now(timezone.utc).strftime("%H%M%S_%f")[:-3] + "Z"
            output_root = output_root / f"sequential_isp_{stamp}"

    state_key = str(pert["state_key"])
    start_state = str(pert["start_state"])
    goal_state = str(pert["end_state"])

    max_ncells = args.max_ncells if args.max_ncells is not None else isp_cfg.get("max_ncells")
    emb_layer = int(isp_cfg.get("emb_layer", 0))
    nproc = args.nproc if args.nproc is not None else int(runtime.get("nproc", 4))
    typed_steps = parse_sequential_steps(seq_cfg)
    save_default = False if typed_steps else True
    save_datasets = not args.no_save_datasets and bool(
        seq_cfg.get("save_intermediate_datasets", save_default)
    )

    species_key = "human" if "human" in str((cfg.get("species") or {}).get("model_organism", "human")).lower() else "mouse"
    log_species_banner(species_from_config(cfg))

    from datasets import load_from_disk

    print(f"Loading dataset: {dataset_path}", flush=True)
    dataset = load_from_disk(str(dataset_path))
    start_ds = _select_start_cells(dataset, state_key, start_state, max_ncells)
    print(f"Start cells ({start_state}): n={len(start_ds)}", flush=True)

    model_type = mdl.get("type", "CellClassifier")
    num_classes = int(mdl.get("num_classes", 2))
    model = isp.load_model(model_type, num_classes, str(model_path))
    layer_to_quant = isp.quant_layers(model) + emb_layer
    model_input_size = isp.get_model_input_size(model)

    import pickle

    backend = backend_from_config(cfg)
    with open(backend.token_dictionary, "rb") as fh:
        token_dict = pickle.load(fh)
    pad_token_id = token_dict.get("<pad>")

    fbs_raw = args.forward_batch_size or runtime.get("forward_batch_size", isp_cfg.get("forward_batch_size", "auto"))
    species_default = default_isp_forward_batch_size(backend.max_input_size)
    forward_batch_size = resolve_sequential_forward_batch_size(
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
    centroid_ds = _centroid_dataset(dataset, state_key, centroid_states, max_ncells, nproc)
    state_embs = isp.get_cell_state_avg_embs(
        model,
        centroid_ds,
        cell_states,
        layer_to_quant,
        pad_token_id,
        forward_batch_size,
        _gpu_resident_map_workers(nproc),
    )
    _empty_cuda_cache()

    output_root.mkdir(parents=True, exist_ok=True)

    if typed_steps:
        run_meta = {
            "config": str(args.config.resolve()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "start_state": start_state,
            "goal_state": goal_state,
            "n_cells": len(start_ds),
            "mode": "sequential_steps",
            "steps": [
                {"name": s["name"], "type": s["type"], "genes": s["genes"]}
                for s in typed_steps
            ],
            "rank_convention": "later_oe_step_leftmost",
            "forward_batch_size": forward_batch_size,
        }
        (output_root / "run_manifest.json").write_text(json.dumps(run_meta, indent=2) + "\n")
        _save_dataset(start_ds, output_root / "start.dataset")
        print(
            f"Sequential ISP: {len(typed_steps)} step(s) "
            f"({' → '.join(s['type'] + ':' + s['name'] for s in typed_steps)})",
            flush=True,
        )
        step_rows = run_typed_steps(
            model,
            start_ds,
            typed_steps,
            cfg,
            goal_state,
            cell_states,
            state_embs,
            output_root,
            layer_to_quant=layer_to_quant,
            pad_token_id=pad_token_id,
            model_input_size=model_input_size,
            forward_batch_size=forward_batch_size,
            save_datasets=save_datasets,
            nproc=nproc,
            resume=not args.no_resume,
        )
        pd.DataFrame(step_rows).to_csv(output_root / "step_summary.csv", index=False)
        pd.DataFrame(step_rows).to_csv(output_root / "order_summary.csv", index=False)
        print(f"Wrote {output_root / 'step_summary.csv'}", flush=True)
        return 0

    token_by_factor = _resolve_factor_tokens(species_key, cfg)
    all_tokens = [token_by_factor[k] for k in OSKM_FACTOR_KEYS]
    orders = _resolve_orders(args.orders or seq_cfg.get("orders"))

    output_root.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "config": str(args.config.resolve()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "start_state": start_state,
        "goal_state": goal_state,
        "n_cells": len(start_ds),
        "orders": [order_label(o) for o in orders],
        "rank_convention": "later_step_leftmost",
        "mode": "sequential_token_oe",
    }
    (output_root / "run_manifest.json").write_text(json.dumps(run_meta, indent=2) + "\n")

    summaries: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    resume = not args.no_resume

    sim_csv = output_root / "simultaneous_oskm4" / "single_gene_per_cell_shifts.csv"
    if not args.skip_simultaneous:
        if resume and sim_csv.is_file():
            print("Resume: skip simultaneous OSKM4 (already on disk)", flush=True)
            sim_df = pd.read_csv(sim_csv)
            sim_summary = _summarize_series(sim_df["Shift_to_goal_end"])
            sim_summary["order"] = "simultaneous_OSKM4"
            summaries.append(sim_summary)
        else:
            print("Running simultaneous OSKM4 baseline...", flush=True)
            sim_dir = output_root / "simultaneous_oskm4"
            sim_summary = run_simultaneous_baseline(
                model,
                start_ds,
                all_tokens,
                goal_state,
                cell_states,
                state_embs,
                sim_dir,
                layer_to_quant=layer_to_quant,
                pad_token_id=pad_token_id,
                model_input_size=model_input_size,
                forward_batch_size=forward_batch_size,
                save_dataset=save_datasets,
                nproc=nproc,
            )
            sim_summary["order"] = "simultaneous_OSKM4"
            summaries.append(sim_summary)
            print(f"  simultaneous median shift={sim_summary['median']:.6f}", flush=True)

    for order in orders:
        tag = order_label(order)
        if resume and _order_complete(output_root, order):
            print(f"Resume: skip order {tag}", flush=True)
            steps = _load_step_rows_from_disk(output_root, order)
            step_rows.extend(steps)
            final = steps[-1]
            summaries.append(
                {
                    "order": tag,
                    "n": final["n"],
                    "median": final["median_shift"],
                    "mean": final["mean_shift"],
                    "p25": final["p25"],
                    "p75": final["p75"],
                    "frac_positive": final["frac_positive"],
                    "steps": len(order),
                }
            )
            continue
        print(f"Order {tag}...", flush=True)
        steps, final = run_order(
            model,
            start_ds,
            order,
            token_by_factor,
            goal_state,
            cell_states,
            state_embs,
            output_root,
            layer_to_quant=layer_to_quant,
            pad_token_id=pad_token_id,
            model_input_size=model_input_size,
            forward_batch_size=forward_batch_size,
            save_datasets=save_datasets,
            nproc=nproc,
        )
        step_rows.extend(steps)
        summaries.append(final)
        print(f"  final median shift={final['median']:.6f}", flush=True)
        pd.DataFrame(summaries).sort_values("median", ascending=False).to_csv(
            output_root / "order_summary.csv", index=False
        )
        pd.DataFrame(step_rows).to_csv(output_root / "step_summary.csv", index=False)

    order_df = pd.DataFrame(summaries).sort_values("median", ascending=False)
    order_df.to_csv(output_root / "order_summary.csv", index=False)
    pd.DataFrame(step_rows).to_csv(output_root / "step_summary.csv", index=False)
    print(f"Wrote {output_root / 'order_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
