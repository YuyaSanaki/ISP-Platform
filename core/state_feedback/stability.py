"""Stability metrics for multi-step State-feedback ISP.

A chain is observed as a sequence of encodings (one token list per cell):

    start -> step_1 -> feedback_1 -> step_2 -> ... -> feedback_T -> idle_1 -> ... -> idle_K

``idle`` events rerank again with no new perturbation. For every observed encoding
``ChainTracker`` records the per-cell change from the previous encoding and the
cumulative distance from a reference path (Ordered rank-edit at the same step).
``stability_verdict`` applies the pre-registered criteria S1-S4 in
docs/state_feedback_decode_methods.md.

All comparisons use the tokens shared by both encodings, so they stay defined when
an overexpression drops a tail token.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from state_feedback.metrics import median_iqr

DEFAULT_TOPK = (100, 500, 1000)


def _as_array(ids: Sequence[int]) -> np.ndarray:
    return np.asarray(ids, dtype=np.int64)


def pair_metrics(
    a: Sequence[int] | np.ndarray,
    b: Sequence[int] | np.ndarray,
    topk: Iterable[int] = DEFAULT_TOPK,
) -> dict[str, float]:
    """Per-cell change between two encodings of one cell."""
    xa = _as_array(a)
    xb = _as_array(b)
    _, ia, ib = np.intersect1d(xa, xb, assume_unique=True, return_indices=True)
    n = int(ia.size)
    out: dict[str, float] = {
        "n_shared": float(n),
        "n_entered": float(xb.size - n),
    }
    if n >= 2:
        ra = np.argsort(np.argsort(ia)).astype(np.float64)
        rb = np.argsort(np.argsort(ib)).astype(np.float64)
        ra -= ra.mean()
        rb -= rb.mean()
        den = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
        out["spearman"] = float((ra * rb).sum() / den) if den > 0 else float("nan")
        disp = np.abs(ib - ia).astype(np.float64)
        out["disp_median"] = float(np.median(disp))
        out["disp_p90"] = float(np.quantile(disp, 0.9))
        out["disp_max"] = float(disp.max())
        out["frac_moved"] = float((disp > 0).mean())
    else:
        for key in ("spearman", "disp_median", "disp_p90", "disp_max", "frac_moved"):
            out[key] = float("nan")
    for k in topk:
        k = int(k)
        sa = set(xa[:k].tolist())
        sb = set(xb[:k].tolist())
        union = sa | sb
        out[f"jaccard_{k}"] = float(len(sa & sb) / len(union)) if union else float("nan")
    return out


def cellwise(
    before: Sequence[Sequence[int]],
    after: Sequence[Sequence[int]],
    topk: Iterable[int] = DEFAULT_TOPK,
) -> list[dict[str, float]]:
    if len(before) != len(after):
        raise ValueError(f"cell counts differ: {len(before)} vs {len(after)}")
    topk = tuple(int(k) for k in topk)
    return [pair_metrics(b, a, topk) for b, a in zip(before, after)]


def summarize_cells(cells: Sequence[Mapping[str, float]], prefix: str = "") -> dict[str, float]:
    """Median and IQR over cells for every metric."""
    if not cells:
        return {}
    out: dict[str, float] = {}
    for key in cells[0]:
        stats = median_iqr([c[key] for c in cells])
        out[f"{prefix}{key}_median"] = stats["median"]
        out[f"{prefix}{key}_q25"] = stats["q25"]
        out[f"{prefix}{key}_q75"] = stats["q75"]
    return out


class ChainTracker:
    """Per-event change and drift from a reference path for one chain."""

    def __init__(
        self,
        chain: str,
        start_ids: Sequence[Sequence[int]],
        *,
        reference: Mapping[int, Sequence[Sequence[int]]] | None = None,
        topk: Iterable[int] = DEFAULT_TOPK,
    ):
        self.chain = chain
        self.topk = tuple(int(k) for k in topk)
        self.reference = dict(reference or {})
        self.snapshots: dict[str, list[np.ndarray]] = {}
        self.prev = [_as_array(r) for r in start_ids]
        self.rows: list[dict[str, Any]] = []
        self.event_index = 0

    def observe(
        self,
        kind: str,
        step: int,
        ids: Sequence[Sequence[int]],
        *,
        ref_step: int | None = None,
        keep: bool = False,
        **extra: Any,
    ) -> dict[str, Any]:
        """Record one encoding; ``ref_step`` picks the reference (default ``step``)."""
        cur = [_as_array(r) for r in ids]
        self.event_index += 1
        label = f"{kind}{int(step)}"
        row: dict[str, Any] = {
            "chain": self.chain,
            "event_index": self.event_index,
            "kind": kind,
            "step": int(step),
            "label": label,
            **extra,
            **summarize_cells(cellwise(self.prev, cur, self.topk), "event_"),
        }
        ref = self.reference.get(int(step if ref_step is None else ref_step))
        if ref is not None:
            row.update(summarize_cells(cellwise(ref, cur, self.topk), "vs_ref_"))
        self.rows.append(row)
        if keep:
            self.snapshots[label] = cur
        self.prev = cur
        return row

    @property
    def last(self) -> list[np.ndarray]:
        return self.prev


def cross_seed_agreement(
    finals: Mapping[Any, Sequence[Sequence[int]]],
    topk: Iterable[int] = DEFAULT_TOPK,
) -> list[dict[str, Any]]:
    """Per-cell agreement of the final encodings between every pair of seeds."""
    rows: list[dict[str, Any]] = []
    for sa, sb in combinations(sorted(finals), 2):
        cells = cellwise(finals[sa], finals[sb], topk)
        rows.append({"seed_a": sa, "seed_b": sb, **summarize_cells(cells)})
    return rows


def specific_gain(
    configured: Sequence[float],
    configured_ref: Sequence[float],
    randoms: Sequence[Sequence[float]],
    random_refs: Sequence[Sequence[float]],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Per-cell gain of the configured chain beyond the mean random-chain gain.

    Every argument is per cell in the same cell order; ``*_ref`` are the Ordered
    rank-edit end points of the same chain. Returns the cell mean of
    ``gain_configured - mean_r gain_random_r`` with a cell-bootstrap 95% CI.
    """
    conf = np.asarray(configured, dtype=np.float64) - np.asarray(configured_ref, dtype=np.float64)
    if not randoms:
        raise ValueError("specific_gain needs at least one random chain")
    rand = np.mean(
        [np.asarray(r, dtype=np.float64) - np.asarray(rr, dtype=np.float64)
         for r, rr in zip(randoms, random_refs)],
        axis=0,
    )
    if conf.shape != rand.shape:
        raise ValueError(f"cell counts differ: {conf.shape} vs {rand.shape}")
    diff = conf - rand
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, diff.size, size=(int(n_boot), diff.size))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    gain = float(conf.mean())
    spec = float(diff.mean())
    return {
        "n_cells": float(diff.size),
        "configured_gain_mean": gain,
        "random_gain_mean": float(rand.mean()),
        "specific_gain_mean": spec,
        "specific_gain_median": float(np.median(diff)),
        "specific_gain_ci_low": float(lo),
        "specific_gain_ci_high": float(hi),
        "specific_fraction": spec / gain if gain != 0 else float("nan"),
        "frac_cells_positive": float((diff > 0).mean()),
    }


COMPARISON_CRITERIA = {"D2_ratio_min": 0.5, "D2_ratio_max": 2.0}


def mode_comparison_verdict(
    spec_by_seed: Mapping[Any, Mapping[str, Mapping[str, float]]],
    *,
    multi: str = "multi_step",
    single: str = "single_last",
    criteria: Mapping[str, float] = COMPARISON_CRITERIA,
) -> dict[str, Any]:
    """D1/D2 over seeds; ``spec_by_seed[seed][mode]`` is a ``specific_gain`` result."""
    c = {**COMPARISON_CRITERIA, **dict(criteria)}

    def excludes_zero(r: Mapping[str, float]) -> bool:
        return bool(r["specific_gain_ci_low"] > 0 or r["specific_gain_ci_high"] < 0)

    per_seed: dict[Any, dict[str, Any]] = {}
    for seed, modes in spec_by_seed.items():
        if multi not in modes or single not in modes:
            continue
        m, s = modes[multi], modes[single]
        ratio = (
            m["specific_gain_mean"] / s["specific_gain_mean"]
            if s["specific_gain_mean"] != 0
            else float("nan")
        )
        per_seed[seed] = {
            "multi_excludes_zero": excludes_zero(m),
            "single_excludes_zero": excludes_zero(s),
            "ratio_multi_over_single": ratio,
            "ratio_in_range": bool(ratio == ratio and c["D2_ratio_min"] <= ratio <= c["D2_ratio_max"]),
        }
    seeds = list(per_seed.values())
    d1_multi = bool(seeds) and all(v["multi_excludes_zero"] for v in seeds)
    d1_single = bool(seeds) and all(v["single_excludes_zero"] for v in seeds)
    d2 = bool(seeds) and all(v["ratio_in_range"] for v in seeds)
    if d1_multi and d1_single and d2:
        primary = "every_step"
    elif d1_single:
        primary = "single_event"
    else:
        primary = "none"
    return {
        "criteria": c,
        "per_seed": per_seed,
        "D1_multi": d1_multi,
        "D1_single": d1_single,
        "D2": d2,
        "primary": primary,
    }


def coefficient_of_variation(values: Sequence[float]) -> float:
    vals = np.asarray([v for v in values if v == v], dtype=np.float64)
    if vals.size < 2:
        return float("nan")
    mean = float(vals.mean())
    if mean == 0.0:
        return float("nan")
    return float(vals.std(ddof=1) / abs(mean))


def _series(rows: Sequence[Mapping[str, Any]], kind: str, key: str) -> list[float]:
    return [float(r[key]) for r in rows if r["kind"] == kind and key in r]


CRITERIA = {
    "S1_spearman_drop_max": 0.02,
    "S1_disp_growth_max": 1.5,
    "S2_idle_spearman_min": 0.99,
    "S4_seed_spearman_min": 0.95,
    "S4_endpoint_cv_max": 0.10,
}


def stability_verdict(
    event_rows_by_seed: Mapping[Any, Sequence[Mapping[str, Any]]],
    endpoint_gain_by_seed: Mapping[Any, Mapping[str, float]],
    endpoint_by_seed: Mapping[Any, float],
    seed_agreement: Sequence[Mapping[str, Any]],
    *,
    configured: str = "configured",
    criteria: Mapping[str, float] = CRITERIA,
) -> dict[str, Any]:
    """Apply S1-S4 to the configured chain.

    ``event_rows_by_seed[s]`` are the ``ChainTracker`` rows of the configured
    multi-step chain; ``endpoint_gain_by_seed[s]`` maps each chain name to its
    endpoint shift minus the Ordered rank-edit endpoint of the same chain.
    """
    c = {**CRITERIA, **dict(criteria)}
    per_seed: dict[Any, dict[str, Any]] = {}
    for seed, rows in event_rows_by_seed.items():
        rho = _series(rows, "feedback", "event_spearman_median")
        disp = _series(rows, "feedback", "event_disp_median_median")
        idle_rho = _series(rows, "idle", "event_spearman_median")
        idle_disp = _series(rows, "idle", "event_disp_median_median")
        s1 = bool(
            rho
            and min(rho) >= rho[0] - c["S1_spearman_drop_max"]
            and max(disp) <= c["S1_disp_growth_max"] * max(disp[0], 1.0)
        )
        s2 = bool(
            idle_rho
            and idle_rho[-1] >= c["S2_idle_spearman_min"]
            and idle_disp[-1] <= idle_disp[0]
        )
        gains = dict(endpoint_gain_by_seed.get(seed, {}))
        own = gains.pop(configured, float("nan"))
        random_max = max(gains.values()) if gains else float("nan")
        s3 = bool(gains and own == own and own > random_max)
        per_seed[seed] = {
            "feedback_spearman": rho,
            "feedback_disp_median": disp,
            "idle_spearman": idle_rho,
            "idle_disp_median": idle_disp,
            "configured_gain": own,
            "random_gain_max": random_max,
            "S1": s1,
            "S2": s2,
            "S3": s3,
        }
    seed_rho = [float(r["spearman_median"]) for r in seed_agreement]
    cv = coefficient_of_variation(list(endpoint_by_seed.values()))
    s4 = bool(
        seed_rho
        and min(seed_rho) >= c["S4_seed_spearman_min"]
        and cv == cv
        and cv <= c["S4_endpoint_cv_max"]
    )
    passed = {
        "S1": all(v["S1"] for v in per_seed.values()) if per_seed else False,
        "S2": all(v["S2"] for v in per_seed.values()) if per_seed else False,
        "S3": all(v["S3"] for v in per_seed.values()) if per_seed else False,
        "S4": s4,
    }
    return {
        "criteria": c,
        "per_seed": per_seed,
        "seed_spearman_min": min(seed_rho) if seed_rho else float("nan"),
        "endpoint_cv": cv,
        "passed": passed,
        "multi_step_supported": all(passed.values()),
    }
