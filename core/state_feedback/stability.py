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
    ``gain_configured - mean_r gain_random_r``. The 95% CI resamples cells and, with
    two or more random chains, the random chains too, so it includes the spread
    from which random genes were drawn.

    ``null_*`` places the configured value against the random chains themselves:
    each random chain in turn is scored as if it were the configured one, against
    the mean of the others (needs three or more random chains). ``z_vs_random`` is
    the configured mean gain in SDs of the random-chain mean gains (same minimum).
    """
    conf = np.asarray(configured, dtype=np.float64) - np.asarray(configured_ref, dtype=np.float64)
    if not randoms:
        raise ValueError("specific_gain needs at least one random chain")
    rand_by_chain = np.stack(
        [np.asarray(r, dtype=np.float64) - np.asarray(rr, dtype=np.float64)
         for r, rr in zip(randoms, random_refs)]
    )
    rand = rand_by_chain.mean(axis=0)
    if conf.shape != rand.shape:
        raise ValueError(f"cell counts differ: {conf.shape} vs {rand.shape}")
    diff = conf - rand
    n_chains, n_cells = rand_by_chain.shape
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        cells = rng.integers(0, n_cells, n_cells)
        chains = rng.integers(0, n_chains, n_chains) if n_chains > 1 else np.zeros(1, dtype=int)
        boot[b] = conf[cells].mean() - rand_by_chain[np.ix_(chains, cells)].mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    gain = float(conf.mean())
    spec = float(diff.mean())
    chain_means = rand_by_chain.mean(axis=1)
    out = {
        "n_cells": float(n_cells),
        "n_random_chains": float(n_chains),
        "configured_gain_mean": gain,
        "random_gain_mean": float(rand.mean()),
        "random_gain_sd_chains": float(chain_means.std(ddof=1)) if n_chains > 1 else float("nan"),
        "random_gain_se_chains": (
            float(chain_means.std(ddof=1) / np.sqrt(n_chains)) if n_chains > 1 else float("nan")
        ),
        "specific_gain_mean": spec,
        "specific_gain_median": float(np.median(diff)),
        "specific_gain_ci_low": float(lo),
        "specific_gain_ci_high": float(hi),
        "specific_fraction": spec / gain if gain != 0 else float("nan"),
        "frac_cells_positive": float((diff > 0).mean()),
        "rank_among_random": float(1 + np.sum(chain_means >= gain)),
        "frac_cells_above_all_random": float((conf > rand_by_chain.max(axis=0)).mean()),
    }
    if n_chains >= 3:
        out["z_vs_random"] = float((gain - chain_means.mean()) / chain_means.std(ddof=1))
        pseudo = np.array([
            chain_means[r] - np.delete(chain_means, r).mean() for r in range(n_chains)
        ])
        out["null_sd"] = float(pseudo.std(ddof=1))
        out["null_max"] = float(pseudo.max())
        out["null_empirical_p"] = float((1 + np.sum(pseudo >= spec)) / (1 + n_chains))
    return out


def specific_gain_conditional(
    configured: Sequence[float],
    configured_ref: Sequence[float],
    randoms: Sequence[Sequence[float]],
    random_refs: Sequence[Sequence[float]],
    treated: Sequence[bool],
    available: Sequence[Sequence[bool]],
    *,
    min_random_per_cell: int = 3,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """``specific_gain`` for chains with delete steps.

    Only treated cells count (every deleted factor in the start encoding), and in each
    cell only the random chains whose deleted genes were all in that cell
    (``available[r][i]``): a deletion of an absent gene does nothing, so including those
    chains would favour the configured chain. Cells with fewer than
    ``min_random_per_cell`` such chains are dropped (``n_cells_dropped``). Random-chain
    means, z, rank and the null use the same cells and the same availability.
    """
    conf_all = np.asarray(configured, dtype=np.float64) - np.asarray(configured_ref, dtype=np.float64)
    if not randoms:
        raise ValueError("specific_gain needs at least one random chain")
    rand_all = np.stack(
        [np.asarray(r, dtype=np.float64) - np.asarray(rr, dtype=np.float64)
         for r, rr in zip(randoms, random_refs)]
    )
    avail = np.asarray(available, dtype=bool)
    treat = np.asarray(treated, dtype=bool)
    if avail.shape != rand_all.shape or treat.shape != conf_all.shape:
        raise ValueError("treated / available must match the cells and random chains")
    keep = treat & (avail.sum(axis=0) >= int(min_random_per_cell))
    conf = conf_all[keep]
    rand_by_chain = np.where(avail, rand_all, np.nan)[:, keep]
    used = ~np.all(np.isnan(rand_by_chain), axis=1)
    rand_by_chain = rand_by_chain[used]
    n_chains, n_cells = rand_by_chain.shape
    if n_cells == 0:
        raise ValueError("no treated cell has enough random chains")
    rand = np.nanmean(rand_by_chain, axis=0)
    diff = conf - rand
    rng = np.random.default_rng(int(seed))
    boot = np.full(int(n_boot), np.nan)
    for b in range(int(n_boot)):
        cells = rng.integers(0, n_cells, n_cells)
        chains = rng.integers(0, n_chains, n_chains) if n_chains > 1 else np.zeros(1, dtype=int)
        sub = rand_by_chain[np.ix_(chains, cells)]
        ok = ~np.all(np.isnan(sub), axis=0)
        if ok.any():
            boot[b] = conf[cells][ok].mean() - np.nanmean(sub[:, ok], axis=0).mean()
    lo, hi = np.nanquantile(boot, [0.025, 0.975])
    gain = float(conf.mean())
    spec = float(diff.mean())
    chain_means = np.nanmean(rand_by_chain, axis=1)
    out = {
        "n_cells": float(n_cells),
        "n_cells_dropped": float(int(treat.sum()) - n_cells),
        "n_random_chains": float(n_chains),
        "random_chains_per_cell_mean": float((~np.isnan(rand_by_chain)).sum(axis=0).mean()),
        "configured_gain_mean": gain,
        "random_gain_mean": float(rand.mean()),
        "random_gain_sd_chains": float(chain_means.std(ddof=1)) if n_chains > 1 else float("nan"),
        "random_gain_se_chains": (
            float(chain_means.std(ddof=1) / np.sqrt(n_chains)) if n_chains > 1 else float("nan")
        ),
        "specific_gain_mean": spec,
        "specific_gain_median": float(np.median(diff)),
        "specific_gain_ci_low": float(lo),
        "specific_gain_ci_high": float(hi),
        "specific_fraction": spec / gain if gain != 0 else float("nan"),
        "frac_cells_positive": float((diff > 0).mean()),
        "rank_among_random": float(1 + np.sum(chain_means >= gain)),
        "frac_cells_above_all_random": float((conf > np.nanmax(rand_by_chain, axis=0)).mean()),
    }
    if n_chains >= 3:
        out["z_vs_random"] = float((gain - chain_means.mean()) / chain_means.std(ddof=1))
        pseudo = np.array([
            chain_means[r] - np.delete(chain_means, r).mean() for r in range(n_chains)
        ])
        out["null_sd"] = float(pseudo.std(ddof=1))
        out["null_max"] = float(pseudo.max())
        out["null_empirical_p"] = float((1 + np.sum(pseudo >= spec)) / (1 + n_chains))
    return out


def running_random_mean(
    randoms: Sequence[Sequence[float]],
    random_refs: Sequence[Sequence[float]],
) -> list[dict[str, float]]:
    """Mean random-chain gain as chains are added in draw order, with its chain SE.

    If the random draw is well behaved the mean settles and the SE shrinks as
    ``1/sqrt(n)``; a jump when one chain is added points to an outlier gene.
    """
    chain_means = [
        float(np.mean(np.asarray(r, dtype=np.float64) - np.asarray(rr, dtype=np.float64)))
        for r, rr in zip(randoms, random_refs)
    ]
    rows = []
    for n in range(1, len(chain_means) + 1):
        head = np.asarray(chain_means[:n])
        rows.append({
            "n_chains": n,
            "chain_gain": chain_means[n - 1],
            "running_mean": float(head.mean()),
            "running_se": float(head.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
        })
    return rows


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
