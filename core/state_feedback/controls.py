"""Controls for direction fidelity: base rank and perturbation specificity.

Base-rank control. With an endpoint teacher, where a gene already sits in the
cell predicts much of its observed displacement (top-ranked genes tend to move
down, bottom-ranked genes up). A predictor can therefore score a high Spearman
without reading the perturbation at all. Two quantities separate the two:

* ``crossfit_base_only`` — the best a base-rank-only predictor does (binned mean
  target per base-rank bin, fitted on one half of the genes, applied to the other).
* ``partial_spearman_given_base`` — Spearman after removing the per-bin mean from
  both the ranked prediction and the ranked target, i.e. what a predictor adds
  beyond base rank.

Perturbation-specificity control. The same decoder is fed Δh from other
perturbations (named sets and random gene draws). If the signal beyond base rank
were a property of the scored genes alone, it would not depend on the perturbation.

Everything here is numpy; bootstrap resampling is by cell.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

N_BASE_BINS = 20
N_BASE_STRATA = 5


def average_rank(x: Sequence[float]) -> np.ndarray:
    """0-based ranks; ties share their mean rank."""
    arr = np.asarray(x, dtype=np.float64)
    _, inv, counts = np.unique(arr, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + ends - 1) / 2.0)[inv]


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(np.corrcoef(average_rank(a), average_rank(b))[0, 1])


def base_bins(base: Sequence[float], n_bins: int = N_BASE_BINS) -> np.ndarray:
    """Quantile bin index per sample."""
    base = np.asarray(base, dtype=np.float64)
    edges = np.quantile(base, np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1])
    return np.searchsorted(edges, base, side="right")


def _demean_by_bin(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    sums = np.bincount(bins, weights=x)
    counts = np.bincount(bins)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return x - means[bins]


def _partial(pred: np.ndarray, target: np.ndarray, bins: np.ndarray) -> float:
    p = _demean_by_bin(average_rank(pred), bins)
    t = _demean_by_bin(average_rank(target), bins)
    if p.std() == 0 or t.std() == 0:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def partial_spearman_given_base(
    pred: Sequence[float],
    target: Sequence[float],
    base: Sequence[float],
    *,
    n_bins: int = N_BASE_BINS,
) -> float:
    return _partial(
        np.asarray(pred, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        base_bins(base, n_bins),
    )


def stratified_spearman(
    pred: Sequence[float],
    target: Sequence[float],
    base: Sequence[float],
    *,
    n_strata: int = N_BASE_STRATA,
) -> list[float]:
    """Pooled Spearman within each base-rank quantile stratum (top to bottom)."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    s = base_bins(base, n_strata)
    return [spearman(pred[s == k], target[s == k]) for k in range(int(n_strata))]


def target_by_base_stratum(
    target: Sequence[float],
    base: Sequence[float],
    *,
    n_strata: int = N_BASE_STRATA,
) -> list[float]:
    target = np.asarray(target, dtype=np.float64)
    s = base_bins(base, n_strata)
    return [float(target[s == k].mean()) for k in range(int(n_strata))]


def crossfit_base_only(
    base: Sequence[float],
    target: Sequence[float],
    tokens: Sequence[int],
    *,
    n_bins: int = N_BASE_BINS,
) -> np.ndarray:
    """Base-rank-only predictor, cross-fitted over two gene folds (token parity).

    Non-monotone in base rank, and never sees the target of the gene it predicts.
    """
    base = np.asarray(base, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    fold = np.asarray(tokens, dtype=np.int64) % 2
    bins = base_bins(base, n_bins)
    pred = np.zeros(len(target), dtype=np.float64)
    for f in (0, 1):
        fit, app = fold != f, fold == f
        if not app.any():
            continue
        sums = np.bincount(bins[fit], weights=target[fit], minlength=int(n_bins))
        counts = np.bincount(bins[fit], minlength=int(n_bins))
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        pred[app] = means[bins[app]]
    return pred


def _cell_groups(cells: Sequence[int]) -> list[np.ndarray]:
    cells = np.asarray(cells)
    return [np.flatnonzero(cells == c) for c in np.unique(cells)]


def _ci(values: list[float]) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def bootstrap_partial(
    pred: Sequence[float],
    target: Sequence[float],
    base: Sequence[float],
    cells: Sequence[int],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    n_bins: int = N_BASE_BINS,
) -> dict[str, Any]:
    """Partial Spearman given base rank with a cell-resampled 95% CI.

    Bin edges are fixed on the full sample so resamples share one definition of
    "same base rank".
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    bins = base_bins(base, n_bins)
    point = _partial(pred, target, bins)
    groups = _cell_groups(cells)
    rng = np.random.default_rng(int(seed))
    vals = []
    for _ in range(int(n_boot)):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        vals.append(_partial(pred[idx], target[idx], bins[idx]))
    lo, hi = _ci(vals)
    return {"partial": point, "ci_low": lo, "ci_high": hi,
            "excludes_zero": bool(lo > 0 or hi < 0), "n_boot": int(n_boot)}


def bootstrap_partial_diff(
    pred_a: Sequence[float],
    pred_b: Sequence[float],
    target: Sequence[float],
    base: Sequence[float],
    cells: Sequence[int],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    n_bins: int = N_BASE_BINS,
) -> dict[str, Any]:
    """Paired cell bootstrap on partial(a) - partial(b) over the same samples."""
    pa = np.asarray(pred_a, dtype=np.float64)
    pb = np.asarray(pred_b, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    bins = base_bins(base, n_bins)
    point = _partial(pa, target, bins) - _partial(pb, target, bins)
    groups = _cell_groups(cells)
    rng = np.random.default_rng(int(seed))
    vals = []
    for _ in range(int(n_boot)):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        vals.append(_partial(pa[idx], target[idx], bins[idx]) - _partial(pb[idx], target[idx], bins[idx]))
    lo, hi = _ci(vals)
    return {"diff": float(point), "ci_low": lo, "ci_high": hi,
            "excludes_zero": bool(lo > 0 or hi < 0), "n_boot": int(n_boot)}


def decoder_variants(decoder, samples, *, seed: int = 0) -> dict[str, np.ndarray]:
    """The decoder's Δr̂ with each input isolated.

    ``delta_h_only`` fixes base rank at 0.5, so its ranking depends on the
    direction of Δh alone (tanh is monotone); ``delta_h_shuffled`` permutes Δh
    across genes within each cell, keeping base rank.
    """
    import torch

    dev = next(decoder.parameters()).device
    dh = samples.delta_h.to(dev)
    br = samples.base_rank.to(dev)
    gen = torch.Generator().manual_seed(int(seed))
    perm = torch.arange(len(samples))
    for idx in _cell_groups(samples.cell_index):
        t = torch.from_numpy(idx)
        perm[t] = t[torch.randperm(len(t), generator=gen)]
    with torch.no_grad():
        return {
            "linear_deltarank": decoder.delta_rank(dh, br).cpu().numpy(),
            "delta_h_only": decoder.delta_rank(dh, torch.full_like(br, 0.5)).cpu().numpy(),
            "delta_h_shuffled": decoder.delta_rank(dh[perm.to(dev)], br).cpu().numpy(),
        }


def base_rank_control(
    samples,
    decoder,
    *,
    extra: Mapping[str, Sequence[float]] | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Base-rank control report for one EvalSamples set."""
    base = samples.base_rank.detach().cpu().numpy().astype(np.float64)
    target = samples.target.detach().cpu().numpy().astype(np.float64)
    cells = np.asarray(samples.cell_index)
    preds: dict[str, np.ndarray] = {}
    if decoder is not None:
        preds.update(decoder_variants(decoder, samples, seed=seed))
    for name, p in (extra or {}).items():
        preds[name] = np.asarray(p, dtype=np.float64)

    bins = base_bins(base)
    rows = {}
    for name, p in preds.items():
        rows[name] = {
            "pooled_spearman": spearman(p, target),
            "partial_spearman_given_base": _partial(p, target, bins),
            "stratified_by_base": stratified_spearman(p, target, base),
        }
    report: dict[str, Any] = {
        "n_samples": int(len(target)),
        "base_only_pooled_spearman": spearman(
            crossfit_base_only(base, target, samples.tokens), target
        ),
        "base_rank_pooled_spearman": spearman(base, target),
        "target_by_base_stratum": target_by_base_stratum(target, base),
        "methods": rows,
    }
    if "linear_deltarank" in preds:
        report["linear_partial"] = bootstrap_partial(
            preds["linear_deltarank"], target, base, cells, n_boot=n_boot, seed=seed
        )
        report["mean_delta_h_norm"] = float(samples.delta_h.norm(dim=-1).mean())
    return report


def detection_rate(input_ids: Sequence[Sequence[int]]) -> dict[int, float]:
    """Fraction of cells whose encoding contains each token."""
    counts: dict[int, int] = {}
    for cell in input_ids:
        for t in set(int(x) for x in cell):
            counts[t] = counts.get(t, 0) + 1
    n = max(1, len(input_ids))
    return {t: c / n for t, c in counts.items()}


def detection_matched_pool(
    pool: Sequence[int],
    detection: Mapping[int, float],
    reference: float,
    *,
    low: float = 0.5,
    high: float = 2.0,
) -> list[int]:
    """Tokens whose detection rate lies within [low, high] x ``reference``."""
    lo, hi = float(low) * float(reference), float(high) * float(reference)
    return [int(t) for t in pool if lo <= detection.get(int(t), 0.0) <= hi]


def versus_random(value: float, random_values: Sequence[float]) -> dict[str, Any]:
    """Place one condition's partial Spearman against the random-draw distribution.

    ``empirical_p`` = (1 + #random >= value) / (1 + n_random); its floor is set by
    the number of draws, so also report the distance in random-draw SDs.
    """
    r = np.asarray([v for v in random_values if v == v], dtype=np.float64)
    if len(r) == 0:
        return {"random_n": 0}
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else float("nan")
    n_ge = int(np.sum(r >= value))
    return {
        "random_n": int(len(r)),
        "random_mean": float(r.mean()),
        "random_sd": sd,
        "random_q95": float(np.quantile(r, 0.95)),
        "random_max": float(r.max()),
        "n_random_ge": n_ge,
        "empirical_p": float((1 + n_ge) / (1 + len(r))),
        "sd_above_random_mean": float((value - r.mean()) / sd) if sd and sd == sd else float("nan"),
    }
