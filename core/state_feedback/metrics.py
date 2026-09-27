"""Metrics for State-feedback ISP: rank correlation, overlap, drift."""
from __future__ import annotations

from typing import Mapping, Sequence


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Ranks with ties averaged (1-based)."""
    n = len(values)
    order = sorted(range(n), key=lambda i: float(values[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and float(values[order[j + 1]]) == float(values[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    num = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    den_a = sum((x - mean_a) ** 2 for x in a) ** 0.5
    den_b = sum((x - mean_b) ** 2 for x in b) ** 0.5
    if den_a == 0.0 or den_b == 0.0:
        return float("nan")
    return float(num / (den_a * den_b))


def spearman_values(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rho between two value vectors (ties averaged)."""
    if len(a) != len(b):
        raise ValueError("spearman_values requires equal-length inputs")
    if len(a) < 2:
        return float("nan")
    return _pearson(_average_ranks(a), _average_ranks(b))


def topk_jaccard(a: Sequence[int], b: Sequence[int], k: int) -> float:
    """Jaccard overlap of the leading ``k`` tokens of two encodings."""
    sa = set(int(t) for t in a[:k])
    sb = set(int(t) for t in b[:k])
    if not sa and not sb:
        return float("nan")
    return float(len(sa & sb) / len(sa | sb))


def rank_displacement(before: Sequence[int], after: Sequence[int]) -> list[int]:
    """Signed position change per token (``after`` minus ``before``)."""
    pos_before = {int(t): i for i, t in enumerate(before)}
    out: list[int] = []
    for j, tok in enumerate(after):
        i = pos_before.get(int(tok))
        if i is not None:
            out.append(j - i)
    return out


def displacement_summary(before: Sequence[int], after: Sequence[int]) -> dict[str, float]:
    """Mean/max absolute displacement and moved fraction between two encodings."""
    disp = rank_displacement(before, after)
    if not disp:
        return {
            "mean_abs_displacement": float("nan"),
            "max_abs_displacement": float("nan"),
            "frac_moved": float("nan"),
        }
    abs_disp = [abs(d) for d in disp]
    return {
        "mean_abs_displacement": float(sum(abs_disp) / len(abs_disp)),
        "max_abs_displacement": float(max(abs_disp)),
        "frac_moved": float(sum(1 for d in disp if d != 0) / len(disp)),
    }


def is_two_cycle(
    prev: Sequence[int] | None,
    current: Sequence[int],
    nxt: Sequence[int],
) -> bool:
    """True when ``nxt`` returns to ``prev`` while differing from ``current``."""
    if prev is None:
        return False
    return list(map(int, nxt)) == list(map(int, prev)) and list(map(int, nxt)) != list(
        map(int, current)
    )


def gap_closed_fraction(
    baseline: float,
    method: float,
    ceiling: float,
) -> float:
    """Fraction of the baseline-to-ceiling gap that ``method`` closes."""
    span = float(ceiling) - float(baseline)
    if span == 0.0:
        return float("nan")
    return float((float(method) - float(baseline)) / span)


# --- direction-fidelity metrics -------------------------------------------------
#
# Everything below scores a predicted rank displacement against an observed one.
# Sign convention throughout: displacement is in normalized rank units where
# **negative = moved left = higher expression rank = "up"**, matching
# ``rerank.base_rank_norm``. So "top up-movers" are the most negative values.


def median_iqr(values: Sequence[float]) -> dict[str, float]:
    """Median and interquartile range, ignoring NaN."""
    vals = sorted(float(v) for v in values if float(v) == float(v))
    n = len(vals)
    if n == 0:
        return {"n": 0.0, "median": float("nan"), "q25": float("nan"), "q75": float("nan"), "iqr": float("nan")}

    def _q(p: float) -> float:
        if n == 1:
            return vals[0]
        pos = p * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return vals[lo] * (1.0 - frac) + vals[hi] * frac

    q25, q75 = _q(0.25), _q(0.75)
    return {"n": float(n), "median": _q(0.5), "q25": q25, "q75": q75, "iqr": q75 - q25}


def _top_k_indices(scores: Sequence[float], k: int, *, up: bool) -> list[int]:
    """Indices of the ``k`` most extreme scores (``up`` -> most negative)."""
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=not up)
    return order[: max(0, int(k))]


def precision_at_k(
    pred: Sequence[float],
    obs: Sequence[float],
    k: int,
    *,
    up: bool,
) -> float:
    """Overlap of predicted and observed top-``k`` movers, divided by ``k``.

    With both sets of size ``k`` this equals the top-``k`` overlap fraction, so it
    is reported once rather than under two names.
    """
    if len(pred) != len(obs):
        raise ValueError("precision_at_k requires equal-length inputs")
    k = int(k)
    if k <= 0 or len(pred) < k:
        return float("nan")
    a = set(_top_k_indices(pred, k, up=up))
    b = set(_top_k_indices(obs, k, up=up))
    return float(len(a & b) / k)


def ndcg_at_k(
    pred: Sequence[float],
    obs: Sequence[float],
    k: int,
    *,
    up: bool,
) -> float:
    """NDCG@k with graded relevance = displacement magnitude in the chosen direction.

    Relevance is ``max(0, -obs)`` for ``up`` and ``max(0, obs)`` for down, so genes
    that did not move in the queried direction contribute no gain.
    """
    if len(pred) != len(obs):
        raise ValueError("ndcg_at_k requires equal-length inputs")
    k = int(k)
    if k <= 0 or len(pred) < k:
        return float("nan")
    rel = [max(0.0, -float(o)) if up else max(0.0, float(o)) for o in obs]
    ranked = _top_k_indices(pred, k, up=up)
    dcg = sum(rel[i] / _log2(rank + 2) for rank, i in enumerate(ranked))
    ideal = sorted(rel, reverse=True)[:k]
    idcg = sum(r / _log2(rank + 2) for rank, r in enumerate(ideal))
    if idcg <= 0.0:
        return float("nan")
    return float(dcg / idcg)


def _log2(x: float) -> float:
    from math import log2

    return log2(x)


def calibration_bins(
    pred: Sequence[float],
    obs: Sequence[float],
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Mean observed displacement per quantile bin of predicted displacement.

    A well-calibrated decoder gives bin means that increase monotonically with the
    bin index; ``calibration_monotonicity`` reduces that to one number.
    """
    if len(pred) != len(obs):
        raise ValueError("calibration_bins requires equal-length inputs")
    n = len(pred)
    if n == 0 or n_bins <= 0:
        return []
    order = sorted(range(n), key=lambda i: float(pred[i]))
    bins: list[dict[str, float]] = []
    for b in range(int(n_bins)):
        lo = (b * n) // int(n_bins)
        hi = ((b + 1) * n) // int(n_bins)
        idx = order[lo:hi]
        if not idx:
            continue
        bins.append(
            {
                "bin": float(b),
                "n": float(len(idx)),
                "pred_mean": float(sum(float(pred[i]) for i in idx) / len(idx)),
                "obs_mean": float(sum(float(obs[i]) for i in idx) / len(idx)),
            }
        )
    return bins


def calibration_monotonicity(bins: Sequence[Mapping[str, float]]) -> float:
    """Spearman between bin index and mean observed displacement."""
    if len(bins) < 2:
        return float("nan")
    return spearman_values(
        [float(b["bin"]) for b in bins], [float(b["obs_mean"]) for b in bins]
    )


def bootstrap_delta_rho(
    groups: Sequence[int],
    pred_a: Sequence[float],
    pred_b: Sequence[float],
    obs: Sequence[float],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, float]:
    """Bootstrap CI for ``rho(pred_a, obs) - rho(pred_b, obs)``, resampling ``groups``.

    Resampling is by group (cell), not by sample, because genes within a cell share
    one encoding and are not independent.
    """
    import random

    by_group: dict[int, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(int(g), []).append(i)
    keys = list(by_group)
    point = spearman_values(pred_a, obs) - spearman_values(pred_b, obs)
    undefined = {
        "delta_rho": float(point),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "n_boot": 0.0,
        "excludes_zero": float("nan"),
    }
    if len(keys) < 2:
        return undefined

    rng = random.Random(int(seed))
    draws: list[float] = []
    for _ in range(int(n_boot)):
        idx: list[int] = []
        for _ in range(len(keys)):
            idx.extend(by_group[keys[rng.randrange(len(keys))]])
        if len(idx) < 2:
            continue
        a = [pred_a[i] for i in idx]
        b = [pred_b[i] for i in idx]
        o = [obs[i] for i in idx]
        d = spearman_values(a, o) - spearman_values(b, o)
        if d == d:
            draws.append(d)
    if len(draws) < 2:
        return {**undefined, "n_boot": float(len(draws))}
    draws.sort()
    alpha = (1.0 - float(ci)) / 2.0
    lo = draws[max(0, int(alpha * len(draws)) - 1)]
    hi = draws[min(len(draws) - 1, int((1.0 - alpha) * len(draws)))]
    return {
        "delta_rho": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": float(len(draws)),
        "excludes_zero": float(1.0 if (lo > 0.0 or hi < 0.0) else 0.0),
    }


def paired_sign_flip_test(
    diffs: Sequence[float],
    *,
    n_perm: int = 10000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired permutation test on per-group score differences.

    Under H0 the two methods are equally aligned with the observation, so the sign
    of each group's difference is exchangeable. Reports the two-sided p-value
    alongside the effect size, which is the quantity to lead with.
    """
    import random

    vals = [float(d) for d in diffs if float(d) == float(d)]
    n = len(vals)
    if n == 0:
        return {"n": 0.0, "mean_diff": float("nan"), "p_value": float("nan")}
    observed = sum(vals) / n
    rng = random.Random(int(seed))
    extreme = 0
    for _ in range(int(n_perm)):
        total = 0.0
        for v in vals:
            total += v if rng.random() < 0.5 else -v
        if abs(total / n) >= abs(observed) - 1e-15:
            extreme += 1
    return {
        "n": float(n),
        "mean_diff": float(observed),
        "median_diff": median_iqr(vals)["median"],
        "frac_positive": float(sum(1 for v in vals if v > 0) / n),
        "p_value": float((extreme + 1) / (int(n_perm) + 1)),
    }
