"""Metrics for State-feedback ISP: rank correlation, overlap, drift."""
from __future__ import annotations

from typing import Sequence


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
