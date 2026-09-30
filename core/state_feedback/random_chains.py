"""Matched random control chains for State-feedback ISP.

What overexpression does to an encoding depends on where the gene already is in the
start cells: an absent gene is inserted at the front (every other gene shifts down by
one), a gene already near the front barely moves. A random chain is only a control if
its genes get the same kind of edit, so each configured gene is replaced by a gene
drawn uniformly from its *stratum*:

* population: every gene token in the model vocabulary except special tokens and the
  configured genes (not only genes that pass the teacher's detection filter);
* stratum of a configured gene with start-cell detection ``d``:
  ``d == 0`` -> genes absent from every start cell;
  ``d > 0``  -> genes with detection in ``[d / 2, 2 d]`` whose median normalized
  position when present is within ``rank_window`` of the configured gene's
  (the window widens if fewer than ``min_stratum`` genes qualify; logged).

A random chain keeps the configured chain's structure (steps, genes per step, types).
Genes are drawn per configured gene in token order, so the same seed gives the same
random genes whatever the step order, and without replacement across chains while the
stratum lasts. ``describe`` records every pick with its profile so the draw can be
checked against its stratum (balance) before any result is read.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

RANK_WINDOWS = (0.1, 0.2, 0.3, None)


def start_profiles(rows: Iterable[Sequence[int]]) -> tuple[dict[int, float], dict[int, float]]:
    """Per token: fraction of cells containing it, median normalized position when present."""
    counts: dict[int, int] = {}
    positions: dict[int, list[float]] = {}
    n_cells = 0
    for ids in rows:
        n_cells += 1
        denom = max(1, len(ids) - 1)
        for pos, tok in enumerate(ids):
            t = int(tok)
            counts[t] = counts.get(t, 0) + 1
            positions.setdefault(t, []).append(pos / denom)
    n = max(1, n_cells)
    return (
        {t: c / n for t, c in counts.items()},
        {t: float(np.median(p)) for t, p in positions.items()},
    )


def stratum(
    token: int,
    population: Sequence[int],
    detection: Mapping[int, float],
    rank: Mapping[int, float],
    *,
    min_stratum: int = 50,
) -> tuple[list[int], dict[str, Any]]:
    """Population genes whose start-cell position profile matches ``token``."""
    d = float(detection.get(int(token), 0.0))
    if d == 0.0:
        pool = [int(t) for t in population if detection.get(int(t), 0.0) == 0.0]
        return pool, {"detection": 0.0, "rule": "absent", "rank_window": None, "size": len(pool)}
    lo, hi = d / 2.0, d * 2.0
    by_det = [int(t) for t in population if lo <= detection.get(int(t), 0.0) <= hi]
    r = float(rank[int(token)])
    pool: list[int] = []
    for window in RANK_WINDOWS:
        pool = by_det if window is None else [t for t in by_det if abs(rank[t] - r) <= window]
        if len(pool) >= min_stratum:
            break
    return pool, {
        "detection": d, "rank": r, "rule": "detection_x0.5-2_and_rank",
        "rank_window": window, "size": len(pool),
    }


def draw_random_chains(
    steps: Sequence[Mapping[str, Any]],
    token_by_step: Sequence[Sequence[int]],
    population: Sequence[int],
    detection: Mapping[int, float],
    rank: Mapping[int, float],
    *,
    n_chains: int,
    seed: int,
    min_stratum: int = 50,
    prefix: str = "random",
) -> tuple[dict[str, tuple[list[dict[str, Any]], list[list[int]]]], dict[str, Any]]:
    """``n_chains`` structure- and position-matched random chains plus the draw record."""
    configured = sorted({int(t) for ts in token_by_step for t in ts})
    population = sorted(set(int(t) for t in population) - set(configured))
    rng = np.random.default_rng(int(seed))
    strata: dict[int, dict[str, Any]] = {}
    picks: dict[int, list[int]] = {}
    for tok in configured:
        pool, info = stratum(tok, population, detection, rank, min_stratum=min_stratum)
        if not pool:
            raise ValueError(f"no population gene matches configured token {tok}: {info}")
        replace = len(pool) < int(n_chains)
        info["with_replacement"] = replace
        picks[tok] = [int(t) for t in rng.choice(pool, size=int(n_chains), replace=replace)]
        strata[tok] = {**info, "pool": pool}
    chains: dict[str, tuple[list[dict[str, Any]], list[list[int]]]] = {}
    for r in range(int(n_chains)):
        tbs = [[picks[int(t)][r] for t in ts] for ts in token_by_step]
        chains[f"{prefix}{r}"] = (
            [
                {"index": i + 1, "name": f"{prefix}{r}_s{i + 1}", "type": s["type"],
                 "genes": [str(t) for t in toks]}
                for i, (s, toks) in enumerate(zip(steps, tbs))
            ],
            tbs,
        )
    record = {
        "seed": int(seed), "n_chains": int(n_chains), "min_stratum": int(min_stratum),
        "population_size": len(population), "strata": strata, "picks": picks,
    }
    return chains, record


def standardized_mean_difference(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray([x for x in a if x == x], dtype=np.float64)
    b = np.asarray([x for x in b if x == x], dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0


def describe(
    record: Mapping[str, Any],
    covariates: Mapping[str, Mapping[int, float]],
    names: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Pick table and balance of the picks against their stratum.

    ``covariates`` maps a name (e.g. ``goal_detection``, ``teacher_delta``) to a
    per-token value; missing tokens count as NaN. Matching makes start-cell
    detection and rank balanced by construction; the other covariates show
    whether the draw is a fair sample of the stratum (|SMD| small).
    """
    names = names or {}
    genes = []
    balance = []
    for tok, info in record["strata"].items():
        tok = int(tok)
        chosen = record["picks"][tok]
        pool = info["pool"]
        row = {"configured_token": tok, "configured_gene": names.get(tok, str(tok)),
               **{k: v for k, v in info.items() if k != "pool"}}
        for cov, values in covariates.items():
            p = [float(values.get(t, float("nan"))) for t in chosen]
            s = [float(values.get(t, float("nan"))) for t in pool]
            row[f"{cov}_configured"] = float(values.get(tok, float("nan")))
            row[f"{cov}_picks_mean"] = float(np.nanmean(p)) if np.isfinite(p).any() else float("nan")
            row[f"{cov}_stratum_mean"] = float(np.nanmean(s)) if np.isfinite(s).any() else float("nan")
            row[f"{cov}_smd"] = standardized_mean_difference(p, s)
        balance.append(row)
        for r, t in enumerate(chosen):
            genes.append({
                "chain": r, "configured_token": tok, "configured_gene": names.get(tok, str(tok)),
                "token": int(t), "gene": names.get(int(t), str(t)),
                **{cov: float(values.get(int(t), float("nan"))) for cov, values in covariates.items()},
            })
    n_unique = len({g["token"] for g in genes})
    return {"picks": genes, "balance": balance, "n_picks": len(genes), "n_unique_picks": n_unique}
