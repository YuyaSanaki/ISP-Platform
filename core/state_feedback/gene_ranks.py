"""Pseudobulk gene ranks of Start, Goal and ISP-predicted encodings (rank enrichment).

Each cell's encoding gives a gene the normalized rank ``position / (length - 1)``
(0 = first, highest expressed relative to the corpus median; 1 = last). A gene missing
from a cell's encoding gets 1, the bottom. The pseudobulk rank of a gene is the mean over
all cells of a state, so it reflects both detection and position.

ISP-predicted ranks come from the reranked encodings of State-feedback ISP with the
potential-outcome decoder. Per gene, the displacement of the configured chain
(final minus start, mean over all start cells) minus the mean displacement under the
evaluation placebo chains is the gene-level Delta-spec; the predicted rank is the Start
rank plus Delta-spec. A placebo chain that contains the gene is left out of its mean.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

POSITION_COLUMNS = ["token", "sum_norm_pos", "n_cells"]


def position_sums(rows: Iterable[Sequence[int]]) -> tuple[pd.DataFrame, int]:
    """``(token, sum_norm_pos, n_cells)`` over the cells where the token is present, and N cells."""
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    n = 0
    for ids in rows:
        n += 1
        ids = [int(t) for t in ids]
        denom = max(1, len(ids) - 1)
        for pos, tok in enumerate(ids):
            sums[tok] = sums.get(tok, 0.0) + pos / denom
            counts[tok] = counts.get(tok, 0) + 1
    df = pd.DataFrame(
        {"token": list(sums), "sum_norm_pos": list(sums.values()),
         "n_cells": [counts[t] for t in sums]},
        columns=POSITION_COLUMNS,
    )
    return df.sort_values("token").reset_index(drop=True), n


def pseudobulk_rank(sums: pd.DataFrame, n_cells: int) -> pd.Series:
    """Mean normalized rank over all ``n_cells`` (absent = 1), indexed by token."""
    s = sums.set_index("token")
    n = max(int(n_cells), 1)
    return ((s["sum_norm_pos"] + (n - s["n_cells"])) / n).rename("rank")


def mean_displacement(start: pd.DataFrame, final: pd.DataFrame, n_cells: int) -> pd.Series:
    """Mean over all cells of (final rank - start rank) per token, absent = 1 on both sides."""
    tokens = pd.Index(start["token"]).union(pd.Index(final["token"]))
    r0 = pseudobulk_rank(start, n_cells).reindex(tokens, fill_value=1.0)
    r1 = pseudobulk_rank(final, n_cells).reindex(tokens, fill_value=1.0)
    return (r1 - r0).rename("delta")


def gene_delta_spec(
    start: pd.DataFrame,
    configured: pd.DataFrame,
    placebos: Mapping[str, pd.DataFrame],
    placebo_genes: Mapping[str, set[int]],
    n_cells: int,
) -> pd.DataFrame:
    """Per token: configured displacement, placebo mean displacement, Delta-spec."""
    conf = mean_displacement(start, configured, n_cells)
    tokens = conf.index
    total = pd.Series(0.0, index=tokens)
    used = pd.Series(0, index=tokens)
    for chain, final in placebos.items():
        d = mean_displacement(start, final, n_cells).reindex(tokens, fill_value=0.0)
        keep = ~tokens.isin(list(placebo_genes.get(chain, set())))
        total[keep] += d[keep]
        used[keep] += 1
    placebo_mean = (total / used.where(used > 0)).rename("delta_placebo_mean")
    out = pd.DataFrame({
        "delta_configured": conf,
        "delta_placebo_mean": placebo_mean,
        "n_placebos": used,
    })
    out["delta_spec"] = out["delta_configured"] - out["delta_placebo_mean"]
    out.index.name = "token"
    return out


def background_tokens(
    start: pd.DataFrame,
    goal: pd.DataFrame,
    *,
    min_detection: int,
    rule: str = "both",
    exclude: Iterable[int] = (),
) -> list[int]:
    """Tokens detected in at least ``min_detection`` cells of Start and Goal (``both``) or of either."""
    s = set(start.loc[start["n_cells"] >= int(min_detection), "token"].astype(int))
    g = set(goal.loc[goal["n_cells"] >= int(min_detection), "token"].astype(int))
    if rule == "both":
        keep = s & g
    elif rule == "either":
        keep = s | g
    else:
        raise ValueError("rank_enrichment.background must be 'both' or 'either'")
    return sorted(keep - {int(t) for t in exclude})


def top_fraction(scores: pd.Series, fraction: float) -> list[int]:
    """Tokens with the lowest rank (front of the encoding), ``ceil(fraction * n)`` of them."""
    if not 0 < float(fraction) <= 1:
        raise ValueError("rank_enrichment.top_fraction must be in (0, 1]")
    s = scores.dropna()
    k = math.ceil(float(fraction) * len(s))
    df = pd.DataFrame({"token": s.index.astype(int), "v": s.to_numpy(dtype=float)})
    df = df.sort_values(["v", "token"], kind="mergesort")
    return [int(t) for t in df["token"].iloc[:k]]


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else float("nan")


def rank_table(
    background: Sequence[int],
    start_rank: pd.Series,
    goal_rank: pd.Series,
    delta: pd.DataFrame,
) -> pd.DataFrame:
    """Start, Goal and ISP-predicted pseudobulk ranks on the background tokens."""
    idx = pd.Index([int(t) for t in background], name="token")
    out = pd.DataFrame(index=idx)
    out["start_rank"] = start_rank.reindex(idx, fill_value=1.0)
    out["goal_rank"] = goal_rank.reindex(idx, fill_value=1.0)
    d = delta.reindex(idx)
    for col in ("delta_configured", "delta_placebo_mean", "delta_spec"):
        out[col] = d[col].fillna(0.0) if col in d else 0.0
    out["isp_rank"] = out["start_rank"] + out["delta_spec"]
    return out
