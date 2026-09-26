"""Oracle fixed-gene-set reranking for State-feedback Phase 0.

Oracle = use an *observed* target ranking (e.g. pluripotent endpoint tokens)
to permute the current cell's gene set. Gene membership is unchanged; only
order changes. Absolute expression values are not required — only a total
order over tokens (here: mean position in observed cells).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence


def build_pseudobulk_rank_priority(
    input_ids_rows: Iterable[Sequence[int]],
) -> dict[int, float]:
    """Mean left-to-right position per token across observed cells.

    Lower priority = more highly expressed (leftmost) in the observed state.
    Tokens never seen get no entry (caller keeps relative order via tie-break).
    """
    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    for ids in input_ids_rows:
        for pos, tok in enumerate(ids):
            t = int(tok)
            sums[t] += float(pos)
            counts[t] += 1
    return {t: sums[t] / counts[t] for t in sums}


def rerank_fixed_gene_set(
    input_ids: Sequence[int],
    token_priority: Mapping[int, float],
) -> list[int]:
    """Permute ``input_ids`` by ``token_priority`` (ascending = leftmost).

    Tokens missing from ``token_priority`` keep a large priority so they sink
    rightward while preserving their relative order (stable sort by
    (priority, original_index)).
    """
    ids = [int(t) for t in input_ids]
    n = len(ids)
    missing = float(n) + 1.0

    def sort_key(item: tuple[int, int]) -> tuple[float, int]:
        idx, tok = item
        return (float(token_priority.get(tok, missing)), idx)

    ordered = sorted(enumerate(ids), key=sort_key)
    return [tok for _, tok in ordered]


def spearman_rank_correlation(a: Sequence[int], b: Sequence[int]) -> float:
    """Spearman ρ between two permutations of the same multiset (by position).

    Builds rank-by-token from list ``a`` (first occurrence) and compares to
    positions in ``b`` for the shared unique token order of ``a``. Returns
    0.0 if fewer than 2 unique tokens.
    """
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    # Position of each token in a (first wins if duplicates — Geneformer IDs are unique per cell)
    pos_a = {int(t): i for i, t in enumerate(a)}
    pos_b = {int(t): i for i, t in enumerate(b)}
    shared = [t for t in pos_a if t in pos_b]
    if len(shared) < 2:
        return 0.0
    xa = [pos_a[t] for t in shared]
    xb = [pos_b[t] for t in shared]
    n = len(shared)
    # Rank data are already positions; Pearson on ranks = Spearman
    mean_a = sum(xa) / n
    mean_b = sum(xb) / n
    num = sum((xa[i] - mean_a) * (xb[i] - mean_b) for i in range(n))
    den_a = sum((x - mean_a) ** 2 for x in xa) ** 0.5
    den_b = sum((x - mean_b) ** 2 for x in xb) ** 0.5
    if den_a == 0.0 or den_b == 0.0:
        return 0.0
    return float(num / (den_a * den_b))
