"""Fixed-gene-set reranking: priority convention, guardrails, Phase 1 scorers.

**Priority convention (used everywhere in this package):** a per-gene scalar
where **lower = leftmost = higher rank**. This matches ``base_rank_norm =
position / (n - 1)`` and the ascending sort in
``oracle_rerank.rerank_fixed_gene_set``.

A scorer therefore *lowers* the priority of genes it wants to promote.
"""
from __future__ import annotations

from typing import Sequence

PRIORITY_CONVENTION = "ascending = leftmost (higher rank)"


def base_rank_norm(n_genes: int) -> list[float]:
    """Normalized base positions in ``[0, 1]`` for a length-``n`` encoding."""
    if n_genes <= 1:
        return [0.0] * max(0, n_genes)
    denom = float(n_genes - 1)
    return [i / denom for i in range(n_genes)]


def quantize(value: float, epsilon: float) -> float:
    """Snap to an ``epsilon`` grid so near-equal priorities become exact ties."""
    if epsilon <= 0.0:
        return float(value)
    return round(float(value) / epsilon) * epsilon


def apply_priority_order(
    input_ids: Sequence[int],
    priority: Sequence[float],
    *,
    hysteresis: float = 0.0,
) -> list[int]:
    """Permute ``input_ids`` by ``priority`` (ascending), preserving the gene set.

    ``hysteresis`` implements the swap-hysteresis guardrail: priorities are
    snapped to an ``epsilon`` grid before a **stable** sort, so genes whose
    scores differ by less than ``epsilon`` keep their original relative order.
    """
    ids = [int(t) for t in input_ids]
    if len(priority) != len(ids):
        raise ValueError(f"priority length {len(priority)} != input_ids length {len(ids)}")
    keyed = sorted(
        range(len(ids)),
        key=lambda i: (quantize(priority[i], hysteresis), i),
    )
    return [ids[i] for i in keyed]


def bounded_priority(
    base: Sequence[float],
    signed_shift: Sequence[float],
    *,
    max_shift: float,
) -> list[float]:
    """``base + clamp(signed_shift, +/- max_shift)`` in normalized rank units."""
    if len(base) != len(signed_shift):
        raise ValueError("base and signed_shift must have equal length")
    lo, hi = -abs(max_shift), abs(max_shift)
    return [float(b) + min(hi, max(lo, float(s))) for b, s in zip(base, signed_shift)]


def _zscore(values: Sequence[float]) -> list[float]:
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return []
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    sd = var**0.5
    if sd <= 0.0:
        return [0.0] * n
    return [(v - mean) / sd for v in vals]


def norm_priority(
    delta_norms: Sequence[float],
    *,
    alpha: float,
    max_shift: float,
) -> list[float]:
    """Null baseline: rerank by change in hidden-state norm.

    A larger norm increase moves a gene left. There is no theory linking
    hidden-state norm to expression rank; this exists to show that an
    arbitrary monotone readout does **not** reproduce the oracle.
    """
    base = base_rank_norm(len(delta_norms))
    shift = [-alpha * z for z in _zscore(delta_norms)]
    return bounded_priority(base, shift, max_shift=max_shift)


def delta_mlm_priority(
    delta_self_logits: Sequence[float],
    *,
    alpha: float,
    max_shift: float,
) -> list[float]:
    """Parameter-free differential baseline: delta-MLM self-logit plus inertia.

    ``delta_self_logits[i] = logit_pert(token_i) - logit_ctrl(token_i)``. A token
    that becomes *more* plausible after perturbation moves left. ``base_rank_norm``
    is the inertia term. This measures token-identity plausibility change, which
    is not the same quantity as an expression-rank change, so it is a comparator
    rather than the primary decoder.
    """
    base = base_rank_norm(len(delta_self_logits))
    shift = [-alpha * z for z in _zscore(delta_self_logits)]
    return bounded_priority(base, shift, max_shift=max_shift)
