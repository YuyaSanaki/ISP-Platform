"""Direction-fidelity evaluation: does a method recover the *observed* Δrank?

The endpoint classifier shift cannot settle this. It responds to whether the
encoding was disturbed at least as much as to which direction it was permuted, so
a theory-free baseline can match the learned decoder on it. This module scores
every method against the same teacher, on the same held-out gene split, over the
same (cell, gene) samples, in the same normalized rank units.

Methods and their intended role:

| Method | Input | Training | Role |
|--------|-------|----------|------|
| ``identity`` | — | none | strict null (Δr̂ = 0) |
| ``random`` | rank-independent noise | none | permutation null |
| ``norm`` | Δ‖h‖₂ | none | unsupervised scalar representation baseline |
| ``delta_mlm`` | Δ self-logit | none | pretrained-head differential baseline |
| ``base_rank`` | r_base only | cross-fitted bin means | base-rank control |
| ``linear_deltarank`` | [Δh, r_base] | supervised | primary method |

Every method also reports its Spearman after removing base rank
(``partial_spearman_given_base``); see ``state_feedback.controls``.

Sign convention: negative displacement = moved left = higher expression rank.
"""
from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

import torch

from state_feedback.controls import (
    base_rank_control,
    crossfit_base_only,
    partial_spearman_given_base,
)
from state_feedback.metrics import (
    bootstrap_delta_rho,
    calibration_bins,
    calibration_monotonicity,
    median_iqr,
    ndcg_at_k,
    paired_sign_flip_test,
    precision_at_k,
    spearman_values,
)
from state_feedback.samples import EvalSamples

NULL_METHODS = ("identity", "random")
BASELINE_METHODS = ("norm", "delta_mlm")
BASE_RANK_METHOD = "base_rank"
PRIMARY_METHOD = "linear_deltarank"
DEFAULT_TOPK = (50, 100, 500)


def _zscore_within_cells(
    values: Sequence[float],
    cell_index: Sequence[int],
) -> list[float]:
    """Z-score each cell's genes separately.

    Ranks are cell-specific relative quantities, so a global z-score would let
    cells with different sequence lengths dominate one another (guardrail 7).
    """
    by_cell: dict[int, list[int]] = {}
    for i, c in enumerate(cell_index):
        by_cell.setdefault(int(c), []).append(i)
    out = [0.0] * len(values)
    for idx in by_cell.values():
        vals = [float(values[i]) for i in idx]
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        sd = var**0.5
        for i, v in zip(idx, vals):
            out[i] = 0.0 if sd <= 0.0 else (v - mean) / sd
    return out


def predicted_delta_rank(
    method: str,
    samples: EvalSamples,
    *,
    alpha: float,
    max_shift: float,
    decoder=None,
    seed: int = 0,
    bound: bool = True,
) -> list[float]:
    """Δr̂ for one method, in normalized rank units.

    Baselines use the same shift rule as the decoder (``-alpha * z``, optionally
    clamped to ``±max_shift``), so only the scalar being read differs.

    ``bound=False`` for scoring. The clamp is a deployment guardrail, not part of
    the scoring function, and it is **not** rank-preserving for the z-score
    baselines: with ``alpha=1`` and ``max_shift=0.1`` roughly 92% of a unit-normal
    z lands outside the bound, collapsing the baseline to a two-valued sign
    predictor and handicapping it against the decoder. The decoder's ``tanh`` bound
    is monotone in its pre-activation, so dropping ``max_shift`` leaves its ranking
    untouched and the comparison symmetric.
    """
    n = len(samples)
    if method == "identity":
        return [0.0] * n
    if method == "random":
        rng = random.Random(int(seed))
        return [rng.uniform(-max_shift, max_shift) for _ in range(n)]
    if method == "norm":
        z = _zscore_within_cells(samples.delta_norm.tolist(), samples.cell_index)
        return [_maybe_clamp(-alpha * v, max_shift, bound) for v in z]
    if method == "delta_mlm":
        if samples.delta_self_logit is None:
            raise ValueError("delta_mlm requires self-logit features (pass mlm_model)")
        z = _zscore_within_cells(samples.delta_self_logit.tolist(), samples.cell_index)
        return [_maybe_clamp(-alpha * v, max_shift, bound) for v in z]
    if method == BASE_RANK_METHOD:
        pred = crossfit_base_only(samples.base_rank.tolist(), samples.target.tolist(), samples.tokens)
        return [_maybe_clamp(v, max_shift, bound) for v in pred.tolist()]
    if method == PRIMARY_METHOD:
        if decoder is None:
            raise ValueError("linear_deltarank requires a trained decoder")
        dev = next(decoder.parameters()).device
        with torch.no_grad():
            pred = decoder.delta_rank(samples.delta_h.to(dev))
        return pred.detach().cpu().tolist()
    raise ValueError(f"Unknown method {method!r}")


def _maybe_clamp(value: float, bound: float, apply: bool) -> float:
    if not apply:
        return float(value)
    b = abs(float(bound))
    return max(-b, min(b, float(value)))


def _cellwise_spearman(
    pred: Sequence[float],
    obs: Sequence[float],
    cell_index: Sequence[int],
    *,
    min_genes: int = 10,
) -> list[float]:
    """Spearman within each cell: did we recover that cell's local permutation?"""
    by_cell: dict[int, list[int]] = {}
    for i, c in enumerate(cell_index):
        by_cell.setdefault(int(c), []).append(i)
    out: list[float] = []
    for idx in by_cell.values():
        if len(idx) < int(min_genes):
            continue
        rho = spearman_values([pred[i] for i in idx], [obs[i] for i in idx])
        if rho == rho:
            out.append(rho)
    return out


def _gene_aggregate(
    pred: Sequence[float],
    obs: Sequence[float],
    tokens: Sequence[int],
) -> tuple[list[int], list[float], list[float]]:
    """Mean Δr̂ per gene against that gene's teacher value.

    Note on what is *not* computable here: a within-gene / across-cells Spearman is
    undefined with a pseudobulk teacher, because the target is one constant per
    gene and has zero variance across cells. Gene-level agreement is therefore
    measured by aggregating predictions per gene, and gene-identity memorization is
    controlled by the held-out gene split itself.
    """
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    target: dict[int, float] = {}
    for p, o, t in zip(pred, obs, tokens):
        tok = int(t)
        sums[tok] = sums.get(tok, 0.0) + float(p)
        counts[tok] = counts.get(tok, 0) + 1
        target[tok] = float(o)
    toks = sorted(sums)
    return toks, [sums[t] / counts[t] for t in toks], [target[t] for t in toks]


def evaluate_method(
    method: str,
    pred: Sequence[float],
    samples: EvalSamples,
    *,
    topk: Sequence[int] = DEFAULT_TOPK,
    n_calibration_bins: int = 10,
) -> dict[str, Any]:
    """Full metric battery for one method's Δr̂."""
    obs = samples.target.tolist()
    row: dict[str, Any] = {
        "method": method,
        "n_samples": len(pred),
        "pooled_spearman": spearman_values(pred, obs),
        "partial_spearman_given_base": partial_spearman_given_base(
            pred, obs, samples.base_rank.tolist()
        ),
    }

    cell_rhos = _cellwise_spearman(pred, obs, samples.cell_index)
    cell_stats = median_iqr(cell_rhos)
    row["n_cells_scored"] = cell_stats["n"]
    row["cellwise_spearman_median"] = cell_stats["median"]
    row["cellwise_spearman_q25"] = cell_stats["q25"]
    row["cellwise_spearman_q75"] = cell_stats["q75"]
    row["cellwise_spearman_iqr"] = cell_stats["iqr"]

    toks, gene_pred, gene_obs = _gene_aggregate(pred, obs, samples.tokens)
    row["n_genes_scored"] = float(len(toks))
    row["gene_aggregated_spearman"] = spearman_values(gene_pred, gene_obs)

    for k in topk:
        row[f"precision_at_{k}_up"] = precision_at_k(gene_pred, gene_obs, k, up=True)
        row[f"precision_at_{k}_down"] = precision_at_k(gene_pred, gene_obs, k, up=False)
        row[f"ndcg_at_{k}_up"] = ndcg_at_k(gene_pred, gene_obs, k, up=True)
        row[f"ndcg_at_{k}_down"] = ndcg_at_k(gene_pred, gene_obs, k, up=False)

    bins = calibration_bins(pred, obs, n_calibration_bins)
    row["calibration_monotonicity"] = calibration_monotonicity(bins)
    row["_calibration_bins"] = bins
    row["_cellwise_spearman"] = cell_rhos
    return row


def compare_methods(
    samples: EvalSamples,
    *,
    alpha: float,
    max_shift: float,
    decoder=None,
    methods: Sequence[str] | None = None,
    topk: Sequence[int] = DEFAULT_TOPK,
    n_boot: int = 1000,
    n_perm: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Score every method on the shared sample set and contrast against baselines."""
    if methods is None:
        methods = [*NULL_METHODS, *BASELINE_METHODS, BASE_RANK_METHOD, PRIMARY_METHOD]
    available = [
        m
        for m in methods
        if not (m == "delta_mlm" and samples.delta_self_logit is None)
        and not (m == PRIMARY_METHOD and decoder is None)
    ]
    skipped = [m for m in methods if m not in available]

    preds: dict[str, list[float]] = {}
    rows: list[dict[str, Any]] = []
    for method in available:
        pred = predicted_delta_rank(
            method,
            samples,
            alpha=alpha,
            max_shift=max_shift,
            decoder=decoder,
            seed=seed,
            bound=False,
        )
        preds[method] = pred
        rows.append(evaluate_method(method, pred, samples, topk=topk))

    contrasts: list[dict[str, Any]] = []
    if PRIMARY_METHOD in preds:
        obs = samples.target.tolist()
        for other in (*BASELINE_METHODS, BASE_RANK_METHOD, *NULL_METHODS):
            if other not in preds:
                continue
            boot = bootstrap_delta_rho(
                samples.cell_index,
                preds[PRIMARY_METHOD],
                preds[other],
                obs,
                n_boot=n_boot,
                seed=seed,
            )
            a = _cellwise_spearman_map(preds[PRIMARY_METHOD], obs, samples.cell_index)
            b = _cellwise_spearman_map(preds[other], obs, samples.cell_index)
            diffs = [a[c] - b[c] for c in a if c in b]
            perm = paired_sign_flip_test(diffs, n_perm=n_perm, seed=seed)
            contrasts.append(
                {
                    "primary": PRIMARY_METHOD,
                    "baseline": other,
                    **{f"boot_{k}": v for k, v in boot.items()},
                    **{f"perm_{k}": v for k, v in perm.items()},
                }
            )

    result: dict[str, Any] = {"methods": rows, "contrasts": contrasts, "skipped": skipped}
    if PRIMARY_METHOD in preds:
        extra = {m: preds[m] for m in BASELINE_METHODS if m in preds}
        result["base_rank_control"] = base_rank_control(
            samples, decoder, extra=extra, n_boot=n_boot, seed=seed
        )
    return result


def _cellwise_spearman_map(
    pred: Sequence[float],
    obs: Sequence[float],
    cell_index: Sequence[int],
    *,
    min_genes: int = 10,
) -> dict[int, float]:
    by_cell: dict[int, list[int]] = {}
    for i, c in enumerate(cell_index):
        by_cell.setdefault(int(c), []).append(i)
    out: dict[int, float] = {}
    for cell, idx in by_cell.items():
        if len(idx) < int(min_genes):
            continue
        rho = spearman_values([pred[i] for i in idx], [obs[i] for i in idx])
        if rho == rho:
            out[cell] = rho
    return out


def direction_fidelity_verdict(result: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce the comparison to the Phase 2 direction-fidelity axis.

    Passing requires the primary decoder to (1) beat *both* parameter-free
    baselines with a bootstrap CI on Δρ that excludes zero, and (2) add signal
    beyond base rank: its partial Spearman given base rank must have a CI above
    zero. (2) is needed because a base-rank-only predictor can match the decoder's
    pooled Spearman with an endpoint teacher. Beating the nulls only shows the
    method does something, not that it reads biology.
    """
    by_method = {r["method"]: r for r in result.get("methods", [])}
    primary = by_method.get(PRIMARY_METHOD)
    verdict: dict[str, Any] = {
        "axis": "direction_fidelity",
        "primary_pooled_spearman": primary["pooled_spearman"] if primary else float("nan"),
    }
    beats: list[str] = []
    fails: list[str] = []
    for c in result.get("contrasts", []):
        name = str(c["baseline"])
        delta = c.get("boot_delta_rho", float("nan"))
        excludes = c.get("boot_excludes_zero", float("nan"))
        verdict[f"delta_rho_vs_{name}"] = delta
        verdict[f"ci_excludes_zero_vs_{name}"] = excludes
        (beats if (delta > 0 and excludes == 1.0) else fails).append(name)
    verdict["beats"] = beats
    verdict["does_not_beat"] = fails
    evaluated = set(beats) | set(fails)
    required = [b for b in BASELINE_METHODS if b in evaluated]
    beats_baselines = bool(required) and all(b in beats for b in required)

    partial = (result.get("base_rank_control") or {}).get("linear_partial") or {}
    verdict["partial_spearman_given_base"] = partial.get("partial", float("nan"))
    verdict["partial_ci_low"] = partial.get("ci_low", float("nan"))
    verdict["partial_ci_high"] = partial.get("ci_high", float("nan"))
    adds_beyond_base = bool(partial) and partial.get("ci_low", float("nan")) > 0
    verdict["adds_beyond_base_rank"] = adds_beyond_base

    verdict["beats_required_baselines"] = beats_baselines
    verdict["pass"] = beats_baselines and adds_beyond_base
    verdict["required_baselines"] = required
    return verdict
