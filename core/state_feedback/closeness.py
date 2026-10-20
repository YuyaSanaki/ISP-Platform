"""How far ISP moves the Start ranks toward the Goal ranks, at gene and gene-set level.

All metrics compare the ISP change ``d`` (gene-level Delta-spec, ISP rank minus Start rank)
with the observed change ``G`` (Goal rank minus Start rank) on the background genes, using
pseudobulk ranks (``state_feedback.gene_ranks``). At gene-set level each term is scored by
the mean rank of its member genes in the background, so ``d`` and ``G`` become term vectors.

- ``progress``: fraction of the Start-to-Goal change that ISP covers, the projection
  ``<d, G> / <G, G>``. 0 = no move toward Goal, 1 = all the way.
- ``cosine``: direction agreement of ``d`` and ``G`` (scale-free).
- ``spearman``: rank correlation of ``d`` and ``G``.
- ``distance_closed``: ``1 - |ISP - Goal| / |Start - Goal|``. Change orthogonal to ``G``
  lowers it, so it can be negative when ISP mostly moves elsewhere.
- ``toward_goal``: among the 20% of genes (terms) with the largest ``|G|``, the fraction that
  ISP moves in the direction of Goal. 0.5 = chance.
- ``auc_goal_gain`` (gene level): probability that a gene among the 10% moving most toward
  the front from Start to Goal moves further toward the front under ISP than a gene among
  the 10% moving most toward the back. 0.5 = chance.

Each metric is compared with leave-one-out placebo values: for each comparison placebo chain,
its displacement minus the mean displacement of the other placebo chains, the same contrast
that gives Delta-spec of the query chain. A placebo chain's own genes get 0, as the query
chain's genes are outside the background.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import gene_ranks as gr

MIN_SET_SIZE = 10
TOWARD_TOP = 0.2
AUC_FRACTION = 0.1
VECTOR_METRICS = ("progress", "cosine", "spearman", "distance_closed", "toward_goal")


def chain_displacements(
    start: pd.DataFrame, by_chain: Mapping[str, pd.DataFrame], n_cells: int, tokens: Sequence[int]
) -> pd.DataFrame:
    """Pseudobulk displacement (final minus start) per chain on ``tokens``; columns = chains."""
    idx = pd.Index([int(t) for t in tokens], name="token")
    return pd.DataFrame({
        c: gr.mean_displacement(start, final, n_cells).reindex(idx, fill_value=0.0)
        for c, final in by_chain.items()
    })


def _contrast(disp: pd.DataFrame, chain: str, others: list[str],
              chain_tokens: Mapping[str, Iterable[int]]) -> pd.Series:
    idx = disp.index
    vals = pd.DataFrame({
        c: disp[c].where(~idx.isin([int(t) for t in chain_tokens.get(c, ())])) for c in others
    })
    return disp[chain] - vals.mean(axis=1)


def query_and_placebo_deltas(
    disp: pd.DataFrame, configured: str, chain_tokens: Mapping[str, Iterable[int]]
) -> tuple[pd.Series, pd.DataFrame]:
    """Delta-spec of the configured chain and leave-one-out pseudo Delta-spec per placebo chain."""
    placebos = [c for c in disp.columns if c != configured]
    if len(placebos) < 2:
        raise ValueError("closeness needs at least 2 comparison placebo chains")
    query = _contrast(disp, configured, placebos, chain_tokens).fillna(0.0)
    pseudo = {}
    for p in placebos:
        d = _contrast(disp, p, [c for c in placebos if c != p], chain_tokens).fillna(0.0)
        own = disp.index.isin([int(t) for t in chain_tokens.get(p, ())])
        pseudo[p] = d.where(~own, 0.0)
    return query.rename("delta_spec"), pd.DataFrame(pseudo, index=disp.index)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def vector_metrics(d: np.ndarray, G: np.ndarray, *, top: float = TOWARD_TOP) -> dict[str, float]:
    d = np.asarray(d, dtype=float)
    G = np.asarray(G, dtype=float)
    gg = float(G @ G)
    nd = float(np.linalg.norm(d))
    big = np.abs(G) >= np.quantile(np.abs(G), 1.0 - float(top)) if len(G) else np.zeros(0, bool)
    big &= G != 0
    return {
        "progress": float(d @ G) / gg if gg > 0 else float("nan"),
        "cosine": float(d @ G) / (nd * np.sqrt(gg)) if nd > 0 and gg > 0 else float("nan"),
        "spearman": _spearman(d, G),
        "distance_closed": 1.0 - float(np.linalg.norm(G - d)) / np.sqrt(gg) if gg > 0 else float("nan"),
        "toward_goal": float(np.mean(np.sign(d[big]) == np.sign(G[big]))) if big.any() else float("nan"),
    }


def auc_goal_gain(d: np.ndarray, G: np.ndarray, *, fraction: float = AUC_FRACTION) -> float:
    """P(d of a Goal-gaining gene < d of a Goal-losing gene), ties count half."""
    d = np.asarray(d, dtype=float)
    G = np.asarray(G, dtype=float)
    k = max(1, int(np.ceil(float(fraction) * len(G))))
    order = np.argsort(G, kind="mergesort")
    pos, neg = d[order[:k]], d[order[-k:]]
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    u = ranks[k:].sum() - k * (k + 1) / 2.0
    return float(u / (k * k))


def set_members(
    tokens: Sequence[int], symbols: Mapping[int, str], gene_sets: Mapping[str, set[str]],
    *, min_size: int = MIN_SET_SIZE,
) -> dict[str, np.ndarray]:
    """Row indices into ``tokens`` of each term with at least ``min_size`` background genes."""
    sym = pd.Series([str(symbols.get(int(t), t)).upper() for t in tokens])
    out = {}
    for term, genes in gene_sets.items():
        rows = np.flatnonzero(sym.isin(genes).to_numpy())
        if len(rows) >= int(min_size):
            out[term] = rows
    return out


def set_means(values: np.ndarray, members: Mapping[str, np.ndarray]) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return np.array([v[rows].mean() for rows in members.values()])


def all_metrics(
    d: np.ndarray, G: np.ndarray, members_by_lib: Mapping[str, Mapping[str, np.ndarray]]
) -> dict[str, dict[str, float]]:
    """``{"gene": {...}, <library>: {...}}`` for one ISP change ``d``."""
    gene = vector_metrics(d, G)
    gene["auc_goal_gain"] = auc_goal_gain(d, G)
    out = {"gene": gene}
    for lib, members in members_by_lib.items():
        out[lib] = vector_metrics(set_means(d, members), set_means(G, members))
        out[lib]["n_terms"] = float(len(members))
    return out


def compare_to_placebos(query: float, placebos: Sequence[float]) -> dict[str, float]:
    """Query value, placebo mean/SD, z, and the empirical upper-tail p ``(1 + #>=) / (1 + n)``."""
    p = np.asarray([v for v in placebos if np.isfinite(v)], dtype=float)
    out = {"value": float(query), "placebo_mean": float("nan"), "placebo_sd": float("nan"),
           "z": float("nan"), "p_empirical": float("nan"), "n_placebos": int(len(p))}
    if len(p):
        out["placebo_mean"] = float(p.mean())
        out["p_empirical"] = float((1 + np.sum(p >= query)) / (1 + len(p)))
    if len(p) > 1:
        sd = float(p.std(ddof=1))
        out["placebo_sd"] = sd
        out["z"] = float((query - p.mean()) / sd) if sd > 0 else float("nan")
    return out


def ora_profile_metrics(
    results: Mapping[str, pd.DataFrame], reference: pd.DataFrame, *, fdr: float
) -> dict[str, dict[str, float]]:
    """Per list: Jaccard of FDR < ``fdr`` terms with the reference list, and cosine of -log10 p."""
    ref_sig = set(reference.loc[reference["Adjusted P-value"] < fdr, "Term"])
    ref_p = reference.set_index("Term")["P-value"]
    out = {}
    for name, res in results.items():
        sig = set(res.loc[res["Adjusted P-value"] < fdr, "Term"])
        union = sig | ref_sig
        terms = ref_p.index.union(pd.Index(res["Term"]))
        a = -np.log10(res.set_index("Term")["P-value"].reindex(terms).fillna(1.0).clip(lower=1e-300))
        b = -np.log10(ref_p.reindex(terms).fillna(1.0).clip(lower=1e-300))
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        out[name] = {
            "sig_jaccard": len(sig & ref_sig) / len(union) if union else float("nan"),
            "sig_shared": float(len(sig & ref_sig)),
            "neglog10p_cosine": float(a @ b) / (na * nb) if na > 0 and nb > 0 else float("nan"),
        }
    return out
