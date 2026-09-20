"""Shared color / order helpers for ISP UMAP cluster_coexpr_analysis figures."""

from __future__ import annotations

from typing import Any, Hashable, Iterable, Sequence

import pandas as pd
import seaborn as sns

DEFAULT_COARSE_ORDER = [
    "Vascular_SMC-like",
    "SMC-intermediate",
    "Microglia-like",
    "Other/Ambiguous",
]


def _sorted_cluster_ids(values: Iterable[Any]) -> list[Any]:
    return list(sorted(pd.unique(pd.Series(list(values)))))


def palette_for_groups(labels: Sequence[Hashable], group_col: str) -> dict[str, Any]:
    """Return ``{str(label): color}`` aligned with joint UMAP overlay panels."""
    series = pd.Series(list(labels))
    if group_col == "cluster":
        ids = _sorted_cluster_ids(series)
        colors = sns.color_palette("tab10", n_colors=max(len(ids), 1))
        return {str(c): colors[i % len(colors)] for i, c in enumerate(ids)}

    types = list(series.value_counts().index)
    colors = sns.color_palette("husl", n_colors=max(len(types), 1))
    return {str(t): colors[i % len(colors)] for i, t in enumerate(types)}


def order_for_groups(labels: Sequence[Hashable], group_col: str) -> list[str]:
    """X-axis order for L2 plots (display only)."""
    series = pd.Series(list(labels)).astype(str)
    counts = series.value_counts()
    if group_col == "cluster":
        return [str(x) for x in counts.index]

    order = [c for c in DEFAULT_COARSE_ORDER if c in set(series)]
    order += [c for c in counts.index if c not in order]
    return order
