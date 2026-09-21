"""Shared color / order helpers for ISP UMAP cluster_coexpr_analysis figures."""

from __future__ import annotations

from typing import Any, Hashable, Iterable, Sequence

import pandas as pd
import seaborn as sns

# Labels that are not real cell-type detections (never shown on pred L2 bars).
EXCLUDED_PRED_LABELS = frozenset(
    {
        "",
        "nan",
        "None",
        "Unknown",
        "Ambiguous",
        "Other/Ambiguous",
    }
)


def _sorted_cluster_ids(values: Iterable[Any]) -> list[Any]:
    return list(sorted(pd.unique(pd.Series(list(values)))))


def is_detected_pred_label(label: Hashable) -> bool:
    """True when ``label`` is a real detected cell type (not Unknown/Ambiguous)."""
    s = str(label).strip()
    return bool(s) and s not in EXCLUDED_PRED_LABELS


def palette_for_groups(labels: Sequence[Hashable], group_col: str) -> dict[str, Any]:
    """Return ``{str(label): color}`` aligned with joint UMAP overlay panels."""
    series = pd.Series(list(labels))
    if group_col == "cluster":
        ids = _sorted_cluster_ids(series)
        n = max(len(ids), 1)
        try:
            colors = list(sns.color_palette("tab10", n_colors=n))
        except Exception:  # noqa: BLE001
            colors = []
        if not colors:
            colors = ["#4C72B0"] * n
        return {str(c): colors[i % len(colors)] for i, c in enumerate(ids)}

    types = [t for t in series.astype(str).value_counts().index if is_detected_pred_label(t)]
    n = max(len(types), 1)
    try:
        colors = list(sns.color_palette("husl", n_colors=n))
    except Exception:  # noqa: BLE001
        colors = []
    if not colors:
        colors = ["#4C72B0"] * n
    return {str(t): colors[i % len(colors)] for i, t in enumerate(types)}


def order_for_groups(
    labels: Sequence[Hashable],
    group_col: str,
    *,
    shift_l2: Sequence[float] | None = None,
    min_cells: int = 2,
) -> list[str]:
    """X-axis order for L2 plots (display only).

    For cell-type columns, order by mean ``shift_l2`` descending when provided,
    drop Unknown/Ambiguous, and drop types with fewer than ``min_cells`` cells
    (singleton false calls such as Neuron with n=1). Cluster columns stay
    frequency-sorted.
    """
    series = pd.Series(list(labels)).astype(str)
    if group_col == "cluster":
        return [str(x) for x in series.value_counts().index]

    mask = series.map(is_detected_pred_label)
    series = series[mask]
    if series.empty:
        return []

    counts = series.value_counts()
    keep = set(counts[counts >= max(1, int(min_cells))].index)

    if shift_l2 is not None:
        shifts = pd.Series(list(shift_l2), dtype=float)
        if len(shifts) == len(mask):
            shifts = shifts[mask.to_numpy()]
        if len(shifts) == len(series):
            means = (
                pd.DataFrame({"g": series.to_numpy(), "l2": shifts.to_numpy()})
                .groupby("g", sort=False)["l2"]
                .mean()
                .sort_values(ascending=False)
            )
            return [str(x) for x in means.index if x in keep]

    return [str(x) for x in counts.index if x in keep]
