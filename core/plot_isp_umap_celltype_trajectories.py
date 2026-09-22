#!/usr/bin/env python3
"""Track ISP-perturbed cells on UMAP and summarize displacement by cell type.

Writes under ``<run-dir>/cluster_coexpr_analysis/``:
  umap_celltype_trajectories.png       — arrows colored by cell type + mean vectors
  celltype_shift_summary.csv           — per-type mean/median shift_l2 / umap_shift_l2
  l2_mean_by_pred_celltype.png         — mean±SEM bar (n>=2 detected types)
  l2_mean_by_pred_celltype_n_gt20.png  — same bar restricted to types with n>20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_CORE = Path(__file__).resolve().parent
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

from isp_umap_postprocess_style import (  # noqa: E402
    is_detected_pred_label,
    order_for_groups,
    palette_for_groups,
)
from plot_l2_by_coarse_celltype import (  # noqa: E402
    GROUP_CANDIDATES,
    _normalize_toward_col,
    _resolve_annot_csv,
    _resolve_group_col,
)

TRAJECTORY_GROUP_CANDIDATES = (
    "pred_cell_type",
    "celltype_plot",
    "cell_type",
    "cluster",
)


def _resolve_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (before_xy, after_xy) from joint or per-cell UMAP columns."""
    joint_before = ("umap1_joint", "umap2_joint")
    joint_after = ("umap1_isp", "umap2_isp")
    flat_before = ("umap1_before", "umap2_before")
    flat_after = ("umap1_after", "umap2_after")

    if all(c in df.columns for c in joint_before + joint_after):
        before = df[list(joint_before)].to_numpy(dtype=float)
        after = df[list(joint_after)].to_numpy(dtype=float)
        return before, after
    if all(c in df.columns for c in flat_before + flat_after):
        before = df[list(flat_before)].to_numpy(dtype=float)
        after = df[list(flat_after)].to_numpy(dtype=float)
        return before, after
    raise ValueError(
        "Need UMAP before/after columns "
        "(umap1_joint/umap2_joint + umap1_isp/umap2_isp, or "
        "umap1_before/umap2_before + umap1_after/umap2_after)"
    )


def _ensure_umap_shift(df: pd.DataFrame, before: np.ndarray, after: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    if "umap_shift_l2" not in out.columns:
        out["umap_shift_l2"] = np.linalg.norm(after - before, axis=1)
    return out


def build_celltype_shift_summary(
    df: pd.DataFrame,
    group_col: str,
    *,
    toward_col: str | None = None,
) -> pd.DataFrame:
    """Aggregate per-cell shift metrics by cell type / group."""
    work = df.copy()
    work["_group"] = work[group_col].astype(str)
    metrics = ["shift_l2"]
    if "umap_shift_l2" in work.columns:
        metrics.append("umap_shift_l2")
    if toward_col and toward_col in work.columns:
        metrics.append(toward_col)

    rows = []
    for g, sub in work.groupby("_group", sort=False):
        row: dict = {"cell_type": g, "n_cells": int(len(sub))}
        for m in metrics:
            vals = pd.to_numeric(sub[m], errors="coerce").dropna()
            row[f"mean_{m}"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"median_{m}"] = float(vals.median()) if len(vals) else float("nan")
            row[f"sem_{m}"] = float(vals.sem()) if len(vals) > 1 else 0.0
        # Mean UMAP displacement vector (tracking summary).
        if all(c in sub.columns for c in ("umap1_before", "umap2_before", "umap1_after", "umap2_after")):
            dx = float(
                (
                    pd.to_numeric(sub["umap1_after"], errors="coerce")
                    - pd.to_numeric(sub["umap1_before"], errors="coerce")
                ).mean()
            )
            dy = float(
                (
                    pd.to_numeric(sub["umap2_after"], errors="coerce")
                    - pd.to_numeric(sub["umap2_before"], errors="coerce")
                ).mean()
            )
            row["mean_du1"] = dx
            row["mean_du2"] = dy
        elif all(c in sub.columns for c in ("umap1_joint", "umap2_joint", "umap1_isp", "umap2_isp")):
            dx = float(
                (
                    pd.to_numeric(sub["umap1_isp"], errors="coerce")
                    - pd.to_numeric(sub["umap1_joint"], errors="coerce")
                ).mean()
            )
            dy = float(
                (
                    pd.to_numeric(sub["umap2_isp"], errors="coerce")
                    - pd.to_numeric(sub["umap2_joint"], errors="coerce")
                ).mean()
            )
            row["mean_du1"] = dx
            row["mean_du2"] = dy
        rows.append(row)

    summary = pd.DataFrame(rows)
    if "mean_shift_l2" in summary.columns:
        summary = summary.sort_values("mean_shift_l2", ascending=False)
    return summary.reset_index(drop=True)


def _plot_l2_mean_bar(
    plot_df: pd.DataFrame,
    *,
    order_bar: list[str],
    pal: dict[str, str],
    group_label: str,
    out_path: Path,
    title_suffix: str = "",
) -> Path | None:
    """Write mean±SEM shift_l2 bar chart for ``order_bar`` groups."""
    if not order_bar or "shift_l2" not in plot_df.columns:
        return None
    stat = (
        plot_df.groupby("_group")["shift_l2"]
        .agg(mean="mean", sem=lambda s: float(s.sem()), count="count")
        .reindex(order_bar)
    )
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(order_bar) + 2), 4.8))
    colors = [pal[o] for o in order_bar]
    ax.bar(
        range(len(order_bar)),
        stat["mean"].values,
        yerr=stat["sem"].values,
        color=colors,
        capsize=4,
    )
    ax.set_xticks(range(len(order_bar)))
    ax.set_xticklabels(order_bar, rotation=25, ha="right")
    ax.set_ylabel("mean shift_l2 ± SEM")
    title = f"Mean embedding L2 shift by {group_label}"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    ax.set_title(title)
    for i, n_i in enumerate(stat["count"].values):
        ax.text(
            i,
            float(stat["mean"].iloc[i]) + float(stat["sem"].iloc[i]) + 0.04,
            f"n={int(n_i)}",
            ha="center",
            fontsize=8,
        )
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def run_celltype_trajectory_plots(
    run_dir: Path | None = None,
    annot_csv: Path | None = None,
    group_col: str | None = None,
    out_dir: Path | None = None,
    *,
    num_trajectory_arrows: int = 200,
    seed: int = 42,
) -> tuple[Path, Path, Path | None, Path | None]:
    """Write trajectory + summary figures.

    Returns (traj_png, summary_csv, mean_bar_png, mean_bar_n_gt20_png).
    """
    annot_path = _resolve_annot_csv(run_dir, annot_csv)
    df = pd.read_csv(annot_path)
    if "shift_l2" not in df.columns:
        raise ValueError(f"Missing column 'shift_l2' in {annot_path}")

    before, after = _resolve_xy(df)
    df = _ensure_umap_shift(df, before, after)

    candidates = TRAJECTORY_GROUP_CANDIDATES + GROUP_CANDIDATES
    if group_col is not None:
        resolved_group = _resolve_group_col(df, group_col)
    else:
        resolved_group = next((c for c in candidates if c in df.columns), None)
        if resolved_group is None:
            raise ValueError(
                "No grouping column found. Expected one of: " + ", ".join(candidates)
            )

    toward_col = None
    try:
        toward_col = _normalize_toward_col(df)
    except ValueError:
        toward_col = None

    dest = out_dir or annot_path.parent
    if not dest.is_absolute():
        dest = Path.cwd() / dest
    dest.mkdir(parents=True, exist_ok=True)

    labels = df[resolved_group].astype(str)
    detected = labels.map(is_detected_pred_label)
    labels_plot = labels[detected]
    df_plot = df.loc[detected].copy()
    if labels_plot.empty:
        raise ValueError(
            f"No detected cell types in {resolved_group!r} "
            "(all Unknown/Ambiguous or empty)."
        )

    pal = palette_for_groups(labels_plot, resolved_group)
    order = order_for_groups(
        labels_plot,
        resolved_group,
        shift_l2=df_plot["shift_l2"] if "shift_l2" in df_plot.columns else None,
    )
    order = [o for o in order if o in set(labels_plot)]
    # Restrict rows to types that pass the detection / min-cell filter.
    in_order = labels_plot.isin(order)
    labels_plot = labels_plot[in_order]
    df_plot = df_plot.loc[in_order].copy()
    if labels_plot.empty:
        raise ValueError(
            f"No cell types with enough cells in {resolved_group!r} "
            "(need n>=2 after dropping Unknown/Ambiguous)."
        )

    summary = build_celltype_shift_summary(df_plot, resolved_group, toward_col=toward_col)
    # Detected types only; drop singleton / Ambiguous noise (n < 2).
    summary = summary[summary["n_cells"] >= 2].reset_index(drop=True)
    summary_path = dest / "celltype_shift_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    # Use detected-only rows for trajectories / mean bars.
    before = before[detected.to_numpy()][in_order.to_numpy()]
    after = after[detected.to_numpy()][in_order.to_numpy()]
    labels = labels_plot.reset_index(drop=True)
    df = df_plot.reset_index(drop=True)

    n = len(df)
    rng = np.random.default_rng(seed)
    n_arrows = max(0, min(int(num_trajectory_arrows), n))
    if n_arrows > 0 and n_arrows < n:
        # Stratify arrows across types so rare types still appear.
        idx_parts: list[np.ndarray] = []
        remaining = n_arrows
        groups = list(labels.value_counts().index)
        for i, g in enumerate(groups):
            g_idx = np.flatnonzero(labels.values == g)
            share = max(1, int(round(n_arrows * len(g_idx) / n))) if remaining > 0 else 0
            if i == len(groups) - 1:
                share = remaining
            share = min(share, len(g_idx), remaining)
            if share <= 0:
                continue
            pick = rng.choice(g_idx, size=share, replace=False)
            idx_parts.append(np.sort(pick))
            remaining -= share
        arrow_idx = np.concatenate(idx_parts) if idx_parts else np.array([], dtype=int)
    else:
        arrow_idx = np.arange(n) if n_arrows else np.array([], dtype=int)

    sns.set_theme(style="white", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))

    # Left: per-cell trajectories colored by cell type.
    ax = axes[0]
    ax.scatter(before[:, 0], before[:, 1], s=4, c="#d0d0d0", alpha=0.35, rasterized=True, zorder=1)
    for i in arrow_idx:
        g = str(labels.iloc[i])
        color = pal.get(g, "#555555")
        ax.annotate(
            "",
            xy=(after[i, 0], after[i, 1]),
            xytext=(before[i, 0], before[i, 1]),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                alpha=0.55,
                lw=0.9,
                mutation_scale=8,
            ),
            zorder=2,
        )
    # Legend proxies
    handles = [
        plt.Line2D([0], [0], color=pal[o], lw=2, label=f"{o} (n={(labels == o).sum()})")
        for o in order
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="best")
    ax.set_title(f"ISP trajectories by {resolved_group.replace('_', ' ')}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # Right: mean displacement vector per type (from type centroid).
    ax = axes[1]
    ax.scatter(before[:, 0], before[:, 1], s=4, c="#e8e8e8", alpha=0.4, rasterized=True, zorder=1)
    for o in order:
        m = labels.values == o
        if not m.any():
            continue
        cx, cy = float(before[m, 0].mean()), float(before[m, 1].mean())
        dx = float((after[m, 0] - before[m, 0]).mean())
        dy = float((after[m, 1] - before[m, 1]).mean())
        mean_l2 = float(pd.to_numeric(df.loc[m, "shift_l2"], errors="coerce").mean())
        color = pal[o]
        ax.scatter(
            before[m, 0],
            before[m, 1],
            s=10,
            color=color,
            alpha=0.35,
            rasterized=True,
            zorder=2,
            edgecolors="none",
        )
        ax.annotate(
            "",
            xy=(cx + dx, cy + dy),
            xytext=(cx, cy),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=2.4,
                mutation_scale=14,
            ),
            zorder=3,
        )
        ax.scatter([cx], [cy], s=40, color=color, edgecolors="black", linewidths=0.6, zorder=4)
        ax.text(
            cx + dx,
            cy + dy,
            f"{o}\nL2={mean_l2:.2f}",
            fontsize=7,
            color=color,
            ha="left",
            va="bottom",
        )
    ax.set_title("Mean UMAP displacement by cell type")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    plt.tight_layout()
    traj_png = dest / "umap_celltype_trajectories.png"
    fig.savefig(traj_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {traj_png}")

    # Fine-type mean bars when pred_cell_type (or chosen group) is available.
    plot_df = df.copy()
    plot_df["_group"] = labels
    order_bar = [o for o in order if o in set(plot_df["_group"])]
    group_label = resolved_group.replace("_", " ")
    mean_png = _plot_l2_mean_bar(
        plot_df,
        order_bar=order_bar,
        pal=pal,
        group_label=group_label,
        out_path=dest / "l2_mean_by_pred_celltype.png",
    )
    counts = plot_df["_group"].value_counts()
    order_n20 = [o for o in order_bar if int(counts.get(o, 0)) > 20]
    mean_png_n20 = _plot_l2_mean_bar(
        plot_df,
        order_bar=order_n20,
        pal=pal,
        group_label=group_label,
        out_path=dest / "l2_mean_by_pred_celltype_n_gt20.png",
        title_suffix="n>20",
    )

    return traj_png, summary_path, mean_png, mean_png_n20


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--annot-csv", type=Path, default=None)
    parser.add_argument("--group-col", type=str, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--num-trajectory-arrows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_celltype_trajectory_plots(
        run_dir=args.run_dir,
        annot_csv=args.annot_csv,
        group_col=args.group_col,
        out_dir=args.out_dir,
        num_trajectory_arrows=args.num_trajectory_arrows,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
