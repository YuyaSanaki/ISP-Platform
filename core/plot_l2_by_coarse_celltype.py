#!/usr/bin/env python3
"""Plot L2 / toward-end L2 shift by coarse cell program (or fallback group).

Writes under ``<run-dir>/cluster_coexpr_analysis/``:
  l2_by_coarse_celltype.png
  l2_mean_by_coarse_celltype.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_CORE = Path(__file__).resolve().parent
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

from isp_umap_postprocess_style import order_for_groups, palette_for_groups  # noqa: E402

GROUP_CANDIDATES = (
    "coarse_type",
    "pred_cell_type",
    "pred_cell_type_v2",
    "celltype_plot",
    "cell_type",
    "cluster",
)


def _resolve_annot_csv(run_dir: Path | None, annot_csv: Path | None) -> Path:
    if annot_csv is not None:
        path = annot_csv if annot_csv.is_absolute() else Path.cwd() / annot_csv
        if not path.exists():
            raise FileNotFoundError(f"Annotation CSV not found: {path}")
        return path

    if run_dir is None:
        raise ValueError("Provide --annot-csv and/or --run-dir")

    run_dir = run_dir if run_dir.is_absolute() else Path.cwd() / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    candidates = [
        run_dir / "cluster_coexpr_analysis" / "per_cell_cluster_l2_celltype.csv",
        run_dir / "per_cell_isp_shift_with_celltype.csv",
        run_dir / "per_cell_isp_shift.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No annotation CSV found. Tried:\n  " + "\n  ".join(str(p) for p in candidates)
    )


def _normalize_toward_col(df: pd.DataFrame) -> str:
    if "shift_toward_WT_l2" in df.columns:
        return "shift_toward_WT_l2"

    toward_cols = [c for c in df.columns if re.fullmatch(r"shift_toward_.+", c)]
    preferred = next((c for c in toward_cols if c in ("shift_toward_WT", "shift_toward_Ctrl")), None)
    if preferred is None and toward_cols:
        preferred = toward_cols[0]
    if preferred is None:
        raise ValueError(
            "Missing toward-end shift column. Expected one of: "
            "shift_toward_WT_l2, shift_toward_WT, shift_toward_<end_state>"
        )

    if preferred != "shift_toward_WT_l2":
        df["shift_toward_WT_l2"] = df[preferred]
        print(f"Using {preferred!r} as shift_toward_WT_l2")
    return "shift_toward_WT_l2"


def _resolve_group_col(df: pd.DataFrame, group_col: str | None) -> str:
    if group_col is not None:
        if group_col not in df.columns:
            raise ValueError(f"Group column {group_col!r} not in CSV columns: {list(df.columns)}")
        return group_col
    for col in GROUP_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError("No grouping column found. Expected one of: " + ", ".join(GROUP_CANDIDATES))


def run_l2_by_group(
    run_dir: Path | None = None,
    annot_csv: Path | None = None,
    group_col: str | None = None,
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Write L2 box and mean±SEM figures. Returns (box_path, mean_path)."""
    annot_path = _resolve_annot_csv(run_dir, annot_csv)
    df = pd.read_csv(annot_path)
    if "shift_l2" not in df.columns:
        raise ValueError(f"Missing column 'shift_l2' in {annot_path}")
    toward_col = _normalize_toward_col(df)
    resolved_group = _resolve_group_col(df, group_col)
    print(f"Annot: {annot_path}")
    print(f"Group by: {resolved_group}")

    plot_df = df.copy()
    raw_labels = plot_df[resolved_group]
    plot_df["coarse_type"] = raw_labels.astype(str)

    dest = out_dir or annot_path.parent
    if not dest.is_absolute():
        dest = Path.cwd() / dest
    dest.mkdir(parents=True, exist_ok=True)

    pal = palette_for_groups(raw_labels, resolved_group)
    order = order_for_groups(raw_labels, resolved_group)
    order = [o for o in order if o in set(plot_df["coarse_type"])]

    group_label = (
        "coarse cell program" if resolved_group == "coarse_type" else resolved_group.replace("_", " ")
    )

    sns.set_theme(style="whitegrid", context="talk")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(
        data=plot_df,
        x="coarse_type",
        y="shift_l2",
        hue="coarse_type",
        order=order,
        hue_order=order,
        ax=axes[0],
        palette=pal,
        fliersize=2,
        legend=False,
    )
    axes[0].set_title(f"L2 shift by {group_label}")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("shift_l2")
    axes[0].tick_params(axis="x", rotation=25)

    sns.boxplot(
        data=plot_df,
        x="coarse_type",
        y=toward_col,
        hue="coarse_type",
        order=order,
        hue_order=order,
        ax=axes[1],
        palette=pal,
        fliersize=2,
        legend=False,
    )
    axes[1].set_title(f"Toward-end L2 by {group_label}")
    axes[1].set_xlabel("")
    axes[1].set_ylabel(toward_col)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].axhline(0, color="gray", ls="--", lw=0.8)

    plt.tight_layout()
    box_path = dest / "l2_by_coarse_celltype.png"
    fig.savefig(box_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {box_path}")

    stat = (
        plot_df.groupby("coarse_type")["shift_l2"]
        .agg(mean="mean", sem=lambda s: float(s.sem()), count="count")
        .reindex(order)
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = [pal[o] for o in order]
    ax.bar(range(len(order)), stat["mean"].values, yerr=stat["sem"].values, color=colors, capsize=4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("mean shift_l2 ± SEM")
    ax.set_title(f"Mean L2 shift by {group_label}")
    for i, n in enumerate(stat["count"].values):
        ax.text(
            i,
            float(stat["mean"].iloc[i]) + float(stat["sem"].iloc[i]) + 0.04,
            f"n={int(n)}",
            ha="center",
            fontsize=9,
        )
    plt.tight_layout()
    mean_path = dest / "l2_mean_by_coarse_celltype.png"
    fig.savefig(mean_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {mean_path}")
    return box_path, mean_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--annot-csv", type=Path, default=None)
    parser.add_argument("--group-col", type=str, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    run_l2_by_group(
        run_dir=args.run_dir,
        annot_csv=args.annot_csv,
        group_col=args.group_col,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
