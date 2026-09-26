"""
ISP post-stats figures and tables (CLI; former notebook workflow).

Writes PNGs to a figures directory and CSV summaries to the stats directory.
Uses pandas + matplotlib; optional cuDF is not required for this script.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def _is_single_gene_shift_only(df: pd.DataFrame) -> bool:
    """True when stats are per-cell shifts for one targeted gene (no p-values)."""
    return "Goal_end_vs_random_pval" not in df.columns


def _plot_significant_lollipop(
    df: pd.DataFrame,
    figures_dir: Path,
    *,
    label_start: str,
    label_end: str,
    n_isp_cells: int | None = None,
    min_n_detections: int = 20,
    top_n: int = 15,
) -> None:
    """Dual lollipop: significant genes by shift direction (gene names + stems only)."""
    required = {"Gene_name", "Shift_to_goal_end", "N_Detections", "Goal_end_FDR"}
    if not required.issubset(df.columns):
        print("[isp_analysis] Skipping significant_genes_lollipop.png (missing columns).")
        return

    work = df.copy()
    work["Shift_to_goal_end"] = pd.to_numeric(work["Shift_to_goal_end"], errors="coerce")
    work["N_Detections"] = pd.to_numeric(work["N_Detections"], errors="coerce").fillna(0).astype(int)
    work["Goal_end_FDR"] = pd.to_numeric(work["Goal_end_FDR"], errors="coerce")

    if "Sig" in work.columns:
        sig = work[work["Sig"] == 1].copy()
    else:
        sig = work[work["Goal_end_FDR"] < 0.05].copy()
    if sig.empty:
        print("[isp_analysis] No significant genes; skipping significant_genes_lollipop.png")
        return

    robust = sig[sig["N_Detections"] >= min_n_detections]
    pos = robust[robust["Shift_to_goal_end"] > 0].nlargest(top_n, "Shift_to_goal_end")
    neg = robust[robust["Shift_to_goal_end"] < 0].nsmallest(top_n, "Shift_to_goal_end")

    pos_color = "#2a6f97"
    neg_color = "#9b2226"
    gene_fs = 26

    def _label_width_in(names: list[str], fontsize: float) -> float:
        if not names:
            return 1.2
        max_len = max(len(str(n)) for n in names)
        return max(1.2, max_len * fontsize * 0.58 / 72.0 + 0.55)

    pos_names = [] if pos.empty else pos["Gene_name"].astype(str).tolist()
    neg_names = [] if neg.empty else neg["Gene_name"].astype(str).tolist()
    # Widen canvas from longest gene symbols so y-labels never clip after tight crop.
    fig_w = 10.5 + _label_width_in(pos_names, gene_fs) + _label_width_in(neg_names, gene_fs)
    fig_h = 10.0

    def _panel(ax, sub: pd.DataFrame, color: str, title: str, *, labels_side: str) -> None:
        if sub.empty:
            ax.set_axis_off()
            ax.set_title(title + " (none)", fontsize=18, color=color)
            return
        sub = sub.iloc[::-1].reset_index(drop=True)
        y = np.arange(len(sub))
        x = sub["Shift_to_goal_end"].to_numpy()
        ax.hlines(y, 0, x, color=color, lw=5.5, alpha=0.9)
        ax.scatter(x, y, s=200, color=color, zorder=3, edgecolors="white", linewidths=1.4)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["Gene_name"].tolist(), fontsize=gene_fs)
        ax.set_xlabel(f"Shift ({label_start} → {label_end})", fontsize=20)
        ax.set_title(title, fontsize=18, color=color, pad=8, fontweight="bold", linespacing=1.15)
        ax.tick_params(axis="x", labelsize=16)
        ax.axvline(0, color="#222", lw=1.2)
        ax.grid(axis="x", alpha=0.3, lw=1.0)
        ax.set_axisbelow(True)

        x_max = max(abs(float(x.min())), abs(float(x.max())), 1e-6)
        pad = x_max * 0.12
        if labels_side == "left":
            ax.set_xlim(0, x_max + pad)
            ax.yaxis.tick_left()
            ax.tick_params(axis="y", labelleft=True, labelright=False, labelsize=gene_fs)
            for lab in ax.get_yticklabels():
                lab.set_ha("right")
        else:
            ax.set_xlim(-(x_max + pad), 0)
            ax.yaxis.tick_right()
            ax.tick_params(axis="y", labelleft=False, labelright=True, labelsize=gene_fs)
            for lab in ax.get_yticklabels():
                lab.set_ha("left")

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), layout="constrained")
    _panel(
        axes[0],
        pos,
        pos_color,
        f"Toward {label_end} (+)\nFDR<0.05 · N≥{min_n_detections} · top {top_n}",
        labels_side="left",
    )
    _panel(
        axes[1],
        neg,
        neg_color,
        f"Away from {label_end} (−)\nFDR<0.05 · N≥{min_n_detections} · top {top_n}",
        labels_side="right",
    )
    # Keep main title just above panel titles (constrained layout owns the gap).
    fig.suptitle("ISP significant genes", fontsize=22, fontweight="bold")
    fig.get_layout_engine().set(h_pad=0.04, w_pad=0.02, hspace=0.02, wspace=0.03)

    out = figures_dir / "significant_genes_lollipop.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    print(f"[isp_analysis] Wrote {out.name}")


def run_isp_figure_analysis(
    parquet_path: str | Path,
    figures_dir: str | Path,
    stats_dir: str | Path,
    *,
    label_start: str = "start",
    label_end: str = "goal",
    n_isp_cells: int | None = None,
) -> None:
    parquet_path = Path(parquet_path)
    figures_dir = Path(figures_dir)
    stats_dir = Path(stats_dir)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    figures_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 150

    df = pd.read_parquet(parquet_path)
    print(f"[isp_analysis] Loaded {len(df)} rows from {parquet_path}")
    if "Sig" in df.columns:
        print(f"[isp_analysis] Significant (Sig=1): {int(df['Sig'].sum())}")

    goal_label = f"{label_start} → {label_end}"

    # --- Summary to stdout ---
    print("=" * 60)
    print("Summary Statistics for Shift_to_goal_end")
    print("=" * 60)
    print(df["Shift_to_goal_end"].describe())
    if "Goal_end_FDR" in df.columns:
        print(f"\nGenes with FDR < 0.05: {(df['Goal_end_FDR'] < 0.05).sum()}")
        print(f"Genes with FDR < 0.10: {(df['Goal_end_FDR'] < 0.10).sum()}")

    if _is_single_gene_shift_only(df):
        print(
            "[isp_analysis] Single-gene perturbation stats (no random-null p-values); "
            "generating shift-only figures."
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df["Shift_to_goal_end"], bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Zero shift")
        ax.set_xlabel(f"Shift to goal ({label_end})")
        ax.set_ylabel("Number of cells")
        ax.set_title(f"Per-cell cosine shift: {goal_label}")
        ax.legend()
        plt.tight_layout()
        fig.savefig(figures_dir / "shift_distribution.png", bbox_inches="tight")
        plt.close(fig)

        df_sorted = df.sort_values("Shift_to_goal_end", ascending=False).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(14, 5))
        colors_wf = ["steelblue" if v >= 0 else "coral" for v in df_sorted["Shift_to_goal_end"]]
        ax.bar(range(len(df_sorted)), df_sorted["Shift_to_goal_end"], color=colors_wf, width=1.0, edgecolor="none")
        ax.set_xlabel("Cell rank", fontsize=13)
        ax.set_ylabel("Shift to goal end", fontsize=13)
        ax.set_title(f"Per-cell waterfall: {goal_label}", fontsize=14)
        ax.axhline(y=0, color="black", linewidth=0.5)
        plt.tight_layout()
        fig.savefig(figures_dir / "waterfall_plot.png", bbox_inches="tight")
        plt.close(fig)

        summary = df["Shift_to_goal_end"].describe().to_frame(name="Shift_to_goal_end")
        summary.to_csv(stats_dir / "single_gene_shift_summary.csv")
        df_sorted.to_csv(stats_dir / "single_gene_per_cell_shifts.csv", index=False)
        print(f"[isp_analysis] Figures saved under: {figures_dir}")
        print(f"[isp_analysis] Tables saved under:   {stats_dir}")
        return

    # --- 1. Shift + p-value distributions ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].hist(df["Shift_to_goal_end"], bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Zero shift")
    axes[0].set_xlabel(f"Shift to goal ({label_end})")
    axes[0].set_ylabel("Number of genes")
    axes[0].set_title("Distribution of cosine similarity shifts")
    axes[0].legend()
    axes[1].hist(df["Goal_end_vs_random_pval"], bins=50, color="coral", edgecolor="white", alpha=0.8)
    axes[1].axvline(x=0.05, color="red", linestyle="--", linewidth=1.5, label="p=0.05")
    axes[1].set_xlabel("P-value (vs random)")
    axes[1].set_ylabel("Number of genes")
    axes[1].set_title("Distribution of p-values")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(figures_dir / "shift_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # --- 2. Volcano ---
    df = df.copy()
    df["neg_log10_fdr"] = -np.log10(df["Goal_end_FDR"].clip(lower=1e-300))
    df["neg_log10_pval"] = -np.log10(df["Goal_end_vs_random_pval"].clip(lower=1e-300))
    colors = np.where(
        df["Goal_end_FDR"] < 0.05,
        "red",
        np.where(df["Goal_end_vs_random_pval"] < 0.05, "orange", "grey"),
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(df["Shift_to_goal_end"], df["neg_log10_pval"], c=colors, alpha=0.5, s=20, edgecolors="none")
    n_label = 15
    top_genes = pd.concat(
        [df.nlargest(n_label, "Shift_to_goal_end"), df.nsmallest(n_label, "Shift_to_goal_end")]
    ).drop_duplicates()
    for _, row in top_genes.iterrows():
        ax.annotate(
            row["Gene_name"],
            (row["Shift_to_goal_end"], row["neg_log10_pval"]),
            fontsize=7,
            alpha=0.8,
            textcoords="offset points",
            xytext=(5, 3),
        )
    ax.axhline(y=-np.log10(0.05), color="blue", linestyle="--", alpha=0.5, label="p=0.05")
    ax.axvline(x=0, color="grey", linestyle="--", alpha=0.5)
    ax.set_xlabel("Shift to goal end", fontsize=13)
    ax.set_ylabel("-log10(p-value)", fontsize=13)
    ax.set_title(f"Volcano plot: {goal_label}", fontsize=14)
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=8, label="FDR < 0.05"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", markersize=8, label="p < 0.05"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey", markersize=8, label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")
    plt.tight_layout()
    fig.savefig(figures_dir / "volcano_plot.png", bbox_inches="tight")
    plt.close(fig)

    # --- 3. Top genes bar ---
    n_top = 25
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
    top_pos = df.nlargest(n_top, "Shift_to_goal_end")
    colors_pos = ["red" if s == 1 else "steelblue" for s in top_pos["Sig"]]
    axes[0].barh(range(n_top), top_pos["Shift_to_goal_end"].values, color=colors_pos, edgecolor="white")
    axes[0].set_yticks(range(n_top))
    axes[0].set_yticklabels(top_pos["Gene_name"].values, fontsize=13)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Shift to goal end", fontsize=14)
    axes[0].set_title(f"Top {n_top} toward goal ({label_end})", fontsize=14)
    axes[0].tick_params(axis="x", labelsize=12)
    top_neg = df.nsmallest(n_top, "Shift_to_goal_end")
    colors_neg = ["red" if s == 1 else "coral" for s in top_neg["Sig"]]
    axes[1].barh(range(n_top), top_neg["Shift_to_goal_end"].values, color=colors_neg, edgecolor="white")
    axes[1].set_yticks(range(n_top))
    axes[1].set_yticklabels(top_neg["Gene_name"].values, fontsize=13)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Shift to goal end", fontsize=14)
    axes[1].set_title(f"Top {n_top} away from goal ({label_end})", fontsize=14)
    axes[1].tick_params(axis="x", labelsize=12)
    plt.suptitle("Red = significant (FDR)", fontsize=12, y=0.02, color="red")
    plt.tight_layout()
    fig.savefig(figures_dir / "top_genes_barplot.png", bbox_inches="tight", dpi=160)
    plt.close(fig)

    # --- 3b. Top significant genes bar (bar color intensity = N_Detections) ---
    df_sig = df[df["Sig"] == 1]
    if not df_sig.empty:
        n_sig_top = min(25, len(df_sig))
        fig = plt.figure(figsize=(14, 11))
        # Dedicated colorbar column so the legend never overlaps the panels
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.05], wspace=0.45)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
        cax = fig.add_subplot(gs[0, 2])
        n_cmap = plt.cm.Reds

        top_sig_pos = df_sig.nlargest(n_sig_top, "Shift_to_goal_end")
        top_sig_neg = df_sig.nsmallest(n_sig_top, "Shift_to_goal_end")
        n_all = pd.concat(
            [
                pd.to_numeric(top_sig_pos.get("N_Detections", pd.Series(dtype=float)), errors="coerce"),
                pd.to_numeric(top_sig_neg.get("N_Detections", pd.Series(dtype=float)), errors="coerce"),
            ]
        ).fillna(1.0).clip(lower=1.0)
        n_norm = plt.Normalize(vmin=float(np.sqrt(n_all.min())), vmax=float(np.sqrt(n_all.max())))

        def _sig_bar_panel(ax, sub: pd.DataFrame, title: str) -> None:
            n_rows = len(sub)
            y = np.arange(n_rows)
            shifts = sub["Shift_to_goal_end"].to_numpy()
            if "N_Detections" in sub.columns:
                n_det = pd.to_numeric(sub["N_Detections"], errors="coerce").fillna(1).clip(lower=1).to_numpy()
            else:
                n_det = np.ones(n_rows)
            colors = n_cmap(n_norm(np.sqrt(n_det)))
            ax.barh(y, shifts, height=0.75, color=colors, edgecolor="white", linewidth=0.6)
            ax.set_yticks(y)
            labels = [
                f"{name}  (N={int(n)})" for name, n in zip(sub["Gene_name"].tolist(), n_det)
            ]
            ax.set_yticklabels(labels, fontsize=13)
            ax.invert_yaxis()
            ax.set_xlabel("Shift to goal end", fontsize=14)
            ax.set_title(title, fontsize=14)
            ax.tick_params(axis="x", labelsize=12)
            ax.axvline(0, color="#333", lw=0.8)
            ax.grid(axis="x", alpha=0.25)
            ax.set_axisbelow(True)

        _sig_bar_panel(
            axes[0],
            top_sig_pos,
            f"Top {len(top_sig_pos)} significant toward goal ({label_end})",
        )
        _sig_bar_panel(
            axes[1],
            top_sig_neg,
            f"Top {len(top_sig_neg)} significant away from goal ({label_end})",
        )

        sm = plt.cm.ScalarMappable(cmap=n_cmap, norm=n_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        # Keep √ scaling for color, but label ticks as actual N for readability
        sqrt_ticks = np.linspace(float(n_norm.vmin), float(n_norm.vmax), 5)
        cbar.set_ticks(sqrt_ticks)
        cbar.set_ticklabels([f"{int(round(t * t))}" for t in sqrt_ticks])
        cbar.set_label("N_Detections\n(√ scale; darker = more)", fontsize=12)
        cbar.ax.tick_params(labelsize=11)

        fig.suptitle(
            "Significant genes (FDR < 0.05) · bar color intensity ∝ √N_Detections",
            fontsize=15,
            y=0.98,
        )
        fig.subplots_adjust(left=0.16, right=0.93, top=0.90, bottom=0.08)
        fig.savefig(figures_dir / "top_significant_genes_barplot.png", bbox_inches="tight", dpi=160)
        plt.close(fig)

        # --- 3c. Significant genes lollipop (direction + N_Detections + FDR) ---
        _plot_significant_lollipop(
            df,
            figures_dir,
            label_start=label_start,
            label_end=label_end,
            n_isp_cells=n_isp_cells,
        )
    else:
        print("[isp_analysis] No significant genes (Sig=1) found; skipping top_significant_genes_barplot.png")

    # --- 4. Waterfall ---
    df_sorted = df.sort_values("Shift_to_goal_end", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    colors_wf = ["steelblue" if v >= 0 else "coral" for v in df_sorted["Shift_to_goal_end"]]
    ax.bar(range(len(df_sorted)), df_sorted["Shift_to_goal_end"], color=colors_wf, width=1.0, edgecolor="none")
    ax.set_xlabel("Gene rank", fontsize=13)
    ax.set_ylabel("Shift to goal end", fontsize=13)
    ax.set_title(f"Waterfall (all genes ranked): {goal_label}", fontsize=14)
    ax.axhline(y=0, color="black", linewidth=0.5)
    top1 = df_sorted.iloc[0]
    bottom1 = df_sorted.iloc[-1]
    ax.annotate(
        top1["Gene_name"],
        (0, top1["Shift_to_goal_end"]),
        fontsize=9,
        fontweight="bold",
        xytext=(30, 10),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    ax.annotate(
        bottom1["Gene_name"],
        (len(df_sorted) - 1, bottom1["Shift_to_goal_end"]),
        fontsize=9,
        fontweight="bold",
        xytext=(-80, -20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    plt.tight_layout()
    fig.savefig(figures_dir / "waterfall_plot.png", bbox_inches="tight")
    plt.close(fig)

    # --- 5. CSV exports (same as notebook) ---
    export_cols = [
        c
        for c in [
            "Gene_name",
            "Ensembl_ID",
            "Shift_to_goal_end",
            "Goal_end_vs_random_pval",
            "Goal_end_FDR",
            "N_Detections",
            "Sig",
        ]
        if c in df.columns
    ]
    top100_pos = df.nlargest(100, "Shift_to_goal_end")[export_cols]
    top100_neg = df.nsmallest(100, "Shift_to_goal_end")[export_cols]
    top100_pos.to_csv(stats_dir / "top100_positive_shifters.csv", index=False)
    top100_neg.to_csv(stats_dir / "top100_negative_shifters.csv", index=False)
    if "Sig" in df.columns:
        sig_genes = df[df["Sig"] == 1].sort_values("Shift_to_goal_end", ascending=False)
        sig_genes.to_csv(stats_dir / "significant_genes.csv", index=False)
    if "Goal_end_vs_random_pval" in df.columns:
        nominal_sig = df[df["Goal_end_vs_random_pval"] < 0.05].sort_values("Shift_to_goal_end", ascending=False)
        nominal_sig.to_csv(stats_dir / "nominal_significant_genes_p0.05.csv", index=False)

    print(f"[isp_analysis] Figures saved under: {figures_dir}")
    print(f"[isp_analysis] Tables saved under:   {stats_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="ISP parquet → figures + CSV tables.")
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--figures-dir", type=Path, required=True)
    p.add_argument("--stats-dir", type=Path, required=True, help="Directory for CSV exports (usually ispstats_results).")
    p.add_argument("--label-start", default="start")
    p.add_argument("--label-end", default="goal")
    p.add_argument(
        "--n-isp-cells",
        type=int,
        default=None,
        help="ISP start-state cell count (isp.max_ncells) for N/total % annotations.",
    )
    args = p.parse_args()
    run_isp_figure_analysis(
        args.parquet,
        args.figures_dir,
        args.stats_dir,
        label_start=args.label_start,
        label_end=args.label_end,
        n_isp_cells=args.n_isp_cells,
    )


if __name__ == "__main__":
    main()
