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
    """Dual lollipop: significant genes with direction, shift, N_Detections, FDR."""
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

    denom = int(n_isp_cells) if n_isp_cells and n_isp_cells > 0 else int(sig["N_Detections"].max())
    denom = max(denom, 1)
    sig["detect_pct"] = 100.0 * sig["N_Detections"] / denom

    robust = sig[sig["N_Detections"] >= min_n_detections]
    pos = robust[robust["Shift_to_goal_end"] > 0].nlargest(top_n, "Shift_to_goal_end")
    neg = robust[robust["Shift_to_goal_end"] < 0].nsmallest(top_n, "Shift_to_goal_end")
    thin = sig[(sig["Shift_to_goal_end"] > 0) & (sig["N_Detections"] < min_n_detections)].nlargest(
        5, "Shift_to_goal_end"
    )

    pos_color = "#2a6f97"
    neg_color = "#9b2226"

    def _panel(ax, sub: pd.DataFrame, color: str, title: str) -> None:
        if sub.empty:
            ax.set_axis_off()
            ax.set_title(title + " (none)", fontsize=15, color=color)
            return
        sub = sub.iloc[::-1].reset_index(drop=True)
        y = np.arange(len(sub))
        x = sub["Shift_to_goal_end"].to_numpy()
        ax.hlines(y, 0, x, color=color, lw=6.0, alpha=0.9)
        ax.scatter(x, y, s=190, color=color, zorder=3, edgecolors="white", linewidths=1.5)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["Gene_name"].tolist(), fontsize=16)
        ax.set_xlabel(f"Shift ({label_start} → {label_end})", fontsize=15)
        ax.set_title(title, fontsize=16, color=color, pad=10, fontweight="bold")
        ax.tick_params(axis="x", labelsize=13)
        ax.axvline(0, color="#222", lw=1.2)
        ax.grid(axis="x", alpha=0.3, lw=1.0)
        ax.set_axisbelow(True)

        x_max = max(abs(float(x.min())), abs(float(x.max())), 1e-6)
        toward = color == pos_color
        for yi, (_, row) in zip(y, sub.iterrows()):
            fdr = float(row["Goal_end_FDR"])
            fdr_s = "≈0" if fdr == 0 else f"{fdr:.1e}"
            note = (
                f"{row['Shift_to_goal_end']:+.3f}   "
                f"N={int(row['N_Detections'])}/{denom} ({row['detect_pct']:.0f}%)   "
                f"FDR {fdr_s}"
            )
            if toward:
                ax.text(x_max * 1.06, yi, note, va="center", ha="left", fontsize=12.5, color="#222")
            else:
                ax.text(-x_max * 1.06, yi, note, va="center", ha="right", fontsize=12.5, color="#222")
        if toward:
            ax.set_xlim(0, x_max * 2.55)
        else:
            ax.set_xlim(-x_max * 2.55, 0)

    fig, axes = plt.subplots(1, 2, figsize=(18, 10), constrained_layout=True)
    _panel(
        axes[0],
        pos,
        pos_color,
        f"Toward {label_end} (+) · FDR<0.05 · N≥{min_n_detections} · top {top_n} by shift",
    )
    _panel(
        axes[1],
        neg,
        neg_color,
        f"Away from {label_end} (−) · FDR<0.05 · N≥{min_n_detections} · top {top_n} by |shift|",
    )

    thin_lines = [f"Thin significant hits (N<{min_n_detections}) — high shift, low support:"]
    if thin.empty:
        thin_lines.append("  (none)")
    else:
        for _, row in thin.iterrows():
            fdr = float(row["Goal_end_FDR"])
            fdr_s = "≈0" if fdr == 0 else f"{fdr:.2e}"
            thin_lines.append(
                f"  {row['Gene_name']}: {row['Shift_to_goal_end']:+.4f}, "
                f"N={int(row['N_Detections'])}/{denom} ({row['detect_pct']:.1f}%), FDR {fdr_s}"
            )
    fig.text(
        0.01,
        -0.02,
        "\n".join(thin_lines),
        ha="left",
        va="top",
        fontsize=12,
        family="monospace",
        color="#444",
    )
    fig.suptitle(
        "ISP significant genes: name · direction · effect size · detection support",
        fontsize=20,
        y=1.03,
        fontweight="bold",
    )
    out = figures_dir / "significant_genes_lollipop.png"
    fig.savefig(out, bbox_inches="tight", dpi=160)
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
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    top_pos = df.nlargest(n_top, "Shift_to_goal_end")
    colors_pos = ["red" if s == 1 else "steelblue" for s in top_pos["Sig"]]
    axes[0].barh(range(n_top), top_pos["Shift_to_goal_end"].values, color=colors_pos, edgecolor="white")
    axes[0].set_yticks(range(n_top))
    axes[0].set_yticklabels(top_pos["Gene_name"].values, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Shift to goal end")
    axes[0].set_title(f"Top {n_top} toward goal ({label_end})")
    top_neg = df.nsmallest(n_top, "Shift_to_goal_end")
    colors_neg = ["red" if s == 1 else "coral" for s in top_neg["Sig"]]
    axes[1].barh(range(n_top), top_neg["Shift_to_goal_end"].values, color=colors_neg, edgecolor="white")
    axes[1].set_yticks(range(n_top))
    axes[1].set_yticklabels(top_neg["Gene_name"].values, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Shift to goal end")
    axes[1].set_title(f"Top {n_top} away from goal ({label_end})")
    plt.suptitle("Red = significant (FDR)", fontsize=10, y=0.02, color="red")
    plt.tight_layout()
    fig.savefig(figures_dir / "top_genes_barplot.png", bbox_inches="tight")
    plt.close(fig)

    # --- 3b. Top significant genes bar (bar color intensity = N_Detections) ---
    df_sig = df[df["Sig"] == 1]
    if not df_sig.empty:
        n_sig_top = min(25, len(df_sig))
        fig = plt.figure(figsize=(19, 9))
        # Dedicated colorbar column so the legend never overlaps the panels
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.045], wspace=0.38)
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
            ax.set_yticklabels(labels, fontsize=10)
            ax.invert_yaxis()
            ax.set_xlabel("Shift to goal end", fontsize=12)
            ax.set_title(title, fontsize=13)
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
        cbar.set_label("N_Detections\n(√ scale; darker = more)", fontsize=11)

        fig.suptitle(
            "Significant genes (FDR < 0.05) · bar color intensity ∝ √N_Detections",
            fontsize=13,
            y=0.98,
        )
        fig.subplots_adjust(left=0.12, right=0.94, top=0.90, bottom=0.08)
        fig.savefig(figures_dir / "top_significant_genes_barplot.png", bbox_inches="tight")
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
