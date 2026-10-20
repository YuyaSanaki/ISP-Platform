"""Rank enrichment figures in the style of the paper's Fig. 2 enrichment panels.

Enrichment: one panel per gene list, two panels per row at 174 mm page width. Bars are
-log10(FDR) of the top terms. Navy, FDR < 0.05. Grey, FDR >= 0.05. The dashed line marks
FDR = 0.05 and the number at the end of a bar is the number of overlapping genes. Terms with
the same overlapping genes and the same FDR are one row (the first term by name, "(+n)" for
the n others), so ties between such terms do not decide which of them are drawn.

Closeness: progress toward Goal and the fraction moved toward Goal per level (genes and the
gene sets of each library), queries as diamonds against leave-one-out placebo chains.
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from umap_plot_style import NAVY, OCHRE  # noqa: E402

GREY = "#8a8a8a"
MM = 1 / 25.4
PAGE_WIDTH_MM = 174
PANEL_HEIGHT_MM = 58

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.grid": False, "pdf.fonttype": 42,
}

_REPLACEMENTS = (
    (r"(?i)^positive regulation of\s+", "+ "),
    (r"(?i)^negative regulation of\s+", "− "),
    (r"(?i)^regulation of\s+", "Reg. "),
    (r"(?i)\bsignaling pathway\b", "signaling"),
    (r"(?i)\bbiological process\b", "process"),
    (r"(?i)\bdifferentiation\b", "diff."),
    (r"(?i)\bproliferation\b", "prolif."),
    (r"(?i)\bmitochondrial\b", "mito."),
    (r"(?i)\btranscriptional regulation of\s+", "TF reg. "),
)


@contextmanager
def figure_style() -> Iterator[None]:
    with plt.rc_context(RC):
        yield


def short_term(term: str, max_len: int = 34) -> str:
    """Compact term label: GO identifier stripped, common phrases shortened, then truncated."""
    t = re.sub(r"\s*\(GO:\d+\)\s*$", "", str(term).strip()).strip()
    for pat, rep in _REPLACEMENTS:
        t = re.sub(pat, rep, t)
    t = re.sub(r"\s+", " ", t).strip(" ,;")
    if len(t) <= max_len:
        return t
    cut = t[: max_len - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;/-") + "…"


def _head_without_split_ties(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """First n rows of an FDR-sorted table, minus a block of equal FDR that the cut would split."""
    if len(df) <= n:
        return df
    p = df["Adjusted P-value"].to_numpy()
    while n > 0 and np.isclose(p[n - 1], p[n], rtol=1e-9, atol=0):
        n -= 1
    return df.head(n)


def top_terms(res: pd.DataFrame | None, k: int = 10, fdr: float = 0.05) -> pd.DataFrame:
    """Top ``k`` rows by FDR, terms with the same genes and FDR merged into one row."""
    cols = ["Term", "Adjusted P-value", "n_overlap", "n_merged", "neglog10_fdr", "sig", "label"]
    if res is None or res.empty:
        return pd.DataFrame(columns=cols)
    df = res.copy()
    df["Adjusted P-value"] = pd.to_numeric(df["Adjusted P-value"], errors="coerce")
    df = df.dropna(subset=["Adjusted P-value"])
    df = df[df["n_overlap"] > 0]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["tie_key"] = [
        (frozenset(x.strip().upper() for x in str(g).split(";") if x.strip()), f"{p:.10e}")
        for g, p in zip(df.get("Genes", pd.Series([""] * len(df), index=df.index)),
                        df["Adjusted P-value"])
    ]
    df = df.sort_values(["Adjusted P-value", "Term"], kind="mergesort")
    n_same = df.groupby("tie_key")["Term"].transform("size")
    df = df.assign(n_merged=n_same - 1).drop_duplicates(subset=["tie_key"])
    top = _head_without_split_ties(df, int(k)).reset_index(drop=True)
    top["neglog10_fdr"] = -np.log10(top["Adjusted P-value"].clip(lower=1e-300))
    top["sig"] = top["Adjusted P-value"] < float(fdr)
    top["label"] = [short_term(t) + (f" (+{n})" if n else "")
                    for t, n in zip(top["Term"], top["n_merged"])]
    return top[cols]


def _bars(ax, top: pd.DataFrame, title: str, fdr: float) -> None:
    ax.set_title(title, loc="left")
    if top.empty:
        ax.text(0.5, 0.5, "No overlapping terms", ha="center", va="center",
                transform=ax.transAxes, color="#555555")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    top = top.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(top))
    v = top["neglog10_fdr"].to_numpy(dtype=float)
    xmax = max(float(v.max()), -np.log10(fdr) * 1.2)
    ax.barh(y, v, color=[NAVY if s else GREY for s in top["sig"]], height=0.72)
    ax.set_yticks(y, top["label"].tolist())
    ax.tick_params(axis="y", length=0, pad=2)
    for yi, (val, n) in enumerate(zip(v, top["n_overlap"])):
        ax.text(val + xmax * 0.02, yi, f"{int(n)}", va="center", ha="left", fontsize=7,
                color="#333333")
    ax.axvline(-np.log10(fdr), color=OCHRE, ls="--", lw=0.8)
    ax.set_xlim(0, xmax * 1.15)
    ax.set_ylim(-0.7, max(len(top), 1) - 0.3)
    ax.set_xlabel("−log$_{10}$ FDR")


def _save(fig, out_stem: Path) -> Path:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out_stem.with_suffix(".png")


def plot_enrichment(
    panels: Sequence[tuple[str, pd.DataFrame | None]],
    out_stem: Path,
    *,
    k: int = 10,
    fdr: float = 0.05,
    suptitle: str | None = None,
) -> Path:
    """One enrichment panel per ``(title, ORA table)``, two per row; the legend fills the
    last row (or its free cell)."""
    with figure_style():
        n = len(panels)
        nrows = -(-n // 2)
        fig, axes = plt.subplots(
            nrows, 2, figsize=(PAGE_WIDTH_MM * MM, (PANEL_HEIGHT_MM * nrows + 12) * MM),
            layout="constrained", squeeze=False,
        )
        flat = axes.ravel()
        for ax, (title, res) in zip(flat, panels):
            _bars(ax, top_terms(res, k, fdr), title, fdr)
        for ax in flat[n:]:
            ax.set_visible(False)
        if suptitle:
            fig.suptitle(suptitle, x=0.01, ha="left", fontsize=8.5)
        fig.legend(handles=[
            Patch(facecolor=NAVY, edgecolor="none", label=f"Enriched, FDR < {fdr:g}"),
            Patch(facecolor=GREY, edgecolor="none", label=f"FDR ≥ {fdr:g}"),
            Line2D([0], [0], color=OCHRE, ls="--", lw=0.8, label=f"FDR = {fdr:g}"),
        ], loc="outside lower center", ncol=3, frameon=False)
        return _save(fig, out_stem)


def plot_closeness(
    label: str,
    query: Mapping[str, Mapping[str, float]],
    placebo: Mapping[str, Mapping[str, Mapping[str, float]]],
    library_labels: Mapping[str, str],
    out_stem: Path,
) -> Path:
    """Progress toward Goal and the fraction moved toward Goal, ISP against placebo chains.

    Rows are genes and the gene sets of each library. Navy diamonds are ISP, grey dots the
    leave-one-out placebo chains.
    """
    levels = ["gene"] + [lib for lib in library_labels if lib in query]
    labels = ["Genes"] + [library_labels[lib] for lib in levels[1:]]
    panels = (
        ("progress", "Progress toward Goal (%)", 0.0),
        ("toward_goal", "Moved toward Goal (%)", 50.0),
    )
    with figure_style():
        fig, axes = plt.subplots(
            1, 2, figsize=(PAGE_WIDTH_MM * MM, (9 * len(levels) + 24) * MM),
            layout="constrained", sharey=True,
        )
        y = np.arange(len(levels))[::-1]
        rng = np.random.default_rng(0)
        for ax, (metric, xlabel, ref) in zip(axes, panels):
            for yi, level in zip(y, levels):
                nul = np.array([m.get(level, {}).get(metric, np.nan) for m in placebo.values()])
                nul = nul[np.isfinite(nul)] * 100
                ax.scatter(nul, yi + rng.uniform(-0.3, 0.3, len(nul)), s=10, color=GREY,
                           alpha=0.7, edgecolor="none", zorder=2)
                q = query.get(level, {}).get(metric, np.nan) * 100
                ax.scatter([q], [yi], s=34, marker="D", color=NAVY,
                           edgecolor="white", linewidth=0.5, zorder=3)
            ax.axvline(ref, color="black", ls="--", lw=0.7, alpha=0.6, zorder=1)
            ax.set_xlabel(xlabel)
        axes[0].set_yticks(y, labels)
        axes[0].set_ylim(-0.6, len(levels) - 0.4)
        axes[1].tick_params(axis="y", length=0)
        handles = [
            Line2D([], [], marker="D", ls="", color=NAVY, mec="white", ms=5, label=label),
            Line2D([], [], marker="o", ls="", color=GREY, ms=3.5, label="Placebo chains"),
        ]
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
                   frameon=False)
        return _save(fig, out_stem)
