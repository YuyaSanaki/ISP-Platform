"""Shared UMAP appearance: white background, L-shaped axes (Fig.2 endpoint).

Fig.2 endpoint UMAP, Fig.3/4 manuscript ISP UMAP, and WebUI `run_isp_umap.py`
use this so seaborn's `sns.set()` (pulled in via `geneformer`) does not paint
darkgrid onto figures.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

NAVY = "#1f4e79"
OCHRE = "#c47a3a"
GREEN = "#2f6b4f"


def apply_endpoint_umap_rc() -> None:
    """White figure/axes, no grid, outward ticks. Preserves the current backend."""
    backend = mpl.get_backend()
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams["backend"] = backend
    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["axes.facecolor"] = "white"
    mpl.rcParams["savefig.facecolor"] = "white"
    mpl.rcParams["savefig.edgecolor"] = "none"
    mpl.rcParams["axes.grid"] = False
    mpl.rcParams["axes.edgecolor"] = "black"
    mpl.rcParams["axes.labelcolor"] = "black"
    mpl.rcParams["xtick.color"] = "black"
    mpl.rcParams["ytick.color"] = "black"
    mpl.rcParams["text.color"] = "black"
    mpl.rcParams["xtick.direction"] = "out"
    mpl.rcParams["ytick.direction"] = "out"


def style_spines(ax) -> None:
    """L-frame (hide top/right), white face, no grid."""
    ax.set_facecolor("white")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(labelsize=8, direction="out", colors="black", length=3.5, width=0.8)


def style_umap_axes(ax) -> None:
    style_spines(ax)
    ax.set_xlabel("UMAP-1", fontsize=9)
    ax.set_ylabel("UMAP-2", fontsize=9)


def save_white(fig, path: Path | str, *, dpi: int = 300) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")


def plot_isp_umap_scatter(
    umap_embs: np.ndarray,
    *,
    n_end: int,
    n_start: int,
    end_state: str,
    start_state: str,
    pert_label: str,
    title: str,
    out_path: Path | str,
    show_arrows: bool,
    num_arrows: int,
) -> None:
    """WebUI / pipeline ISP UMAP scatter (same axes as Fig.2 endpoint)."""
    apply_endpoint_umap_rc()
    xy = np.asarray(umap_embs, dtype=float)
    end_xy = xy[:n_end]
    start_xy = xy[n_end : n_end + n_start]
    pert_xy = xy[n_end + n_start :]

    fig, ax = plt.subplots(figsize=(7.2, 6.4), facecolor="white")
    ax.set_facecolor("white")
    ax.scatter(
        end_xy[:, 0],
        end_xy[:, 1],
        s=8,
        alpha=0.7,
        c=GREEN,
        linewidths=0,
        rasterized=True,
        zorder=1,
        label=f"{end_state} (n={len(end_xy)})",
    )
    ax.scatter(
        start_xy[:, 0],
        start_xy[:, 1],
        s=8,
        alpha=0.7,
        c=OCHRE,
        linewidths=0,
        rasterized=True,
        zorder=1,
        label=f"{start_state} (n={len(start_xy)})",
    )
    ax.scatter(
        pert_xy[:, 0],
        pert_xy[:, 1],
        s=8,
        alpha=0.75,
        c=NAVY,
        linewidths=0,
        rasterized=True,
        zorder=2,
        label=f"{pert_label} (n={len(pert_xy)})",
    )

    if show_arrows and n_start > 0 and len(pert_xy) >= n_start:
        step = max(1, n_start // max(int(num_arrows), 1))
        for i in range(0, n_start, step):
            ax.annotate(
                "",
                xy=(float(pert_xy[i, 0]), float(pert_xy[i, 1])),
                xytext=(float(start_xy[i, 0]), float(start_xy[i, 1])),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=NAVY,
                    lw=0.7,
                    alpha=0.45,
                    mutation_scale=8,
                    shrinkA=0,
                    shrinkB=0,
                ),
                zorder=3,
            )

    style_umap_axes(ax)
    ax.set_title(title, fontsize=10)
    handles = [
        Line2D([], [], marker="o", linestyle="", color=GREEN, label=f"{end_state} (n={len(end_xy)})", markersize=6),
        Line2D([], [], marker="o", linestyle="", color=OCHRE, label=f"{start_state} (n={len(start_xy)})", markersize=6),
        Line2D([], [], marker="o", linestyle="", color=NAVY, label=f"{pert_label} (n={len(pert_xy)})", markersize=6),
    ]
    if show_arrows:
        handles.append(Line2D([], [], color=NAVY, lw=1.2, label="start → perturbed"))
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="best", markerscale=1.4)
    save_white(fig, out_path, dpi=300)
    plt.close(fig)
