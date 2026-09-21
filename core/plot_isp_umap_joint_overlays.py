#!/usr/bin/env python3
"""Recompute joint end+start+ISP UMAP and overlay L2 / cluster / cell type.

Writes under ``<run-dir>/cluster_coexpr_analysis/``:
  umap_joint_l2_cluster_celltype.png
  per_cell_cluster_l2_celltype.csv
  joint_umap_coords.npy
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_CORE = Path(__file__).resolve().parent
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

from isp_umap_postprocess_style import palette_for_groups  # noqa: E402

# Silhouette search bounds when n_clusters="auto".
_AUTO_K_MIN = 2
_AUTO_K_MAX = 15
_AUTO_SILHOUETTE_MAX_SAMPLES = 2000


def resolve_n_clusters(
    n_clusters: int | str,
    X: np.ndarray,
    *,
    seed: int = 42,
    k_min: int = _AUTO_K_MIN,
    k_max: int = _AUTO_K_MAX,
) -> int:
    """Return an integer KMeans ``n_clusters``.

    ``n_clusters`` may be an int (>=2) or ``\"auto\"``. Auto picks the K in
    ``[k_min, k_max]`` that maximises silhouette score on (optionally subsampled)
    rows of ``X``.
    """
    if isinstance(n_clusters, str) and n_clusters.strip().lower() == "auto":
        return _auto_n_clusters(X, seed=seed, k_min=k_min, k_max=k_max)
    try:
        k = int(n_clusters)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"n_clusters must be an int >= 2 or 'auto' (got {n_clusters!r})"
        ) from exc
    if k < 2:
        raise ValueError(f"n_clusters must be >= 2 (got {k})")
    return k


def _auto_n_clusters(
    X: np.ndarray,
    *,
    seed: int = 42,
    k_min: int = _AUTO_K_MIN,
    k_max: int = _AUTO_K_MAX,
) -> int:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n_samples = int(X.shape[0])
    if n_samples < 3:
        print(f"auto n_clusters: n_samples={n_samples} < 3; using k=2")
        return 2

    upper = min(int(k_max), n_samples - 1)
    lower = min(int(k_min), upper)
    if upper < lower:
        return max(2, upper)

    rng = np.random.default_rng(seed)
    if n_samples > _AUTO_SILHOUETTE_MAX_SAMPLES:
        idx = rng.choice(n_samples, size=_AUTO_SILHOUETTE_MAX_SAMPLES, replace=False)
        X_score = np.asarray(X)[idx]
    else:
        X_score = np.asarray(X)

    best_k = lower
    best_score = float("-inf")
    scores: list[tuple[int, float]] = []
    for k in range(lower, upper + 1):
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X_score)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(X_score, labels))
        scores.append((k, score))
        if score > best_score:
            best_k, best_score = k, score

    score_str = ", ".join(f"k={k}:{s:.3f}" for k, s in scores)
    print(
        f"auto n_clusters: chose k={best_k} (silhouette={best_score:.3f}; "
        f"searched {lower}..{upper}; {score_str})"
    )
    return int(best_k)


def fit_umap(embs: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, seed: int = 42):
    try:
        import cuml

        reducer = cuml.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
        xy = np.asarray(reducer.fit_transform(embs))
        return xy, "cuml"
    except Exception as exc:  # noqa: BLE001 — fall back to CPU UMAP
        print(f"cuML UMAP unavailable ({exc}); falling back to umap-learn")
        import umap

        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
        return reducer.fit_transform(embs), "umap-learn"


def _detect_states_and_gene(run_dir: Path, gene: str | None) -> tuple[str, str, str]:
    """Infer (end_state, start_state, gene) from embedding filenames."""
    isp_files = sorted(run_dir.glob("*_ISP_*_embs.npy"))
    if not isp_files:
        raise FileNotFoundError(f"No *_ISP_*_embs.npy under {run_dir}")

    if gene:
        match = next(
            (
                p
                for p in isp_files
                if p.name == f"AD_ISP_{gene}_embs.npy" or p.name.endswith(f"_ISP_{gene}_embs.npy")
            ),
            None,
        )
        if match is None:
            match = next((p for p in isp_files if f"_ISP_{gene}_embs.npy" in p.name), None)
        if match is None:
            raise FileNotFoundError(f"No ISP embedding for gene={gene!r} in {run_dir}")
        isp_path = match
    else:
        isp_path = isp_files[0]

    m = re.match(r"(.+)_ISP_(.+)_embs\.npy$", isp_path.name)
    if not m:
        raise ValueError(f"Unexpected ISP embedding name: {isp_path.name}")
    start_state = m.group(1)
    gene_out = m.group(2)

    candidates = []
    for p in run_dir.glob("*_embs.npy"):
        if "_ISP_" in p.name:
            continue
        if p.name == f"{start_state}_embs.npy":
            continue
        candidates.append(p)
    if not candidates:
        raise FileNotFoundError(
            f"No end-state *_embs.npy found alongside {start_state} in {run_dir}"
        )
    preferred = next((p for p in candidates if p.stem.replace("_embs", "") == "WT"), candidates[0])
    end_state = preferred.name[: -len("_embs.npy")]
    return end_state, start_state, gene_out


def _load_or_build_annot(
    run_dir: Path,
    out_dir: Path,
    annot_csv: Path,
    start_embs: np.ndarray,
    n_clusters: int | str,
    seed: int,
) -> tuple[pd.DataFrame, Path]:
    """Load annotation CSV or build a minimal one from per_cell_isp_shift.csv."""
    sources = []
    if annot_csv.exists():
        sources.append(annot_csv)
    fallback = run_dir / "per_cell_isp_shift.csv"
    if fallback.exists() and fallback.resolve() != annot_csv.resolve():
        sources.append(fallback)
    alt = run_dir / "per_cell_isp_shift_with_celltype.csv"
    if alt.exists():
        sources.append(alt)

    if not sources:
        raise FileNotFoundError(
            f"No annotation CSV found. Tried:\n"
            f"  {annot_csv}\n"
            f"  {fallback}\n"
            f"  {alt}\n"
            "Need at least per_cell_isp_shift.csv (from run_isp_umap.py)."
        )

    df = None
    used = None
    scored = []
    for path in sources:
        cand = pd.read_csv(path)
        if len(cand) != len(start_embs):
            continue
        richness = sum(
            1
            for c in ("coarse_type", "pred_cell_type", "cluster", "shift_l2")
            if c in cand.columns
        )
        scored.append((richness, path, cand))
    if scored:
        scored.sort(key=lambda x: (-x[0], str(x[1])))
        _, used, df = scored[0]
    if df is None:
        raise ValueError(
            f"No annotation CSV with {len(start_embs)} rows (start-state embedding count). "
            f"Tried: {[str(p) for p in sources]}"
        )
    print(f"Using annotation table: {used} ({len(df)} rows)")

    df = df.copy()
    if "shift_l2" not in df.columns:
        raise ValueError(f"{used} is missing required column 'shift_l2'")

    if "cluster" not in df.columns:
        from sklearn.cluster import KMeans

        k = resolve_n_clusters(n_clusters, start_embs, seed=seed)
        print(
            f"No 'cluster' column; fitting KMeans(n_clusters={k}, seed={seed}) "
            f"on start embeddings (requested={n_clusters!r})"
        )
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        df["cluster"] = km.fit_predict(start_embs)

    return df, used


def _celltype_column(df: pd.DataFrame) -> str:
    for col in ("coarse_type", "pred_cell_type", "pred_cell_type_v2", "cell_type", "celltype_plot"):
        if col in df.columns:
            return col
    if "sample_id" in df.columns:
        return "sample_id"
    df["celltype_plot"] = "Unknown"
    return "celltype_plot"


def run_joint_overlays(
    run_dir: Path,
    gene: str | None = None,
    annot_csv: Path | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
    num_trajectory_arrows: int = 100,
    n_clusters: int | str = 4,
) -> Path:
    """Build joint UMAP overlays for an ISP UMAP run directory. Returns the PNG path."""
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    end_state, start_state, gene = _detect_states_and_gene(run_dir, gene)
    print(f"States: end={end_state} start={start_state} gene={gene}")

    out_dir = run_dir / "cluster_coexpr_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    annot_path = (
        Path(annot_csv) if annot_csv is not None else (out_dir / "per_cell_cluster_l2_celltype.csv")
    )

    end_embs = np.load(run_dir / f"{end_state}_embs.npy")
    start_embs = np.load(run_dir / f"{start_state}_embs.npy")
    isp_embs = np.load(run_dir / f"{start_state}_ISP_{gene}_embs.npy")

    df, used_annot = _load_or_build_annot(
        run_dir, out_dir, annot_path, start_embs, n_clusters, seed
    )
    if len(df) != len(start_embs):
        raise ValueError(
            f"Annotation rows ({len(df)}) != {start_state} embeddings ({len(start_embs)})"
        )

    all_embs = np.vstack([end_embs, start_embs, isp_embs])
    print(f"Fitting joint UMAP on {all_embs.shape} ...")
    xy, backend = fit_umap(all_embs, n_neighbors, min_dist, seed)
    print(f"UMAP backend: {backend}")

    n_end, n_start = len(end_embs), len(start_embs)
    end_xy = xy[:n_end]
    start_xy = xy[n_end : n_end + n_start]
    isp_xy = xy[n_end + n_start :]

    df = df.copy()
    df["umap1_joint"] = start_xy[:, 0]
    df["umap2_joint"] = start_xy[:, 1]
    df["umap1_isp"] = isp_xy[:, 0]
    df["umap2_isp"] = isp_xy[:, 1]
    out_annot = out_dir / "per_cell_cluster_l2_celltype.csv"
    df.to_csv(out_annot, index=False)
    if used_annot.resolve() != out_annot.resolve():
        print(f"Wrote working annotation copy to {out_annot}")
    np.save(out_dir / "joint_umap_coords.npy", xy)

    ctype_col = _celltype_column(df)
    step = max(1, n_start // max(num_trajectory_arrows, 1))

    sns.set_theme(style="white", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].scatter(
        end_xy[:, 0], end_xy[:, 1], s=6, c="steelblue", alpha=0.55, label=end_state, rasterized=True
    )
    axes[0, 0].scatter(
        start_xy[:, 0],
        start_xy[:, 1],
        s=6,
        c="coral",
        alpha=0.55,
        label=start_state,
        rasterized=True,
    )
    axes[0, 0].scatter(
        isp_xy[:, 0],
        isp_xy[:, 1],
        s=6,
        c="red",
        alpha=0.45,
        label=f"{start_state}+ISP({gene})",
        rasterized=True,
    )
    for i in range(0, n_start, step):
        axes[0, 0].arrow(
            start_xy[i, 0],
            start_xy[i, 1],
            isp_xy[i, 0] - start_xy[i, 0],
            isp_xy[i, 1] - start_xy[i, 1],
            color="gray",
            alpha=0.25,
            width=0.01,
            head_width=0.12,
            length_includes_head=True,
        )
    axes[0, 0].legend(markerscale=3, frameon=False, fontsize=9)
    axes[0, 0].set_title(f"Joint UMAP: {end_state} / {start_state} / ISP")
    axes[0, 0].set_xlabel("UMAP 1")
    axes[0, 0].set_ylabel("UMAP 2")

    from matplotlib.colors import PowerNorm

    axes[0, 1].scatter(end_xy[:, 0], end_xy[:, 1], s=3, c="lightgray", alpha=0.25, rasterized=True)
    shift = np.asarray(df["shift_l2"].values, dtype=float)
    zero = shift <= 0
    if zero.any():
        axes[0, 1].scatter(
            start_xy[zero, 0],
            start_xy[zero, 1],
            s=6,
            c="#cfcfcf",
            alpha=0.55,
            rasterized=True,
            label="shift_l2 = 0",
            zorder=2,
            edgecolors="none",
        )
    sc = None
    if (~zero).any():
        movers = shift[~zero]
        vmax = float(np.percentile(movers, 95))
        vmax = max(vmax, float(np.percentile(movers, 50)) + 1e-6)
        norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)
        sc = axes[0, 1].scatter(
            start_xy[~zero, 0],
            start_xy[~zero, 1],
            c=movers,
            s=12,
            cmap="turbo",
            norm=norm,
            alpha=0.92,
            rasterized=True,
            zorder=3,
            edgecolors="none",
        )
        cbar = plt.colorbar(sc, ax=axes[0, 1], label="shift_l2 (>0)")
        if vmax < float(movers.max()):
            cbar.ax.set_xlabel(f"vmax=p95 ({vmax:.2f})", fontsize=8)
    axes[0, 1].set_title(f"{start_state} on joint UMAP: colored by L2 shift")
    axes[0, 1].set_xlabel("UMAP 1")
    axes[0, 1].set_ylabel("UMAP 2")
    if zero.any() and sc is not None:
        axes[0, 1].legend(markerscale=1.5, frameon=False, fontsize=8, loc="upper right")

    axes[1, 0].scatter(end_xy[:, 0], end_xy[:, 1], s=3, c="lightgray", alpha=0.2, rasterized=True)
    cluster_ids = sorted(pd.unique(df["cluster"]))
    cpal = palette_for_groups(df["cluster"], "cluster")
    for c in cluster_ids:
        m = df["cluster"].values == c
        axes[1, 0].scatter(
            start_xy[m, 0],
            start_xy[m, 1],
            s=8,
            color=cpal[str(c)],
            label=f"C{c}",
            alpha=0.85,
            rasterized=True,
        )
    axes[1, 0].legend(markerscale=2, frameon=False, fontsize=9)
    axes[1, 0].set_title(f"{start_state} on joint UMAP: embedding cluster")
    axes[1, 0].set_xlabel("UMAP 1")
    axes[1, 0].set_ylabel("UMAP 2")

    axes[1, 1].scatter(end_xy[:, 0], end_xy[:, 1], s=3, c="lightgray", alpha=0.2, rasterized=True)
    types = list(df[ctype_col].value_counts().index)
    tpal = palette_for_groups(df[ctype_col], ctype_col)
    for t in types:
        m = df[ctype_col].values == t
        axes[1, 1].scatter(
            start_xy[m, 0],
            start_xy[m, 1],
            s=8,
            color=tpal[str(t)],
            label=str(t),
            alpha=0.85,
            rasterized=True,
        )
    axes[1, 1].legend(
        markerscale=2, frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left"
    )
    axes[1, 1].set_title(f"{start_state} on joint UMAP: {ctype_col}")
    axes[1, 1].set_xlabel("UMAP 1")
    axes[1, 1].set_ylabel("UMAP 2")

    plt.tight_layout()
    out_png = out_dir / "umap_joint_l2_cluster_celltype.png"
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")
    return out_png


def _parse_n_clusters_arg(value: str) -> int | str:
    text = str(value).strip()
    if text.lower() == "auto":
        return "auto"
    k = int(text)
    if k < 2:
        raise argparse.ArgumentTypeError(f"n_clusters must be >= 2 or 'auto' (got {k})")
    return k


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="ISP UMAP run directory containing state/ISP embedding npy files",
    )
    parser.add_argument(
        "--gene",
        type=str,
        default=None,
        help="Perturbed gene label used in npy filenames (default: auto-detect)",
    )
    parser.add_argument("--annot-csv", type=Path, default=None)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-trajectory-arrows", type=int, default=100)
    parser.add_argument(
        "--n-clusters",
        type=_parse_n_clusters_arg,
        default=4,
        help="KMeans cluster count, or 'auto' (silhouette over k=2..15).",
    )
    args = parser.parse_args()
    run_joint_overlays(
        run_dir=args.run_dir,
        gene=args.gene,
        annot_csv=args.annot_csv,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        seed=args.seed,
        num_trajectory_arrows=args.num_trajectory_arrows,
        n_clusters=args.n_clusters,
    )


if __name__ == "__main__":
    main()
