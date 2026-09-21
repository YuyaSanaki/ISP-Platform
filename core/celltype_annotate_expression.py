"""Pre-ISP cell-type annotation from the expression matrix (AnnData).

Scores curated marker panels with ``scanpy.tl.score_genes`` on a normalized copy
of the count matrix, then writes:

- ``cell_type`` — predicted label (or Ambiguous / Unknown)
- ``tissue`` — coarse tissue compartment
- ``celltype_score`` — winning panel score
- ``celltype_annotator`` — ``isp_expression_v1`` (platform provenance)

This is intended for paper-oriented ISP UMAP grouping. Token-rank marker scoring
in ``isp_umap_celltype`` remains available as a fallback when these columns are
absent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ANNOTATOR_ID = "isp_expression_v1"
_PANEL_PATH = (
    Path(__file__).resolve().parent / "geneformer" / "dicts" / "celltype_panels" / "isp_expression_v1.json"
)

# Columns written onto AnnData.obs / loom / HF dataset.
ANNOTATION_OBS_COLUMNS = (
    "cell_type",
    "tissue",
    "celltype_score",
    "celltype_annotator",
)


def default_panel_path() -> Path:
    return _PANEL_PATH


def load_panel(panel_path: Path | str | None = None) -> dict[str, Any]:
    path = Path(panel_path) if panel_path else _PANEL_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Cell-type panel not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _gene_symbols(adata: ad.AnnData) -> pd.Index:
    """Prefer symbol-like var names; fall back to a symbol column if present."""
    for col in ("gene_symbols", "gene_symbol", "symbol", "Gene", "gene_name"):
        if col in adata.var.columns:
            return pd.Index(adata.var[col].astype(str))
    return pd.Index(adata.var_names.astype(str))


def _resolve_markers(
    markers: list[str],
    gene_index: pd.Index,
) -> list[str]:
    """Case-tolerant intersection of panel markers with adata genes."""
    lookup = {g: g for g in gene_index}
    for g in list(gene_index):
        lookup.setdefault(g.upper(), g)
        lookup.setdefault(g.lower(), g)
        if len(g) > 1:
            lookup.setdefault(g[0].upper() + g[1:].lower(), g)
    out: list[str] = []
    seen: set[str] = set()
    for m in markers:
        hit = lookup.get(m) or lookup.get(m.upper()) or lookup.get(m.lower())
        if hit is None and len(m) > 1:
            hit = lookup.get(m[0].upper() + m[1:].lower())
        if hit is not None and hit not in seen:
            seen.add(hit)
            out.append(hit)
    return out


def _prepare_scoring_adata(adata: ad.AnnData) -> ad.AnnData:
    """Normalized copy for score_genes (does not modify the raw-count object)."""
    import scanpy as sc

    if adata.n_obs == 0 or adata.n_vars == 0:
        return adata.copy()
    scor = adata.copy()
    symbols = _gene_symbols(adata).astype(str)
    # score_genes matches on var_names; keep symbols unique.
    scor.var_names = symbols
    scor.var_names_make_unique()
    sc.pp.normalize_total(scor, target_sum=1e4)
    sc.pp.log1p(scor)
    return scor


def annotate_adata_cell_types(
    adata: ad.AnnData,
    *,
    organism: str = "mouse",
    panel_path: Path | str | None = None,
    enabled: bool = True,
    min_score: float | None = None,
    min_margin: float | None = None,
    inplace: bool = True,
) -> ad.AnnData:
    """Annotate ``adata.obs`` with expression-matrix cell-type labels.

    Operates on a normalized copy for scoring; the returned object keeps the
    original ``X`` when ``inplace=True`` (only obs columns are added).
    """
    if not enabled:
        return adata

    import scanpy as sc

    panel = load_panel(panel_path)
    org = (organism or "mouse").strip().lower()
    if org not in ("mouse", "human"):
        logger.warning("Unsupported organism %r for expression annotation; using mouse", organism)
        org = "mouse"

    thr = float(panel.get("min_score", 0.15) if min_score is None else min_score)
    margin = float(panel.get("min_margin", 0.02) if min_margin is None else min_margin)
    types: Mapping[str, Any] = panel.get("types") or {}

    target = adata if inplace else adata.copy()
    scor = _prepare_scoring_adata(target)
    gene_index = scor.var_names

    score_cols: list[str] = []
    tissue_map: dict[str, str] = {}
    for ct, spec in types.items():
        markers = list(spec.get(org) or spec.get("mouse") or [])
        resolved = _resolve_markers(markers, gene_index)
        tissue_map[ct] = str(spec.get("tissue") or "Other")
        col = f"_isp_score_{ct}"
        if len(resolved) < 2:
            logger.info(
                "Expression annotator: %s — fewer than 2 markers resolved (%s); skipping",
                ct,
                resolved,
            )
            scor.obs[col] = 0.0
        else:
            sc.tl.score_genes(scor, gene_list=resolved, score_name=col, use_raw=False)
        score_cols.append(col)

    if not score_cols:
        logger.warning("Expression annotator: no panels scored; labeling all Unknown")
        target.obs["cell_type"] = "Unknown"
        target.obs["tissue"] = "Other"
        target.obs["celltype_score"] = 0.0
        target.obs["celltype_annotator"] = ANNOTATOR_ID
        return target

    score_mat = scor.obs[score_cols].to_numpy(dtype=float)
    best_idx = np.argmax(score_mat, axis=1)
    best_scores = score_mat[np.arange(score_mat.shape[0]), best_idx]
    # Runner-up for margin.
    if score_mat.shape[1] >= 2:
        part = np.partition(score_mat, -2, axis=1)
        second = part[:, -2]
    else:
        second = np.zeros(score_mat.shape[0], dtype=float)

    type_names = [c.replace("_isp_score_", "", 1) for c in score_cols]
    labels: list[str] = []
    tissues: list[str] = []
    for i in range(score_mat.shape[0]):
        sc_best = float(best_scores[i])
        sc_second = float(second[i])
        if sc_best < thr:
            labels.append("Ambiguous")
            tissues.append("Other")
        elif (sc_best - sc_second) < margin and sc_best < 0.5:
            labels.append("Ambiguous")
            tissues.append("Other")
        else:
            ct = type_names[int(best_idx[i])]
            labels.append(ct)
            tissues.append(tissue_map.get(ct, "Other"))

    target.obs["cell_type"] = pd.Categorical(labels)
    target.obs["tissue"] = pd.Categorical(tissues)
    target.obs["celltype_score"] = best_scores.astype(float)
    target.obs["celltype_annotator"] = ANNOTATOR_ID

    vc = pd.Series(labels).value_counts().to_dict()
    logger.info(
        "Expression cell-type annotation (%s, organism=%s): %s",
        ANNOTATOR_ID,
        org,
        vc,
    )
    print(
        f"  celltype_annotation: {ANNOTATOR_ID} organism={org} "
        f"labels={vc}",
        flush=True,
    )
    return target


def annotation_config_from_tokenizer(tokenizer_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """Read ``tokenizer.celltype_annotation`` block with defaults."""
    raw = (tokenizer_cfg or {}).get("celltype_annotation")
    if raw is False:
        return {"enabled": False}
    if not isinstance(raw, dict):
        raw = {}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "panel": raw.get("panel"),
        "min_score": raw.get("min_score"),
        "min_margin": raw.get("min_margin"),
    }


def is_platform_celltype_annotation(series: pd.Series | None) -> bool:
    """True when obs/dataset carries our expression annotator provenance."""
    if series is None:
        return False
    s = series.astype(str)
    return bool(s.str.startswith("isp_expression").fillna(False).any())
