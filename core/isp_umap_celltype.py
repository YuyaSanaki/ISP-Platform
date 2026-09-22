"""Pre-ISP cell-type labels for ISP UMAP (platform ``isp_expression*`` only).

Brain / token-rank marker panels were removed. Cell types must come from
tokenize-time expression annotation (``celltype_annotate_expression`` →
``cell_type`` + ``celltype_annotator=isp_expression_v2`` on the HF dataset).
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# Prefer dataset metadata labels when present (pre-ISP platform annotation).
METADATA_CELLTYPE_COLUMNS = (
    "cell_type",
    "celltype",
    "CellType",
    "cell_type_label",
    "annotation",
)


def is_platform_celltype_annotation(series: pd.Series | None) -> bool:
    """True when a column carries platform expression-annotator provenance."""
    if series is None:
        return False
    s = series.astype(str)
    return bool(s.str.startswith("isp_expression").fillna(False).any())


def resolve_prefer_metadata_flag(
    setting: bool | str | None,
    df: pd.DataFrame | None = None,
) -> bool:
    """Interpret ``prefer_metadata_celltype`` against optional dataframe provenance.

    - ``True`` / ``"true"`` / ``"always"``: always prefer dataset ``cell_type``
    - ``False`` / ``"false"`` / ``"never"`` / ``None``: never prefer metadata
    - ``"auto"`` / ``"platform"``: prefer only when ``celltype_annotator`` starts
      with ``isp_expression`` (pre-ISP platform annotation)
    """
    if setting is True:
        return True
    if setting is False or setting is None:
        return False
    text = str(setting).strip().lower()
    if text in ("1", "true", "yes", "always", "on"):
        return True
    if text in ("0", "false", "no", "never", "off", ""):
        return False
    if text in ("auto", "platform"):
        if df is None or "celltype_annotator" not in df.columns:
            return False
        return is_platform_celltype_annotation(df["celltype_annotator"])
    logger.warning("Unknown prefer_metadata_celltype=%r; treating as false", setting)
    return False


def apply_metadata_cell_types(
    df: pd.DataFrame,
    *,
    prefer_metadata: bool | str = "auto",
    metadata_columns: Sequence[str] = METADATA_CELLTYPE_COLUMNS,
) -> pd.DataFrame:
    """Set ``pred_cell_type`` / ``coarse_type`` from pre-ISP dataset columns.

    Token-rank brain panels are not used.

    - ``auto`` / ``platform``: only ``isp_expression*`` annotator rows
    - ``true`` / ``always``: any non-empty ``cell_type``
    - ``false``: leave Unknown
    """
    out = df.copy()
    n = len(out)
    out["pred_cell_type"] = ["Unknown"] * n
    out["pred_score"] = [0.0] * n
    out["coarse_type"] = ["Other/Ambiguous"] * n
    out["celltype_source"] = ["none"] * n

    meta_col = next((c for c in metadata_columns if c in out.columns), None)
    if meta_col is None:
        logger.warning(
            "No cell_type metadata column; pred_cell_type=Unknown "
            "(enable tokenizer.celltype_annotation / isp_expression_v2)"
        )
        out["celltype_plot"] = out["pred_cell_type"]
        return out

    meta = out[meta_col].astype(str).str.strip()
    empty = meta.isna() | meta.isin(("", "nan", "None", "NA", "unknown", "Unknown"))

    if prefer_metadata is True:
        mode_text = "always"
    elif prefer_metadata is False or prefer_metadata is None:
        mode_text = "never"
    else:
        mode_text = str(prefer_metadata).strip().lower()

    if mode_text in ("0", "false", "no", "never", "off", ""):
        logger.warning(
            "prefer_metadata_celltype=%r → no labels (pre-ISP path disabled)",
            prefer_metadata,
        )
        out["celltype_plot"] = out["pred_cell_type"]
        return out

    if "celltype_annotator" in out.columns:
        platform_mask = (
            out["celltype_annotator"]
            .astype(str)
            .str.startswith("isp_expression")
            .fillna(False)
        )
    else:
        platform_mask = pd.Series(False, index=out.index)

    if mode_text in ("1", "true", "yes", "always", "on"):
        use_meta = ~empty
        # Platform rows get source=platform; other trusted metadata → metadata
        plat = use_meta & platform_mask
        other = use_meta & ~platform_mask
        out.loc[plat, "pred_cell_type"] = meta[plat]
        out.loc[plat, "pred_score"] = 1.0
        out.loc[plat, "coarse_type"] = meta[plat]
        out.loc[plat, "celltype_source"] = "platform"
        out.loc[other, "pred_cell_type"] = meta[other]
        out.loc[other, "pred_score"] = 1.0
        out.loc[other, "coarse_type"] = meta[other]
        out.loc[other, "celltype_source"] = "metadata"
    else:
        # auto / platform (default): isp_expression* only
        use_meta = (~empty) & platform_mask
        if not bool(platform_mask.any()):
            logger.warning(
                "No isp_expression* celltype_annotator on dataset; "
                "re-tokenize with celltype_annotation enabled"
            )
        out.loc[use_meta, "pred_cell_type"] = meta[use_meta]
        out.loc[use_meta, "pred_score"] = 1.0
        out.loc[use_meta, "coarse_type"] = meta[use_meta]
        out.loc[use_meta, "celltype_source"] = "platform"

    n_labeled = int((out["celltype_source"] != "none").sum())
    logger.info(
        "Pre-ISP cell types from %r: %d/%d cells labeled (%s)",
        meta_col,
        n_labeled,
        len(out),
        out["pred_cell_type"].value_counts().to_dict(),
    )
    out["celltype_plot"] = out["pred_cell_type"].astype(str)
    return out


def annotate_dataframe_with_cell_types(
    df: pd.DataFrame,
    input_ids_list: Sequence[Sequence[int]] | None = None,
    markers: Mapping[str, Sequence[str]] | None = None,
    token_dict: Mapping | None = None,
    name_id: Mapping | None = None,
    *,
    species: Mapping[str, Any] | None = None,
    organism: str | None = None,
    use_rank_weights: bool = True,
    prefer_metadata: bool | str = "auto",
    use_negative_markers: bool = True,
    negative_weight: float = 0.55,
    min_resolved_markers: int = 2,
    min_score: float = 0.25,
    min_margin: float = 0.05,
    min_markers_hit: int = 1,
) -> pd.DataFrame:
    """Attach pred_cell_type from pre-ISP metadata columns on ``df``.

    ``input_ids_list`` and marker-related kwargs are ignored (API kept for
    callers); brain token-rank panels were removed.
    """
    if markers is not None or token_dict is not None or name_id is not None:
        logger.warning(
            "Token-rank / brain marker arguments are ignored "
            "(cell types are pre-ISP expression annotation only)"
        )
    if input_ids_list is not None and len(df) != len(input_ids_list):
        raise ValueError(
            f"Row count mismatch: dataframe has {len(df)} rows, "
            f"input_ids has {len(input_ids_list)}"
        )
    # Drop stale prediction columns before applying pre-ISP labels.
    drop_cols = [
        c
        for c in df.columns
        if c
        in {
            "pred_cell_type",
            "pred_score",
            "coarse_type",
            "celltype_plot",
            "celltype_source",
        }
        or str(c).startswith("score_")
        or str(c).startswith("hits_")
    ]
    out = df.drop(columns=drop_cols) if drop_cols else df.copy()
    return apply_metadata_cell_types(out, prefer_metadata=prefer_metadata)


# Back-compat stub: marker prediction removed.
def predict_cell_types_from_input_ids(*_args, **_kwargs) -> pd.DataFrame:
    raise RuntimeError(
        "Token-rank brain cell-type panels were removed. "
        "Enable tokenizer.celltype_annotation (isp_expression_v2) and use "
        "annotate_dataframe_with_cell_types / prefer_metadata_celltype=auto."
    )
