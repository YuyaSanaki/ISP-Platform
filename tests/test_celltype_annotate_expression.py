"""Unit tests for pre-ISP expression-matrix cell-type annotation."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))


def test_load_panel_and_config_defaults():
    from celltype_annotate_expression import (
        ANNOTATOR_ID,
        annotation_config_from_tokenizer,
        default_panel_path,
        load_panel,
    )

    panel = load_panel()
    assert panel["version"] == ANNOTATOR_ID
    assert "Microglia" in panel["types"]
    assert default_panel_path().is_file()

    assert annotation_config_from_tokenizer({})["enabled"] is True
    assert annotation_config_from_tokenizer({"celltype_annotation": False})["enabled"] is False
    cfg = annotation_config_from_tokenizer(
        {"celltype_annotation": {"enabled": True, "min_score": 0.2}}
    )
    assert cfg["min_score"] == 0.2


def test_resolve_markers_case_tolerant():
    from celltype_annotate_expression import _resolve_markers

    genes = pd.Index(["Cx3cr1", "P2ry12", "Acta2"])
    assert _resolve_markers(["CX3CR1", "p2ry12", "missing"], genes) == ["Cx3cr1", "P2ry12"]


def test_is_platform_annotator():
    from celltype_annotate_expression import is_platform_celltype_annotation

    assert is_platform_celltype_annotation(pd.Series(["isp_expression_v1"] * 3))
    assert not is_platform_celltype_annotation(pd.Series(["user", "manual"]))
    assert not is_platform_celltype_annotation(None)


def test_annotate_adata_with_mocked_score_genes():
    """Fully mock scanpy so CI without scanpy still covers labeling logic."""
    anndata = pytest.importorskip("anndata")

    import celltype_annotate_expression as mod

    fake_sc = MagicMock()

    def fake_score_genes(adata, gene_list, score_name, use_raw=False):
        # Microglia markers present → high score; others low.
        n = adata.n_obs
        if any(g.lower().startswith("cx3") or g.lower().startswith("p2ry") for g in gene_list):
            adata.obs[score_name] = np.array([0.8, 0.1], dtype=float)[:n]
        elif any(g.lower().startswith("acta") for g in gene_list):
            adata.obs[score_name] = np.array([0.05, 0.7], dtype=float)[:n]
        else:
            adata.obs[score_name] = np.zeros(n, dtype=float)

    fake_sc.tl.score_genes = fake_score_genes
    fake_sc.pp.normalize_total = MagicMock()
    fake_sc.pp.log1p = MagicMock()

    genes = ["Cx3cr1", "P2ry12", "Tmem119", "Acta2", "Myh11", "Tagln"]
    X = np.array(
        [
            [10, 8, 6, 0, 0, 0],
            [0, 0, 0, 12, 9, 7],
        ],
        dtype=float,
    )
    adata = anndata.AnnData(
        X=X,
        var=pd.DataFrame(index=genes),
        obs=pd.DataFrame(index=["c0", "c1"]),
    )

    saved = sys.modules.get("scanpy")
    sys.modules["scanpy"] = fake_sc
    try:
        out = mod.annotate_adata_cell_types(adata, organism="mouse", inplace=True)
    finally:
        if saved is None:
            sys.modules.pop("scanpy", None)
        else:
            sys.modules["scanpy"] = saved

    assert list(out.obs["cell_type"].astype(str)) == ["Microglia", "Vascular_SMC"]
    assert list(out.obs["celltype_annotator"].astype(str)) == [
        "isp_expression_v1",
        "isp_expression_v1",
    ]
    assert "tissue" in out.obs.columns
    assert float(out.obs["celltype_score"].iloc[0]) > 0.5
    assert float(out.obs["celltype_score"].iloc[1]) > 0.5


def test_prefer_metadata_auto_uses_platform_only():
    from isp_umap_celltype import (
        annotate_dataframe_with_cell_types,
        resolve_prefer_metadata_flag,
    )

    name_id = {"Acta2": "ENS1", "Cx3cr1": "ENS10"}
    token_dict = {"ENS1": 101, "ENS10": 110}
    markers = {"Vascular_SMC": ["Acta2", "Myh11"], "Microglia": ["Cx3cr1", "P2ry12"]}
    ids = [[101, 999], [110, 999]]

    # Without platform annotator, auto must NOT overwrite with user cell_type.
    base_user = pd.DataFrame(
        {
            "cell_index": [0, 1],
            "shift_l2": [1.0, 0.5],
            "cell_type": ["Neuron", "Astrocyte"],
        }
    )
    assert resolve_prefer_metadata_flag("auto", base_user) is False
    annotated_user = annotate_dataframe_with_cell_types(
        base_user,
        ids,
        markers=markers,
        token_dict=token_dict,
        name_id=name_id,
        organism="mouse",
        use_rank_weights=False,
        prefer_metadata="auto",
        use_negative_markers=False,
    )
    assert list(annotated_user["celltype_source"]) == ["markers", "markers"]
    assert "Neuron" not in list(annotated_user["pred_cell_type"])

    # With platform annotator, auto overwrites from cell_type.
    base_plat = base_user.copy()
    base_plat["celltype_annotator"] = "isp_expression_v1"
    assert resolve_prefer_metadata_flag("auto", base_plat) is True
    annotated_plat = annotate_dataframe_with_cell_types(
        base_plat,
        ids,
        markers=markers,
        token_dict=token_dict,
        name_id=name_id,
        organism="mouse",
        use_rank_weights=False,
        prefer_metadata="auto",
        use_negative_markers=False,
    )
    assert list(annotated_plat["pred_cell_type"]) == ["Neuron", "Astrocyte"]
    assert list(annotated_plat["celltype_source"]) == ["platform", "platform"]
