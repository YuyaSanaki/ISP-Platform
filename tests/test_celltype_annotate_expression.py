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


def test_simple_marker_score_fallback_when_score_genes_fails():
    anndata = pytest.importorskip("anndata")
    import celltype_annotate_expression as mod

    fake_sc = MagicMock()
    fake_sc.pp.normalize_total = MagicMock()
    fake_sc.pp.log1p = MagicMock()
    fake_sc.tl.score_genes.side_effect = RuntimeError("No control genes found in any cut.")

    genes = ["Cx3cr1", "P2ry12", "Tmem119", "Acta2", "Myh11", "Tagln"] + [f"D{i}" for i in range(10)]
    X = np.ones((2, len(genes)), dtype=float) * 0.2
    X[0, 0:3] = 20
    X[1, 3:6] = 20
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


def test_prefer_metadata_auto_mixed_platform_and_user():
    from isp_umap_celltype import apply_metadata_cell_types

    df = pd.DataFrame(
        {
            "pred_cell_type": ["Microglia", "Vascular_SMC"],
            "pred_score": [0.4, 0.5],
            "coarse_type": ["Microglia", "Vascular_SMC"],
            "cell_type": ["Neuron", "Astrocyte"],
            "celltype_annotator": ["isp_expression_v1", "user_manual"],
        }
    )
    out = apply_metadata_cell_types(df, prefer_metadata="auto")
    assert list(out["pred_cell_type"]) == ["Neuron", "Vascular_SMC"]
    assert list(out["celltype_source"]) == ["platform", "markers"]


def test_prefer_metadata_false_ignores_platform_labels():
    from isp_umap_celltype import apply_metadata_cell_types

    df = pd.DataFrame(
        {
            "pred_cell_type": ["Microglia"],
            "pred_score": [0.4],
            "coarse_type": ["Microglia"],
            "cell_type": ["Neuron"],
            "celltype_annotator": ["isp_expression_v1"],
        }
    )
    out = apply_metadata_cell_types(df, prefer_metadata=False)
    assert list(out["pred_cell_type"]) == ["Microglia"]
    assert list(out["celltype_source"]) == ["markers"]


def test_yaml_defaults_wire_pre_isp_annotation():
    import yaml

    root = Path(__file__).resolve().parents[1]
    tok = yaml.safe_load((root / "core/config/tokenize.yaml").read_text())
    assert tok["tokenizer"]["celltype_annotation"]["enabled"] is True
    attrs = tok["tokenizer"]["custom_attr_name_dict"]
    for col in ("cell_type", "tissue", "celltype_score", "celltype_annotator"):
        assert attrs[col] == col

    umap = yaml.safe_load((root / "core/config/isp_umap.yaml").read_text())
    assert umap["postprocess"]["prefer_metadata_celltype"] == "auto"
    assert umap["postprocess"]["celltype_prediction"] is True

    isp = yaml.safe_load((root / "core/config/isp.yaml").read_text())
    assert isp["postprocess"]["prefer_metadata_celltype"] == "auto"

    pipe = yaml.safe_load((root / "core/config/pipeline.yaml").read_text())
    assert pipe["stages"]["isp"]["postprocess"]["prefer_metadata_celltype"] == "auto"


def test_apply_pre_isp_annotation_disabled_skips_labels():
    anndata = pytest.importorskip("anndata")
    from celltype_annotate_expression import apply_pre_isp_celltype_annotation

    adata = anndata.AnnData(
        X=np.ones((2, 3), dtype=float),
        var=pd.DataFrame(index=["a", "b", "c"]),
        obs=pd.DataFrame(index=["c0", "c1"]),
    )
    cfg = {"celltype_annotation": False, "custom_attr_name_dict": {"sample_id": "sample_id"}}
    out = apply_pre_isp_celltype_annotation(adata, cfg, species="mouse")
    assert "cell_type" not in out.obs.columns
    assert cfg["custom_attr_name_dict"]["cell_type"] == "cell_type"


def test_ensure_annotation_attrs_and_organism():
    from celltype_annotate_expression import (
        ensure_annotation_attrs,
        organism_from_species,
    )

    cfg: dict = {"custom_attr_name_dict": {"disease": "disease"}}
    ensure_annotation_attrs(cfg)
    assert cfg["custom_attr_name_dict"]["celltype_annotator"] == "celltype_annotator"
    assert organism_from_species({"model_organism": "human"}) == "human"
    assert organism_from_species("mouse") == "mouse"
    assert organism_from_species(None) == "mouse"


def _synthetic_marker_adata(organism: str = "mouse"):
    anndata = pytest.importorskip("anndata")
    from celltype_annotate_expression import load_panel

    panel = load_panel()
    types = panel["types"]
    key = "human" if organism == "human" else "mouse"
    genes: list[str] = []
    for spec in types.values():
        genes.extend(spec.get(key) or spec.get("mouse") or [])
    genes = list(dict.fromkeys(genes))
    decoys = [f"Decoy{i}" for i in range(80)]
    all_genes = genes + decoys
    n = len(all_genes)
    idx = {g: i for i, g in enumerate(all_genes)}
    X = np.ones((3, n), dtype=float) * 0.2
    for g in types["Microglia"][key]:
        X[0, idx[g]] = 80
    for g in types["Vascular_SMC"][key]:
        X[1, idx[g]] = 80
    for g in types["Neuron"][key]:
        X[2, idx[g]] = 80
    return anndata.AnnData(
        X=X,
        var=pd.DataFrame(index=all_genes),
        obs=pd.DataFrame(index=["mg", "smc", "neu"]),
    )


def test_real_scanpy_annotates_synthetic_mouse_cells():
    torch = sys.modules.get("torch")
    if torch is not None and getattr(torch, "__file__", None) is None:
        pytest.skip("torch is stubbed; real scanpy import is unsafe")
    pytest.importorskip("scanpy")
    from celltype_annotate_expression import annotate_adata_cell_types

    adata = _synthetic_marker_adata("mouse")
    x_before = np.array(adata.X, copy=True)
    out = annotate_adata_cell_types(adata, organism="mouse", inplace=True)
    assert list(out.obs["cell_type"].astype(str)) == ["Microglia", "Vascular_SMC", "Neuron"]
    assert (out.obs["celltype_annotator"].astype(str) == "isp_expression_v1").all()
    assert list(out.obs["tissue"].astype(str)) == ["Neural", "Muscle", "Neural"]
    np.testing.assert_array_equal(np.asarray(out.X), x_before)


def test_real_scanpy_annotates_human_and_10x_gene_symbols():
    torch = sys.modules.get("torch")
    if torch is not None and getattr(torch, "__file__", None) is None:
        pytest.skip("torch is stubbed; real scanpy import is unsafe")
    pytest.importorskip("scanpy")
    anndata = pytest.importorskip("anndata")
    from celltype_annotate_expression import annotate_adata_cell_types, load_panel

    src = _synthetic_marker_adata("human")
    # Mimic 10x var_names=gene_ids: Ensembl IDs + gene_symbols column.
    n = src.n_vars
    ens = [f"ENSG{i:011d}" for i in range(n)]
    adata = anndata.AnnData(
        X=src.X.copy(),
        var=pd.DataFrame({"gene_symbols": src.var_names.astype(str)}, index=ens),
        obs=src.obs.copy(),
    )
    out = annotate_adata_cell_types(adata, organism="human", inplace=True)
    assert list(out.obs["cell_type"].astype(str)) == ["Microglia", "Vascular_SMC", "Neuron"]
    panel = load_panel()
    assert "CX3CR1" in panel["types"]["Microglia"]["human"]


def test_apply_pre_isp_annotation_from_tokenizer_cfg():
    torch = sys.modules.get("torch")
    if torch is not None and getattr(torch, "__file__", None) is None:
        pytest.skip("torch is stubbed; real scanpy import is unsafe")
    pytest.importorskip("scanpy")
    from celltype_annotate_expression import apply_pre_isp_celltype_annotation

    adata = _synthetic_marker_adata("mouse")
    cfg = {
        "celltype_annotation": {"enabled": True},
        "custom_attr_name_dict": {"sample_id": "sample_id"},
    }
    out = apply_pre_isp_celltype_annotation(
        adata, cfg, species={"model_organism": "mouse"}
    )
    assert out.obs["cell_type"].astype(str).iloc[0] == "Microglia"
    assert cfg["custom_attr_name_dict"]["cell_type"] == "cell_type"

