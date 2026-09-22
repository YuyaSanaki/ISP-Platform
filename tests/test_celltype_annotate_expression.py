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
    types = panel["types"]
    assert "Microglia" in types
    assert "Resident_macrophage" not in types
    assert "Fast_myofiber" not in types and "Slow_myofiber" not in types
    assert types["C1QC_APOE_macrophage"]["requires"]["myeloid_compartment"] is True
    assert types["C1QC_APOE_macrophage"]["requires"]["microglia_core_negative_or_low"] is True
    lec = types["Lymphatic_endothelial"]
    assert lec["min_core_detected"] == 3
    assert lec["human"]["anti"] == []
    assert "CCL21" in lec["human"]["core"]
    assert "LYVE1" not in lec["human"]["core"]
    assert "MERTK" in types["Macrophage"]["human"]["core"]
    assert "TYROBP" in types["Macrophage"]["human"]["support"]
    assert "RGS5" in types["Myofibroblast"]["human"]["anti"]
    assert types["Fibroblast"]["min_core_detected"] == 3
    assert types["Vascular_SMC"]["min_core_detected"] == 3
    assert "fiber_program" in types["Skeletal_myocyte"]["states"]
    assert "Trp63" in types["Basal_keratinocyte"]["mouse"]["core"]
    assert "Tp63" not in types["Basal_keratinocyte"]["mouse"]["core"]
    assert "LORICRIN" in types["Suprabasal_keratinocyte"]["human"]["core"]
    assert "Ccl21a" in types["Lymphatic_endothelial"]["mouse"]["core"]
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

    def fake_score_genes(adata, gene_list, score_name, use_raw=False, **kwargs):
        # Microglia core → high on cell 0; SMC core → high on cell 1.
        n = adata.n_obs
        gl = [str(g).lower() for g in gene_list]
        if any(g.startswith("p2ry") or g.startswith("tmem119") or g.startswith("hexb") for g in gl):
            adata.obs[score_name] = np.array([0.8, 0.1], dtype=float)[:n]
        elif any(g.startswith("myh11") or g.startswith("cnn1") or g.startswith("actg2") for g in gl):
            adata.obs[score_name] = np.array([0.05, 0.9], dtype=float)[:n]
        else:
            adata.obs[score_name] = np.zeros(n, dtype=float)

    fake_sc.tl.score_genes = fake_score_genes
    fake_sc.pp.normalize_total = MagicMock()
    fake_sc.pp.log1p = MagicMock()

    genes = ["P2ry12", "Tmem119", "Hexb", "Sall1", "Myh11", "Cnn1", "Actg2", "Smtn"]
    X = np.array(
        [
            [10, 8, 6, 4, 0, 0, 0, 0],
            [0, 0, 0, 0, 12, 9, 7, 5],
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
        "isp_expression_v2",
        "isp_expression_v2",
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

    genes = ["P2ry12", "Tmem119", "Hexb", "Sall1", "Myh11", "Cnn1", "Actg2", "Smtn"] + [
        f"D{i}" for i in range(10)
    ]
    X = np.ones((2, len(genes)), dtype=float) * 0.2
    X[0, 0:4] = 20
    X[1, 4:8] = 20
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

    # Without platform annotator, auto must NOT use user cell_type.
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
        prefer_metadata="auto",
    )
    assert list(annotated_user["celltype_source"]) == ["none", "none"]
    assert list(annotated_user["pred_cell_type"]) == ["Unknown", "Unknown"]

    # With platform annotator, auto uses cell_type.
    base_plat = base_user.copy()
    base_plat["celltype_annotator"] = "isp_expression_v1"
    assert resolve_prefer_metadata_flag("auto", base_plat) is True
    annotated_plat = annotate_dataframe_with_cell_types(
        base_plat,
        prefer_metadata="auto",
    )
    assert list(annotated_plat["pred_cell_type"]) == ["Neuron", "Astrocyte"]
    assert list(annotated_plat["celltype_source"]) == ["platform", "platform"]


def test_prefer_metadata_auto_mixed_platform_and_user():
    from isp_umap_celltype import apply_metadata_cell_types

    df = pd.DataFrame(
        {
            "cell_type": ["Neuron", "Astrocyte"],
            "celltype_annotator": ["isp_expression_v1", "user_manual"],
        }
    )
    out = apply_metadata_cell_types(df, prefer_metadata="auto")
    assert list(out["pred_cell_type"]) == ["Neuron", "Unknown"]
    assert list(out["celltype_source"]) == ["platform", "none"]


def test_prefer_metadata_false_ignores_platform_labels():
    from isp_umap_celltype import apply_metadata_cell_types

    df = pd.DataFrame(
        {
            "cell_type": ["Neuron"],
            "celltype_annotator": ["isp_expression_v1"],
        }
    )
    out = apply_metadata_cell_types(df, prefer_metadata=False)
    assert list(out["pred_cell_type"]) == ["Unknown"]
    assert list(out["celltype_source"]) == ["none"]


def test_yaml_defaults_wire_pre_isp_annotation():
    import yaml

    root = Path(__file__).resolve().parents[1]
    tok = yaml.safe_load((root / "core/config/tokenize.yaml").read_text())
    assert tok["tokenizer"]["celltype_annotation"]["enabled"] is True
    attrs = tok["tokenizer"]["custom_attr_name_dict"]
    for col in (
        "cell_type",
        "tissue",
        "celltype_score",
        "celltype_margin",
        "celltype_core_hits",
        "celltype_runner_up",
        "celltype_doublet_flag",
        "celltype_state",
        "celltype_confidence",
        "celltype_annotator",
    ):
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


def _core_genes(spec: dict, key: str) -> list[str]:
    block = spec.get(key) or spec.get("mouse") or []
    if isinstance(block, dict):
        return list(block.get("core") or [])
    return list(block)


def _synthetic_marker_adata(organism: str = "mouse"):
    anndata = pytest.importorskip("anndata")
    from celltype_annotate_expression import load_panel

    panel = load_panel()
    types = panel["types"]
    key = "human" if organism == "human" else "mouse"
    genes: list[str] = []
    for spec in types.values():
        genes.extend(_core_genes(spec, key))
    genes = list(dict.fromkeys(genes))
    decoys = [f"Decoy{i}" for i in range(80)]
    all_genes = genes + decoys
    n = len(all_genes)
    idx = {g: i for i, g in enumerate(all_genes)}
    X = np.ones((3, n), dtype=float) * 0.2
    for g in _core_genes(types["Microglia"], key):
        X[0, idx[g]] = 80
    for g in _core_genes(types["Vascular_SMC"], key):
        X[1, idx[g]] = 80
    for g in _core_genes(types["Neuron"], key):
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
    assert (out.obs["celltype_annotator"].astype(str) == "isp_expression_v2").all()
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
    human_mg = panel["types"]["Microglia"]["human"]
    assert "P2RY12" in human_mg["core"]
    assert "F13A1" not in panel["types"]["Monocyte"]["mouse"]["core"]
    assert "Pneumocyte" not in panel["types"]
    assert "Alveolar_AT1" in panel["types"] and "Alveolar_AT2" in panel["types"]


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
    assert cfg["custom_attr_name_dict"]["celltype_doublet_flag"] == "celltype_doublet_flag"


def test_doublet_and_sparse_core_hits():
    """Cross-compartment co-expression is not a cell type; one core gene is not enough."""
    anndata = pytest.importorskip("anndata")
    import celltype_annotate_expression as mod

    panel = {
        "version": "isp_expression_v2",
        "min_score": 0.15,
        "min_margin": 0.05,
        "min_core_detected": 2,
        "types": {
            "T_cell": {
                "tissue": "Immune",
                "compartment": "immune",
                "mouse": {"core": ["Cd3d", "Cd3e", "Cd3g"]},
            },
            "Alveolar_AT2": {
                "tissue": "Epithelial",
                "compartment": "epithelial",
                "mouse": {"core": ["Sftpc", "Sftpb", "Abca3"]},
            },
            "Erythroid": {
                "tissue": "Blood",
                "compartment": "blood",
                "min_core_detected": 3,
                "mouse": {"core": ["Hbb-bs", "Hba-a1", "Alas2"]},
            },
        },
    }
    genes = ["Cd3d", "Cd3e", "Cd3g", "Sftpc", "Sftpb", "Abca3", "Hbb-bs", "Hba-a1", "Alas2"]
    X = np.zeros((3, len(genes)), dtype=float)
    X[0, 0:3] = 10
    X[1, 0:6] = 10
    X[2, 6] = 50
    adata = anndata.AnnData(X=X, var=pd.DataFrame(index=genes), obs=pd.DataFrame(index=["t", "mix", "hbb"]))

    fake_sc = MagicMock()
    fake_sc.pp.normalize_total = MagicMock()
    fake_sc.pp.log1p = MagicMock()

    def fake_score_genes(adata, gene_list, score_name, use_raw=False, **kwargs):
        adata.obs[score_name] = np.full(adata.n_obs, 0.9, dtype=float)

    fake_sc.tl.score_genes = fake_score_genes
    saved = sys.modules.get("scanpy")
    sys.modules["scanpy"] = fake_sc
    try:
        out = mod.annotate_adata_cell_types(adata, organism="mouse", panel=panel, inplace=True)
    finally:
        if saved is None:
            sys.modules.pop("scanpy", None)
        else:
            sys.modules["scanpy"] = saved

    labels = list(out.obs["cell_type"].astype(str))
    assert labels[0] == "T_cell"
    assert labels[1] == "Doublet_suspected"
    assert labels[2] == "Ambiguous"
    assert list(out.obs["celltype_doublet_flag"].astype(str)) == ["0", "1", "0"]
    assert "T_cell" in str(out.obs["celltype_runner_up"].iloc[1])
    assert "Alveolar_AT2" in str(out.obs["celltype_runner_up"].iloc[1])


def _mock_score_genes_constant(mod, adata, panel, value=0.9):
    fake_sc = MagicMock()
    fake_sc.pp.normalize_total = MagicMock()
    fake_sc.pp.log1p = MagicMock()

    def fake_score_genes(adata, gene_list, score_name, use_raw=False, **kwargs):
        adata.obs[score_name] = np.full(adata.n_obs, value, dtype=float)

    fake_sc.tl.score_genes = fake_score_genes
    saved = sys.modules.get("scanpy")
    sys.modules["scanpy"] = fake_sc
    try:
        return mod.annotate_adata_cell_types(adata, organism="mouse", panel=panel, inplace=True)
    finally:
        if saved is None:
            sys.modules.pop("scanpy", None)
        else:
            sys.modules["scanpy"] = saved


def test_fiber_program_is_state_not_cell_type():
    anndata = pytest.importorskip("anndata")
    import celltype_annotate_expression as mod

    panel = {
        "version": "isp_expression_v2",
        "min_score": 0.15,
        "min_margin": 0.05,
        "min_core_detected": 2,
        "types": {
            "Skeletal_myocyte": {
                "tissue": "Muscle",
                "compartment": "muscle",
                "mouse": {"core": ["Acta1", "Ckm", "Des"]},
                "states": {
                    "fiber_program": {
                        "min_detected": 2,
                        "fast": {"mouse": {"core": ["Myh1", "Myh4", "Tnnc2"]}},
                        "slow": {"mouse": {"core": ["Myh7", "Tnnt1", "Tnni1"]}},
                    }
                },
            }
        },
    }
    genes = ["Acta1", "Ckm", "Des", "Myh1", "Myh4", "Tnnc2", "Myh7", "Tnnt1", "Tnni1"]
    X = np.zeros((3, len(genes)), dtype=float)
    X[0, 0:6] = 10
    X[1, :] = 10
    X[2, 0:3] = 10
    adata = anndata.AnnData(X=X, var=pd.DataFrame(index=genes), obs=pd.DataFrame(index=["fast", "mix", "plain"]))
    out = _mock_score_genes_constant(mod, adata, panel)
    assert list(out.obs["cell_type"].astype(str)) == ["Skeletal_myocyte"] * 3
    assert list(out.obs["celltype_state"].astype(str)) == [
        "fiber_program=fast",
        "fiber_program=mixed",
        "fiber_program=unresolved",
    ]


def test_c1qc_macrophage_requires_and_neural_prior():
    anndata = pytest.importorskip("anndata")
    import celltype_annotate_expression as mod

    panel = {
        "version": "isp_expression_v2",
        "min_score": 0.15,
        "min_margin": 0.05,
        "min_core_detected": 2,
        "gates": {"myeloid": {"min_detected": 2, "mouse": ["Ptprc", "Tyrobp", "Aif1"]}},
        "types": {
            "Microglia": {
                "tissue": "Neural",
                "compartment": "immune",
                "mouse": {"core": ["P2ry12", "Tmem119", "Hexb"]},
            },
            "C1QC_APOE_macrophage": {
                "tissue": "Immune",
                "compartment": "immune",
                "requires": {
                    "myeloid_compartment": True,
                    "microglia_core_negative_or_low": True,
                },
                "mouse": {"core": ["C1qa", "C1qb", "C1qc", "Apoe"]},
            },
        },
    }
    genes = ["P2ry12", "Tmem119", "Hexb", "C1qa", "C1qb", "C1qc", "Apoe", "Ptprc", "Tyrobp", "Aif1"]
    X = np.zeros((4, len(genes)), dtype=float)
    X[0, 0:3] = 10
    X[0, 7:10] = 10
    X[1, 3:10] = 10
    X[2, 3:7] = 10
    X[3, 0] = 5
    X[3, 3:10] = 10
    obs = pd.DataFrame({"organ_major": ["", "", "", "brain"]}, index=["mg", "c1q", "no_my", "dam"])
    adata = anndata.AnnData(X=X, var=pd.DataFrame(index=genes), obs=obs)
    out = _mock_score_genes_constant(mod, adata, panel)
    assert list(out.obs["cell_type"].astype(str)) == [
        "Microglia",
        "C1QC_APOE_macrophage",
        "Ambiguous",
        "Microglia",
    ]


def test_vocab_fraction_low_when_half_missing():
    from celltype_annotate_expression import vocab_fraction_is_low

    assert vocab_fraction_is_low(3, 7)
    assert not vocab_fraction_is_low(4, 7)
    assert vocab_fraction_is_low(2, 4)


def test_negative_epithelial_gate_does_not_reject():
    anndata = pytest.importorskip("anndata")
    import celltype_annotate_expression as mod

    panel = {
        "version": "isp_expression_v2",
        "min_score": 0.2,
        "min_margin": 0.05,
        "min_core_detected": 2,
        "support_weight": 0.35,
        "types": {
            "Epithelial_general": {
                "tissue": "Epithelial",
                "compartment": "epithelial",
                "assign": False,
                "mouse": {"core": ["Epcam", "Krt8", "Krt18"]},
            },
            "Alveolar_AT2": {
                "tissue": "Epithelial",
                "compartment": "epithelial",
                "mouse": {"core": ["Sftpc", "Sftpb", "Abca3"]},
            },
        },
    }
    genes = ["Epcam", "Krt8", "Krt18", "Sftpc", "Sftpb", "Abca3"]
    X = np.ones((1, len(genes)), dtype=float)
    adata = anndata.AnnData(X=X, var=pd.DataFrame(index=genes))
    fake_sc = MagicMock()
    fake_sc.pp.normalize_total = MagicMock()
    fake_sc.pp.log1p = MagicMock()

    def fake_score_genes(adata, gene_list, score_name, use_raw=False, **kwargs):
        gl = [str(g).lower() for g in gene_list]
        value = -1.0 if any(g.startswith("epcam") or g.startswith("krt") for g in gl) else 0.4
        adata.obs[score_name] = np.full(adata.n_obs, value, dtype=float)

    fake_sc.tl.score_genes = fake_score_genes
    saved = sys.modules.get("scanpy")
    sys.modules["scanpy"] = fake_sc
    try:
        out = mod.annotate_adata_cell_types(adata, organism="mouse", panel=panel, inplace=True)
    finally:
        if saved is None:
            sys.modules.pop("scanpy", None)
        else:
            sys.modules["scanpy"] = saved
    assert list(out.obs["cell_type"].astype(str)) == ["Alveolar_AT2"]


def test_cross_compartment_mid_score_is_ambiguous_and_mural_state():
    anndata = pytest.importorskip("anndata")
    import celltype_annotate_expression as mod

    panel = {
        "version": "isp_expression_v2",
        "min_score": 0.2,
        "min_margin": 0.05,
        "min_core_detected": 2,
        "types": {
            "Myofibroblast": {
                "tissue": "Stromal",
                "compartment": "stromal",
                "mouse": {"core": ["Col1a1", "Dcn", "Acta2"]},
            },
            "Pericyte": {
                "tissue": "Vascular",
                "compartment": "stromal",
                "mouse": {"core": ["Rgs5", "Pdgfrb", "Abcc9"]},
            },
            "T_cell": {
                "tissue": "Immune",
                "compartment": "immune",
                "mouse": {"core": ["Cd3d", "Cd3e", "Cd3g"]},
            },
            "Alveolar_AT2": {
                "tissue": "Epithelial",
                "compartment": "epithelial",
                "mouse": {"core": ["Sftpc", "Sftpb", "Abca3"]},
            },
        },
    }
    genes = ["Col1a1", "Dcn", "Acta2", "Rgs5", "Pdgfrb", "Abcc9", "Cd3d", "Cd3e", "Cd3g", "Sftpc", "Sftpb", "Abca3"]
    X = np.ones((2, len(genes)), dtype=float)
    adata = anndata.AnnData(X=X, var=pd.DataFrame(index=genes), obs=pd.DataFrame(index=["mural", "cross"]))
    fake_sc = MagicMock()
    fake_sc.pp.normalize_total = MagicMock()
    fake_sc.pp.log1p = MagicMock()

    def fake_score_genes(adata, gene_list, score_name, use_raw=False, **kwargs):
        gl = [str(g).lower() for g in gene_list]
        if any(g.startswith("col1") or g.startswith("dcn") or g.startswith("acta") for g in gl):
            values = np.array([0.9, 0.0])
        elif any(g.startswith("rgs5") or g.startswith("pdgfrb") for g in gl):
            values = np.array([0.4, 0.0])
        elif any(g.startswith("cd3") for g in gl):
            values = np.array([0.0, 0.9])
        else:
            values = np.array([0.0, 0.5])
        adata.obs[score_name] = values

    fake_sc.tl.score_genes = fake_score_genes
    saved = sys.modules.get("scanpy")
    sys.modules["scanpy"] = fake_sc
    try:
        out = mod.annotate_adata_cell_types(adata, organism="mouse", panel=panel, inplace=True)
    finally:
        if saved is None:
            sys.modules.pop("scanpy", None)
        else:
            sys.modules["scanpy"] = saved
    assert list(out.obs["cell_type"].astype(str)) == [
        "Myofibroblast",
        "ambiguous_cross_compartment",
    ]
    assert out.obs["celltype_state"].iloc[0] == "mural_program=Myofibroblast;alt=Pericyte"

