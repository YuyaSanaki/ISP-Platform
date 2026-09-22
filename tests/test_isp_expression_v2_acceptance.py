"""Acceptance tests: isp_expression_v2 rules are enforced by the annotator."""

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

MYELOID = ["PTPRC", "TYROBP", "AIF1"]
_DOUBLET = {"Doublet_suspected", "ambiguous_cross_compartment", "Ambiguous"}


def _human_gene_index() -> list[str]:
    from celltype_annotate_expression import load_panel

    panel = load_panel()
    genes: list[str] = []
    for spec in panel["types"].values():
        block = spec.get("human") or {}
        if isinstance(block, dict):
            genes.extend(block.get("core") or [])
            genes.extend(block.get("support") or [])
            genes.extend(block.get("anti") or [])
    genes.extend(panel["gates"]["myeloid"]["human"])
    fiber = panel["types"]["Skeletal_myocyte"]["states"]["fiber_program"]
    genes.extend(fiber["fast"]["human"]["core"])
    genes.extend(fiber["slow"]["human"]["core"])
    return list(dict.fromkeys(genes))


def _annotate_programs(programs: dict[str, list[str]]):
    anndata = pytest.importorskip("anndata")
    import celltype_annotate_expression as mod

    genes = _human_gene_index()
    missing = sorted({g for program in programs.values() for g in program if g not in genes})
    assert not missing, missing
    names = list(programs)
    X = np.zeros((len(names), len(genes)), dtype=float)
    idx = {g: i for i, g in enumerate(genes)}
    for row, name in enumerate(names):
        for gene in programs[name]:
            X[row, idx[gene]] = 10.0
    adata = anndata.AnnData(X=X, var=pd.DataFrame(index=genes), obs=pd.DataFrame(index=names))

    fake_sc = MagicMock()
    fake_sc.pp.normalize_total = MagicMock()
    fake_sc.pp.log1p = MagicMock()

    def fake_score_genes(adata, gene_list, score_name, use_raw=False, **kwargs):
        cols = [adata.var_names.get_loc(g) for g in gene_list]
        values = np.asarray(adata.X[:, cols], dtype=float).mean(axis=1)
        adata.obs[score_name] = values

    fake_sc.tl.score_genes = fake_score_genes
    saved = sys.modules.get("scanpy")
    sys.modules["scanpy"] = fake_sc
    try:
        out = mod.annotate_adata_cell_types(adata, organism="human", inplace=True)
    finally:
        if saved is None:
            sys.modules.pop("scanpy", None)
        else:
            sys.modules["scanpy"] = saved
    return out


def test_panel_gates_and_overrides_are_declared():
    from celltype_annotate_expression import load_panel

    panel = load_panel()
    types = panel["types"]
    gated = {
        "Monocyte",
        "Macrophage",
        "C1QC_APOE_macrophage",
        "cDC1",
        "cDC2",
        "pDC",
        "Neutrophil",
        "Mast_cell",
        "Microglia",
    }
    for name in gated:
        assert types[name]["requires"]["myeloid_compartment"] is True
    for name in ("T_cell", "B_cell", "NK_cell", "Plasma_cell"):
        assert "myeloid_compartment" not in (types[name].get("requires") or {})
    assert types["C1QC_APOE_macrophage"]["requires"]["microglia_core_negative_or_low"] is True
    assert types["Microglia"]["requires"]["microglia_core_rescue"] is True
    assert types["Epithelial_general"]["assign"] is False
    for name in (
        "Pericyte",
        "Vascular_SMC",
        "Fibroblast",
        "Myofibroblast",
        "Lymphatic_endothelial",
        "Hepatocyte",
        "Alveolar_AT1",
        "Enterocyte",
        "Erythroid",
        "Megakaryocyte",
        "Choroid_plexus",
    ):
        assert types[name]["min_core_detected"] == 3
    assert "min_core_detected" not in types["T_cell"]
    assert panel["min_core_detected"] == 2


def test_acceptance_programs_follow_v2_rules():
    programs = {
        "t": ["CD3D", "CD3E", "TRAC", "LCK"],
        "b": ["CD79A", "CD79B", "MS4A1", "CD74"],
        "plasma": ["JCHAIN", "MZB1", "SDC1", "XBP1"],
        "cdc1": ["CLEC9A", "XCR1", "BATF3", "IRF8", *MYELOID],
        "cdc2": ["CD1C", "CLEC10A", "FCER1A", *MYELOID],
        "pdc": ["GZMB", "IL3RA", "LILRA4", "CLEC4C", "TCF4", *MYELOID],
        "microglia": ["P2RY12", "TMEM119", "SALL1", "HEXB", "GPR34"],
        "c1q": ["C1QA", "C1QB", "C1QC", "APOE", "MERTK", *MYELOID],
        "hep": ["ALB", "APOA1", "APOB", "FGA", "FGB"],
        "cp": ["TTR", "AQP1", "FOLR1", "KCNJ13", "OTX2"],
        "at2": ["SFTPC", "SFTPA1", "SFTPB", "ABCA3"],
        "at1": ["AGER", "CAV1", "EMP2", "PDPN", "HOPX"],
        "lec": ["PROX1", "PDPN", "FLT4", "CCL21", "PECAM1", "CDH5"],
        "fib": ["DCN", "LUM", "COL1A1", "COL3A1", "DPT"],
        "myofib": ["COL1A1", "DCN", "ACTA2", "TAGLN"],
        "pericyte": ["RGS5", "PDGFRB", "CSPG4", "ABCC9", "KCNJ8"],
        "smc": ["MYH11", "CNN1", "ACTG2", "SMTN", "LMOD1"],
        "hbb_only": ["CD3D", "CD3E", "TRAC", "LCK", "HBB"],
        "pf4_only": ["CD3D", "CD3E", "TRAC", "LCK", "PF4", "PPBP"],
        "t_b": ["CD3D", "CD3E", "TRAC", "LCK", "CD79A", "CD79B", "MS4A1", "CD74"],
        "skel_fast": ["ACTA1", "CKM", "DES", "TTN", "MYLPF", "MYH1", "MYH2", "TNNC2"],
        "skel_mixed": [
            "ACTA1",
            "CKM",
            "DES",
            "TTN",
            "MYLPF",
            "MYH1",
            "MYH2",
            "TNNC2",
            "MYH7",
            "TNNT1",
            "TNNI1",
        ],
        "neutrophil_no_gate": ["FCGR3B", "CSF3R", "CXCR2", "CEACAM8", "S100A8", "S100A9"],
        "nk": ["NKG7", "KLRD1", "KLRF1", "GNLY", "PRF1", "GZMB"],
        "fib_two": ["DCN", "LUM"],
        "at2_epi": ["SFTPC", "SFTPA1", "SFTPB", "ABCA3", "EPCAM", "KRT8", "KRT18", "CDH1"],
    }
    out = _annotate_programs(programs)
    labels = out.obs["cell_type"].astype(str)
    states = out.obs["celltype_state"].astype(str)

    expected = {
        "t": "T_cell",
        "b": "B_cell",
        "plasma": "Plasma_cell",
        "cdc1": "cDC1",
        "cdc2": "cDC2",
        "pdc": "pDC",
        "microglia": "Microglia",
        "c1q": "C1QC_APOE_macrophage",
        "hep": "Hepatocyte",
        "cp": "Choroid_plexus",
        "at2": "Alveolar_AT2",
        "at1": "Alveolar_AT1",
        "lec": "Lymphatic_endothelial",
        "fib": "Fibroblast",
        "myofib": "Myofibroblast",
        "pericyte": "Pericyte",
        "smc": "Vascular_SMC",
        "nk": "NK_cell",
        "at2_epi": "Alveolar_AT2",
    }
    for name, label in expected.items():
        assert labels.loc[name] == label, f"{name}: {labels.loc[name]}"

    assert int(out.obs.loc["t", "celltype_core_hits"]) >= 2
    assert labels.loc["microglia"] != "C1QC_APOE_macrophage"
    assert labels.loc["hbb_only"] != "Erythroid"
    assert labels.loc["hbb_only"] == "T_cell"
    assert labels.loc["pf4_only"] != "Megakaryocyte"
    assert labels.loc["pf4_only"] == "T_cell"
    assert labels.loc["t_b"] in _DOUBLET
    assert labels.loc["t_b"] not in {"T_cell", "B_cell"}
    assert labels.loc["neutrophil_no_gate"] != "Neutrophil"
    assert labels.loc["fib_two"] != "Fibroblast"
    assert labels.loc["skel_fast"] == "Skeletal_myocyte"
    assert labels.loc["skel_mixed"] == "Skeletal_myocyte"
    assert "Fast_myofiber" not in set(labels)
    assert states.loc["skel_fast"] == "fiber_program=fast"
    assert states.loc["skel_mixed"] == "fiber_program=mixed"
    assert labels.loc["at2_epi"] != "Epithelial_general"
