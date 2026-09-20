"""Tests for ISP UMAP cluster_coexpr_analysis postprocess gate and config propagation."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

# run_isp_umap pulls torch / transformers at import time; stub for unit tests.
for _mod in (
    "torch",
    "transformers",
    "datasets",
    "umap_plot_style",
    "geneformer",
    "geneformer.species_context",
    "geneformer.gene_converter",
    "geneformer.tokenizer",
    "geneformer.in_silico_perturber_stats",
):
    sys.modules.setdefault(_mod, MagicMock())

from isp_umap_celltype import (  # noqa: E402
    annotate_dataframe_with_cell_types,
    predict_cell_types_from_input_ids,
)
from run_isp_umap import (  # noqa: E402
    DEFAULT_POSTPROCESS_CFG,
    build_isp_umap_config,
    postprocess_is_enabled,
    postprocess_wants_celltype,
    run_downstream_plots,
)


def test_postprocess_default_off():
    assert DEFAULT_POSTPROCESS_CFG["enabled"] is False
    assert postprocess_is_enabled({}) is False
    assert postprocess_is_enabled({"postprocess": {}}) is False
    assert postprocess_is_enabled({"postprocess": False}) is False
    assert postprocess_is_enabled({"postprocess": {"enabled": False}}) is False
    assert postprocess_is_enabled({"postprocess": {"enabled": True}}) is True


def test_postprocess_wants_celltype_requires_enabled():
    assert (
        postprocess_wants_celltype(
            {"postprocess": {"enabled": False, "celltype_prediction": True}}
        )
        is False
    )
    assert (
        postprocess_wants_celltype(
            {"postprocess": {"enabled": True, "celltype_prediction": True}}
        )
        is True
    )
    assert (
        postprocess_wants_celltype(
            {"postprocess": {"enabled": True, "celltype_prediction": False}}
        )
        is False
    )


def test_build_isp_umap_config_copies_postprocess():
    isp_cfg = {
        "paths": {"dataset": "/tmp/ds", "geneformer_model": "/tmp/model"},
        "perturbation": {"start_state": "AD", "end_state": "WT", "type": "delete"},
        "umap": {"seed": 7},
        "postprocess": {"enabled": True, "n_clusters": 6},
        "model": {"num_classes": 2},
    }
    out = build_isp_umap_config(isp_cfg, ["Igfbp2"])
    assert out["postprocess"]["enabled"] is True
    assert out["postprocess"]["n_clusters"] == 6
    assert out["postprocess"]["celltype_prediction"] is True
    assert out["umap"]["seed"] == 7


def test_build_isp_umap_config_postprocess_default_off():
    isp_cfg = {
        "paths": {"dataset": "/tmp/ds", "geneformer_model": "/tmp/model"},
        "perturbation": {"start_state": "AD", "end_state": "WT"},
        "model": {"num_classes": 2},
    }
    out = build_isp_umap_config(isp_cfg, "Igfbp2")
    assert out["postprocess"]["enabled"] is False
    assert out["postprocess"]["n_clusters"] == 4


def test_run_downstream_plots_skips_when_disabled(tmp_path):
    joint_mod = MagicMock()
    l2_mod = MagicMock()
    with patch.dict(
        sys.modules,
        {
            "plot_isp_umap_joint_overlays": joint_mod,
            "plot_l2_by_coarse_celltype": l2_mod,
        },
    ):
        run_downstream_plots(tmp_path, "Igfbp2", {"postprocess": {"enabled": False}})
        joint_mod.run_joint_overlays.assert_not_called()
        l2_mod.run_l2_by_group.assert_not_called()


def test_run_downstream_plots_calls_when_enabled(tmp_path):
    joint_mod = MagicMock()
    l2_mod = MagicMock()
    with patch.dict(
        sys.modules,
        {
            "plot_isp_umap_joint_overlays": joint_mod,
            "plot_l2_by_coarse_celltype": l2_mod,
        },
    ):
        run_downstream_plots(
            tmp_path,
            "Igfbp2",
            {
                "postprocess": {"enabled": True, "n_clusters": 3},
                "umap": {
                    "n_neighbors": 10,
                    "min_dist": 0.2,
                    "seed": 1,
                    "num_trajectory_arrows": 5,
                },
            },
        )
        joint_mod.run_joint_overlays.assert_called_once()
        kwargs = joint_mod.run_joint_overlays.call_args.kwargs
        assert kwargs["gene"] == "Igfbp2"
        assert kwargs["n_clusters"] == 3
        assert kwargs["n_neighbors"] == 10
        l2_mod.run_l2_by_group.assert_called_once()


def test_celltype_scoring_with_mock_dicts():
    name_id = {"Acta2": "ENS1", "Myh11": "ENS2", "Tagln": "ENS3", "Cx3cr1": "ENS10"}
    token_dict = {"ENS1": 101, "ENS2": 102, "ENS3": 103, "ENS10": 110}
    markers = {"Vascular_SMC": ["Acta2", "Myh11", "Tagln"], "Microglia": ["Cx3cr1"]}
    ids = [[101, 102, 103, 999], [110, 999]]
    pred = predict_cell_types_from_input_ids(
        ids, markers=markers, token_dict=token_dict, name_id=name_id
    )
    assert len(pred) == 2
    assert pred.loc[0, "pred_cell_type"] == "Vascular_SMC"
    assert pred.loc[0, "score_Vascular_SMC"] == 1.0
    assert "coarse_type" in pred.columns

    base = pd.DataFrame({"cell_index": [0, 1], "shift_l2": [1.0, 0.5]})
    annotated = annotate_dataframe_with_cell_types(
        base, ids, markers=markers, token_dict=token_dict, name_id=name_id
    )
    assert "pred_cell_type" in annotated.columns
    assert list(annotated["cell_index"]) == [0, 1]
