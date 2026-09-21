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
    traj_mod = MagicMock()
    with patch.dict(
        sys.modules,
        {
            "plot_isp_umap_joint_overlays": joint_mod,
            "plot_l2_by_coarse_celltype": l2_mod,
            "plot_isp_umap_celltype_trajectories": traj_mod,
        },
    ):
        run_downstream_plots(tmp_path, "Igfbp2", {"postprocess": {"enabled": False}})
        joint_mod.run_joint_overlays.assert_not_called()
        l2_mod.run_l2_by_group.assert_not_called()
        traj_mod.run_celltype_trajectory_plots.assert_not_called()


def test_run_downstream_plots_calls_when_enabled(tmp_path):
    joint_mod = MagicMock()
    l2_mod = MagicMock()
    traj_mod = MagicMock()
    with patch.dict(
        sys.modules,
        {
            "plot_isp_umap_joint_overlays": joint_mod,
            "plot_l2_by_coarse_celltype": l2_mod,
            "plot_isp_umap_celltype_trajectories": traj_mod,
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
        traj_mod.run_celltype_trajectory_plots.assert_called_once()
        traj_kwargs = traj_mod.run_celltype_trajectory_plots.call_args.kwargs
        assert traj_kwargs["num_trajectory_arrows"] == 5
        assert traj_kwargs["seed"] == 1


def test_run_downstream_plots_passes_auto_n_clusters(tmp_path):
    joint_mod = MagicMock()
    l2_mod = MagicMock()
    traj_mod = MagicMock()
    with patch.dict(
        sys.modules,
        {
            "plot_isp_umap_joint_overlays": joint_mod,
            "plot_l2_by_coarse_celltype": l2_mod,
            "plot_isp_umap_celltype_trajectories": traj_mod,
        },
    ):
        run_downstream_plots(
            tmp_path,
            "Igfbp2",
            {"postprocess": {"enabled": True, "n_clusters": "auto"}},
        )
        kwargs = joint_mod.run_joint_overlays.call_args.kwargs
        assert kwargs["n_clusters"] == "auto"
        traj_mod.run_celltype_trajectory_plots.assert_called_once()


def test_resolve_n_clusters_auto_picks_true_k():
    import numpy as np

    # plot_isp_umap_joint_overlays pulls matplotlib/seaborn at import time.
    for _mod in (
        "seaborn",
        "matplotlib",
        "matplotlib.pyplot",
        "isp_umap_postprocess_style",
    ):
        sys.modules.setdefault(_mod, MagicMock())

    # Module-level torch MagicMock breaks scipy/sklearn; unstub for this test.
    torch_stub = sys.modules.pop("torch", None)
    try:
        from plot_isp_umap_joint_overlays import resolve_n_clusters

        rng = np.random.default_rng(0)
        # Three well-separated blobs → silhouette should prefer k=3.
        centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        X = np.vstack([c + rng.normal(0, 0.3, size=(40, 2)) for c in centers])
        assert resolve_n_clusters(5, X, seed=0) == 5
        assert resolve_n_clusters("auto", X, seed=0, k_min=2, k_max=6) == 3
    finally:
        if torch_stub is not None:
            sys.modules["torch"] = torch_stub


def test_celltype_scoring_with_mock_dicts():
    name_id = {"Acta2": "ENS1", "Myh11": "ENS2", "Tagln": "ENS3", "Cx3cr1": "ENS10"}
    token_dict = {"ENS1": 101, "ENS2": 102, "ENS3": 103, "ENS10": 110}
    markers = {"Vascular_SMC": ["Acta2", "Myh11", "Tagln"], "Microglia": ["Cx3cr1"]}
    ids = [[101, 102, 103, 999], [110, 999]]
    pred = predict_cell_types_from_input_ids(
        ids,
        markers=markers,
        token_dict=token_dict,
        name_id=name_id,
        organism="mouse",
        use_rank_weights=False,
    )
    assert len(pred) == 2
    assert pred.loc[0, "pred_cell_type"] == "Vascular_SMC"
    assert pred.loc[0, "score_Vascular_SMC"] == 1.0
    assert "coarse_type" in pred.columns

    base = pd.DataFrame({"cell_index": [0, 1], "shift_l2": [1.0, 0.5]})
    annotated = annotate_dataframe_with_cell_types(
        base,
        ids,
        markers=markers,
        token_dict=token_dict,
        name_id=name_id,
        organism="mouse",
        use_rank_weights=False,
        prefer_metadata=False,
    )
    assert "pred_cell_type" in annotated.columns
    assert list(annotated["cell_index"]) == [0, 1]


def test_human_panel_case_lookup_and_rank_weights():
    from isp_umap_celltype import select_marker_panel

    human = select_marker_panel("human")
    assert human["Microglia"][0] == "CX3CR1"
    name_id = {"CX3CR1": "ENSG1", "P2RY12": "ENSG2", "ACTA2": "ENSG10", "MYH11": "ENSG11"}
    token_dict = {"ENSG1": 201, "ENSG2": 202, "ENSG10": 210, "ENSG11": 211}
    markers = {"Microglia": ["CX3CR1", "P2RY12"], "Vascular_SMC": ["ACTA2", "MYH11"]}
    # Microglia markers at top ranks vs SMC buried — rank weights should favor Microglia.
    ids = [[201, 202, 999, 210, 211]]
    pred = predict_cell_types_from_input_ids(
        ids,
        markers=markers,
        token_dict=token_dict,
        name_id=name_id,
        organism="human",
        use_rank_weights=True,
    )
    assert pred.loc[0, "pred_cell_type"] == "Microglia"
    assert pred.loc[0, "score_Microglia"] > pred.loc[0, "score_Vascular_SMC"]


def test_prefer_metadata_cell_type():
    name_id = {"Acta2": "ENS1", "Cx3cr1": "ENS10"}
    token_dict = {"ENS1": 101, "ENS10": 110}
    markers = {"Vascular_SMC": ["Acta2"], "Microglia": ["Cx3cr1"]}
    ids = [[101, 999], [110, 999]]
    base = pd.DataFrame(
        {
            "cell_index": [0, 1],
            "shift_l2": [1.0, 0.5],
            "cell_type": ["Neuron", "Astrocyte"],
        }
    )
    annotated = annotate_dataframe_with_cell_types(
        base,
        ids,
        markers=markers,
        token_dict=token_dict,
        name_id=name_id,
        organism="mouse",
        use_rank_weights=False,
        prefer_metadata=True,
    )
    assert list(annotated["pred_cell_type"]) == ["Neuron", "Astrocyte"]
    assert list(annotated["celltype_source"]) == ["metadata", "metadata"]
    assert list(annotated["celltype_plot"]) == ["Neuron", "Astrocyte"]


def test_default_postprocess_includes_celltype_options():
    assert DEFAULT_POSTPROCESS_CFG["prefer_metadata_celltype"] is True
    assert DEFAULT_POSTPROCESS_CFG["celltype_rank_weights"] is True
    out = build_isp_umap_config(
        {
            "paths": {"dataset": "/tmp/ds", "geneformer_model": "/tmp/model"},
            "perturbation": {"start_state": "AD", "end_state": "WT"},
            "model": {"num_classes": 2},
            "postprocess": {"enabled": True},
        },
        ["Igfbp2"],
    )
    assert out["postprocess"]["prefer_metadata_celltype"] is True
    assert out["postprocess"]["celltype_rank_weights"] is True

