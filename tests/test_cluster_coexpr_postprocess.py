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
    traj_mod = MagicMock()
    with patch.dict(
        sys.modules,
        {
            "plot_isp_umap_joint_overlays": joint_mod,
            "plot_isp_umap_celltype_trajectories": traj_mod,
        },
    ):
        run_downstream_plots(tmp_path, "Igfbp2", {"postprocess": {"enabled": False}})
        joint_mod.run_joint_overlays.assert_not_called()
        traj_mod.run_celltype_trajectory_plots.assert_not_called()


def test_run_downstream_plots_calls_when_enabled(tmp_path):
    joint_mod = MagicMock()
    traj_mod = MagicMock()
    with patch.dict(
        sys.modules,
        {
            "plot_isp_umap_joint_overlays": joint_mod,
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
        traj_mod.run_celltype_trajectory_plots.assert_called_once()


def test_run_downstream_plots_passes_auto_n_clusters(tmp_path):
    joint_mod = MagicMock()
    traj_mod = MagicMock()
    with patch.dict(
        sys.modules,
        {
            "plot_isp_umap_joint_overlays": joint_mod,
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


def test_celltype_preisp_only_no_token_markers():
    import pytest

    with pytest.raises(RuntimeError, match="brain cell-type panels were removed"):
        predict_cell_types_from_input_ids([[1, 2]])

    # Without platform annotator → Unknown
    base = pd.DataFrame(
        {
            "cell_index": [0, 1],
            "shift_l2": [1.0, 0.5],
            "cell_type": ["Neuron", "Astrocyte"],
        }
    )
    annotated = annotate_dataframe_with_cell_types(base, prefer_metadata="auto")
    assert list(annotated["pred_cell_type"]) == ["Unknown", "Unknown"]
    assert list(annotated["celltype_source"]) == ["none", "none"]

    # With isp_expression_v1 → platform labels
    base["celltype_annotator"] = "isp_expression_v1"
    annotated = annotate_dataframe_with_cell_types(base, prefer_metadata="auto")
    assert list(annotated["pred_cell_type"]) == ["Neuron", "Astrocyte"]
    assert list(annotated["celltype_source"]) == ["platform", "platform"]
