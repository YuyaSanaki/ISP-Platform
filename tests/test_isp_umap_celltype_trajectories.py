"""Unit tests for ISP UMAP cell-type trajectory tracking summaries."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))


def _load_traj_with_stubs():
    """Load trajectory module with plotting deps stubbed (no torch/scipy import)."""
    l2_stub = MagicMock()
    l2_stub.GROUP_CANDIDATES = ("coarse_type", "pred_cell_type", "cell_type")
    style_stub = MagicMock()
    stubs = {
        "seaborn": MagicMock(),
        "matplotlib": MagicMock(),
        "matplotlib.pyplot": MagicMock(),
        "isp_umap_postprocess_style": style_stub,
        "plot_l2_by_coarse_celltype": l2_stub,
    }
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    sys.modules.pop("plot_isp_umap_celltype_trajectories", None)
    import plot_isp_umap_celltype_trajectories as traj_mod

    return traj_mod, saved


def _restore(saved):
    for k, prev in saved.items():
        if prev is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = prev
    sys.modules.pop("plot_isp_umap_celltype_trajectories", None)


def test_build_celltype_shift_summary_orders_by_mean_l2(tmp_path):
    traj_mod, saved = _load_traj_with_stubs()
    try:
        df = pd.DataFrame(
            {
                "pred_cell_type": ["A", "A", "B", "B"],
                "shift_l2": [1.0, 3.0, 0.5, 0.5],
                "umap_shift_l2": [0.2, 0.4, 0.1, 0.1],
                "umap1_before": [0.0, 1.0, 2.0, 3.0],
                "umap2_before": [0.0, 0.0, 0.0, 0.0],
                "umap1_after": [1.0, 2.0, 2.5, 3.5],
                "umap2_after": [1.0, 1.0, 0.0, 0.0],
            }
        )
        summary = traj_mod.build_celltype_shift_summary(df, "pred_cell_type")
        assert list(summary["cell_type"]) == ["A", "B"]
        assert summary.loc[0, "mean_shift_l2"] == 2.0
        assert summary.loc[0, "n_cells"] == 2
        assert abs(summary.loc[0, "mean_du1"] - 1.0) < 1e-9
        out = tmp_path / "celltype_shift_summary.csv"
        summary.to_csv(out, index=False)
        assert out.exists()
    finally:
        _restore(saved)


def test_resolve_xy_prefers_joint_coords():
    traj_mod, saved = _load_traj_with_stubs()
    try:
        df = pd.DataFrame(
            {
                "umap1_joint": [0.0, 1.0],
                "umap2_joint": [0.0, 0.0],
                "umap1_isp": [1.0, 2.0],
                "umap2_isp": [1.0, 1.0],
                "umap1_before": [9.0, 9.0],
                "umap2_before": [9.0, 9.0],
                "umap1_after": [8.0, 8.0],
                "umap2_after": [8.0, 8.0],
            }
        )
        before, after = traj_mod._resolve_xy(df)
        assert list(before[:, 0]) == [0.0, 1.0]
        assert list(after[:, 0]) == [1.0, 2.0]
    finally:
        _restore(saved)
