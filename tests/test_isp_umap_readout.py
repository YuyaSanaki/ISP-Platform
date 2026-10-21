"""ISP UMAP readout record, opt-in ISP-aligned readout, and overlays that follow the main projection."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

try:
    import run_isp_umap as riu

    DEPS_MISSING = None
except Exception as exc:  # pragma: no cover
    DEPS_MISSING = str(exc)


@unittest.skipIf(DEPS_MISSING is not None, f"deps missing: {DEPS_MISSING}")
class TestUmapReadout(unittest.TestCase):
    def test_default_readout_is_last_layer_fp32(self):
        r = riu.resolve_umap_readout({})
        self.assertEqual((r["readout"], r["hidden_index"], r["dtype"]), ("default", -1, "float32"))

    def test_isp_readout_follows_isp_emb_layer(self):
        r = riu.resolve_umap_readout({"readout": "isp", "isp_emb_layer": -1})
        self.assertEqual((r["hidden_index"], r["dtype"]), (-2, "bfloat16"))
        with self.assertRaises(ValueError):
            riu.resolve_umap_readout({"readout": "fp16"})

    def test_pipeline_config_carries_isp_layer_and_cap(self):
        cfg = riu.build_isp_umap_config(
            {
                "paths": {"dataset": "d", "geneformer_model": "m"},
                "perturbation": {"start_state": "AD", "end_state": "WT"},
                "isp": {"emb_layer": -1, "max_ncells": 500},
                "umap": {"readout": "isp"},
            },
            ["Igfbp2"],
        )
        self.assertEqual(cfg["umap"]["isp_emb_layer"], -1)
        self.assertEqual(cfg["umap"]["isp_max_ncells"], 500)
        self.assertEqual(riu.resolve_umap_readout(cfg["umap"])["hidden_index"], -2)

    def test_cosine_shift_toward_end(self):
        end = np.array([[1.0, 0.0], [1.0, 0.0]])
        start = np.array([[0.0, 1.0]])
        pert = np.array([[1.0, 1.0]])
        shift = riu.compute_cosine_shift_toward_end(start, pert, end)
        self.assertAlmostEqual(float(shift[0]), 1 / np.sqrt(2), places=6)

    def test_readout_record(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            umap_cfg = {"readout": "isp", "isp_emb_layer": 0, "isp_max_ncells": 2000, "max_cells_per_state": 2000}
            path = riu.write_umap_readout(
                Path(tmp), riu.resolve_umap_readout(umap_cfg), umap_cfg, n_start=10, n_end=12
            )
            rec = json.loads(path.read_text())
        self.assertTrue(rec["same_layer_and_dtype_as_isp"])
        self.assertEqual(rec["cells"], {"start": 10, "end": 12, "max_cells_per_state": 2000, "sample_key": None})
        self.assertIn("python", rec["environment"])
        rec_default = riu.resolve_umap_readout({})
        self.assertEqual(rec_default["dtype"], "float32")

    def test_downstream_overlays_use_run_pca_and_detect_gene(self):
        joint_mod = MagicMock()
        with patch.dict(
            sys.modules,
            {"plot_isp_umap_joint_overlays": joint_mod, "plot_isp_umap_celltype_trajectories": MagicMock()},
        ):
            riu.run_downstream_plots(
                Path("."), None, {"postprocess": {"enabled": True}, "umap": {"pca_components": 50}}
            )
        kwargs = joint_mod.run_joint_overlays.call_args.kwargs
        self.assertEqual(kwargs["pca_components"], 50)
        self.assertIsNone(kwargs["gene"])


class TestJointOverlayProjection(unittest.TestCase):
    def test_pca_projection_matches_main_plot_method(self):
        try:
            import umap  # noqa: F401
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"umap-learn missing: {exc}")
        from plot_isp_umap_joint_overlays import fit_umap

        embs = np.random.default_rng(0).normal(size=(40, 16))
        xy, backend = fit_umap(embs, n_neighbors=5, min_dist=0.1, seed=1, pca_components=8)
        self.assertEqual(xy.shape, (40, 2))
        self.assertEqual(backend, "PCA(8)→umap-learn")


class TestRefreshOverlayGene(unittest.TestCase):
    def test_single_gene_detected_and_several_rejected(self):
        import tempfile

        from annotate_isp_umap_celltypes import _single_isp_gene

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "AD_ISP_Apoe_embs.npy").write_bytes(b"")
            self.assertEqual(_single_isp_gene(run), "Apoe")
            (run / "AD_ISP_Trem2_embs.npy").write_bytes(b"")
            with self.assertRaises(SystemExit):
                _single_isp_gene(run)


if __name__ == "__main__":
    unittest.main()
