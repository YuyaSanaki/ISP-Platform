"""ISP stats must use the model backend's token/symbol dictionaries."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "core", ROOT / "contracts"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

try:
    from geneformer.backends.registry import get_backend
    from run_isp import genes_perturbed_for_stats, isp_stats_dictionary_kwargs

    DEPS_MISSING = None
except Exception as exc:  # pragma: no cover
    DEPS_MISSING = str(exc)


@unittest.skipIf(DEPS_MISSING, f"deps unavailable: {DEPS_MISSING}")
class TestIspStatsDictionaryKwargs(unittest.TestCase):
    def test_targeted_list_stats_modes(self):
        genes = ["ENSG1", "ENSG2"]
        self.assertEqual(genes_perturbed_for_stats(genes, False), genes)
        self.assertEqual(genes_perturbed_for_stats(genes, True), "all")
        self.assertEqual(genes_perturbed_for_stats([], True), "all")

    def test_each_gene_umap_targets_sig_detected_largest_shift(self):
        import tempfile

        import pandas as pd

        from run_isp import each_gene_umap_targets

        df = pd.DataFrame(
            {
                "Ensembl_ID": ["E1", "E2", "E3", "E4", "E5"],
                "Shift_to_goal_end": [0.01, -0.2, 0.3, 0.5, 0.05],
                "Sig": [1, 1, 1, 0, 1],
                "N_Detections": [50, 40, 5, 60, 30],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stats.parquet"
            df.to_parquet(path)
            genes = ["E1", "E2", "E3", "E4"]
            self.assertEqual(each_gene_umap_targets(path, genes, max_genes=10), ["E2", "E1"])
            self.assertEqual(each_gene_umap_targets(path, genes, max_genes=1), ["E2"])

    def test_umap_config_keeps_isp_filter_data(self):
        from run_isp_umap import build_isp_umap_config

        cfg = {
            "paths": {"dataset": "/d", "geneformer_model": "/m"},
            "perturbation": {"type": "delete", "state_key": "state", "start_state": "A", "end_state": "B"},
            "isp": {"filter_data": {"leiden013": ["5"]}},
        }
        self.assertEqual(build_isp_umap_config(cfg, ["E1"])["filter_data"], {"leiden013": ["5"]})

    def test_mouse_backend_uses_mouse_dicts(self):
        backend = get_backend(
            {"model_organism": "mouse", "model": "mouse_geneformer", "mouse_variant": "base"}
        )
        kwargs = isp_stats_dictionary_kwargs(backend)
        self.assertEqual(kwargs["token_dictionary_file"], backend.token_dictionary)
        self.assertEqual(
            kwargs["gene_name_id_dictionary_file"], backend.gene_symbol_to_ensembl
        )
        self.assertIn("mouse", str(kwargs["token_dictionary_file"]))
        self.assertTrue(Path(kwargs["token_dictionary_file"]).is_file())
        self.assertTrue(Path(kwargs["gene_name_id_dictionary_file"]).is_file())

    def test_human_backend_uses_human_dicts(self):
        backend = get_backend(
            {
                "model_organism": "human",
                "model": "human_geneformer",
                "human_variant": "v2_104m",
            }
        )
        kwargs = isp_stats_dictionary_kwargs(backend)
        self.assertEqual(kwargs["token_dictionary_file"], backend.token_dictionary)
        self.assertEqual(
            kwargs["gene_name_id_dictionary_file"], backend.gene_symbol_to_ensembl
        )
        tok = str(kwargs["token_dictionary_file"]).replace("\\", "/")
        sym = str(kwargs["gene_name_id_dictionary_file"]).replace("\\", "/")
        self.assertIn("/human/", tok)
        self.assertIn("gc104M", tok)
        self.assertIn("/human/", sym)
        self.assertTrue(Path(kwargs["token_dictionary_file"]).is_file())
        self.assertTrue(Path(kwargs["gene_name_id_dictionary_file"]).is_file())
        self.assertNotIn("/mouse/", sym)


if __name__ == "__main__":
    unittest.main()
