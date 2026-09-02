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
    from run_isp import isp_stats_dictionary_kwargs

    DEPS_MISSING = None
except Exception as exc:  # pragma: no cover
    DEPS_MISSING = str(exc)


@unittest.skipIf(DEPS_MISSING, f"deps unavailable: {DEPS_MISSING}")
class TestIspStatsDictionaryKwargs(unittest.TestCase):
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
