"""Ensembl version suffixes must be stripped before loom/tokenize dict lookup."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "core", ROOT / "contracts"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from geneformer.gene_converter import normalize_gene_id


def _load_convert_to_loom():
    path = ROOT / "core" / "convert_to_loom.py"
    spec = importlib.util.spec_from_file_location("convert_to_loom_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestNormalizeGeneId(unittest.TestCase):
    def test_strips_ensembl_version(self):
        self.assertEqual(
            normalize_gene_id("ENSMUSG00000102693.2"), "ENSMUSG00000102693"
        )
        self.assertEqual(normalize_gene_id("ENSG00000115457.15"), "ENSG00000115457")

    def test_leaves_unversioned_and_non_ensembl(self):
        self.assertEqual(normalize_gene_id("ENSMUSG00000102693"), "ENSMUSG00000102693")
        self.assertEqual(normalize_gene_id("Igfbp2"), "Igfbp2")
        self.assertEqual(normalize_gene_id("FBgn0036990"), "FBgn0036990")


class TestConvertToLoomStrip(unittest.TestCase):
    def test_ensembl_ids_without_version(self):
        mod = _load_convert_to_loom()
        names = pd.Index(
            [
                "ENSMUSG00000102693.2",
                "ENSMUSG00000039323",
                "ENSG00000115457.15",
            ]
        )
        self.assertEqual(
            mod.ensembl_ids_without_version(names),
            [
                "ENSMUSG00000102693",
                "ENSMUSG00000039323",
                "ENSG00000115457",
            ],
        )


if __name__ == "__main__":
    unittest.main()
