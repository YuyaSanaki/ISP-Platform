"""Tests for loom_write.write_loom_safe fallback (h5py rejects empty attr names)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scanpy as sc
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))

from loom_write import write_loom_safe  # noqa: E402


class TestWriteLoomSafe(unittest.TestCase):
    def test_fallback_writes_readable_loom(self):
        adata = sc.AnnData(
            X=sparse.csr_matrix([[1, 2], [0, 3]], dtype=np.float32),
            obs={"n_counts": [3.0, 3.0], "sample_id": ["s1", "s2"]},
            var={"ensembl_id": ["ENSG1", "ENSG2"], "gene_symbols": [None, "G2"]},
        )
        adata.obs_names = ["cell_a", "cell_b"]
        adata.var_names = ["ENSG1", "ENSG2"]

        with tempfile.TemporaryDirectory() as tmp:
            loom_path = Path(tmp) / "tiny.loom"
            real_write = sc.AnnData.write_loom

            def _fail_write_loom(self, path, *args, **kwargs):
                raise AttributeError("'NoneType' object has no attribute 'startswith'")

            sc.AnnData.write_loom = _fail_write_loom
            try:
                write_loom_safe(adata, loom_path)
            finally:
                sc.AnnData.write_loom = real_write

            self.assertTrue(loom_path.is_file())
            import loompy

            with loompy.connect(str(loom_path)) as data:
                self.assertEqual(list(data.ra.keys()), ["ensembl_id"])
                self.assertIn("cell_id", data.ca.keys())
                self.assertNotIn("", data.ca.keys())


if __name__ == "__main__":
    unittest.main()
