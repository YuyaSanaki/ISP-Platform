"""Chunked loom tokenization for cross-species ortholog conversion (R3)."""
from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "core", ROOT / "contracts", ROOT / "webui"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)



def _purge_import_stubs() -> None:
    """Clear ModuleType stubs so real geneformer/torch/datasets imports work."""
    for key in list(sys.modules):
        if key == "geneformer" or key.startswith("geneformer."):
            del sys.modules[key]
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None and getattr(torch_mod, "__file__", None) is None:
        for key in list(sys.modules):
            if key == "torch" or key.startswith("torch."):
                del sys.modules[key]
        # Half-imported against the stub breaks later discover runs.
        for key in list(sys.modules):
            if key == "datasets" or key.startswith("datasets."):
                del sys.modules[key]


class TestLoomChunkedConvert(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _purge_import_stubs()

    def setUp(self) -> None:
        # Re-purge in case a prior test module re-stubbed (discover order).
        _purge_import_stubs()

    def _write_tiny_loom(self, path: Path, *, n_genes: int = 8, n_cells: int = 10) -> list[str]:
        import loompy

        gene_ids = [f"ENSG{i:011d}" for i in range(n_genes)]
        mat = np.random.default_rng(0).integers(0, 20, size=(n_genes, n_cells)).astype(np.float32)
        # Avoid all-zero cells (n_counts / norm)
        mat[:, :] += 1
        row_attrs = {"ensembl_id": np.asarray(gene_ids, dtype=object)}
        col_attrs = {
            "n_counts": mat.sum(axis=0),
            "cell_id": np.asarray([f"c{i}" for i in range(n_cells)], dtype=object),
        }
        loompy.create(str(path), mat, row_attrs=row_attrs, col_attrs=col_attrs)
        return gene_ids

    def _write_vocab(self, gene_ids: list[str], tmp: Path) -> tuple[Path, Path]:
        median = {g: 1.0 for g in gene_ids}
        tokens = {g: i + 5 for i, g in enumerate(gene_ids)}
        med_path = tmp / "median.pkl"
        tok_path = tmp / "token.pkl"
        with med_path.open("wb") as fh:
            pickle.dump(median, fh)
        with tok_path.open("wb") as fh:
            pickle.dump(tokens, fh)
        return med_path, tok_path

    def test_loom_view_to_anndata_is_batch_sized(self):
        import loompy
        from geneformer.tokenizer import TranscriptomeTokenizer

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            loom_path = tmp_path / "tiny.loom"
            gene_ids = self._write_tiny_loom(loom_path, n_genes=6, n_cells=9)
            med, tok = self._write_vocab(gene_ids, tmp_path)
            tk = TranscriptomeTokenizer(
                gene_median_file=med,
                token_dictionary_file=tok,
                chunk_size=4,
                max_cells=1000,
            )
            with loompy.connect(str(loom_path)) as data:
                batches = list(
                    data.scan(items=np.arange(9), axis=1, batch_size=4)
                )
                self.assertGreaterEqual(len(batches), 3)
                _ix, _sel, view = batches[0]
                adata = tk._loom_view_to_anndata(view, "ensembl_id")
                self.assertEqual(adata.n_vars, 6)
                self.assertLessEqual(adata.n_obs, 4)
                self.assertIn("ensembl_id", adata.var.columns)
                self.assertIn("n_counts", adata.obs.columns)

    def test_cross_species_path_scans_with_batch_size(self):
        import loompy
        from geneformer.tokenizer import TranscriptomeTokenizer

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            loom_path = tmp_path / "tiny.loom"
            gene_ids = self._write_tiny_loom(loom_path, n_genes=5, n_cells=11)
            med, tok = self._write_vocab(gene_ids, tmp_path)
            tk = TranscriptomeTokenizer(
                gene_median_file=med,
                token_dictionary_file=tok,
                chunk_size=3,
                max_cells=1000,
            )
            # Force convert path without loading real ortholog tables.
            tk._species_convert = True
            tk._maybe_convert_adata = lambda adata: adata  # type: ignore[method-assign]

            seen_batch_sizes: list[int] = []
            real_scan = loompy.LoomConnection.scan

            def tracked_scan(self, *args, **kwargs):
                seen_batch_sizes.append(int(kwargs.get("batch_size") or 0))
                yield from real_scan(self, *args, **kwargs)

            with mock.patch.object(loompy.LoomConnection, "scan", tracked_scan):
                cells, _meta, n_cells = tk.tokenize_loom(loom_path, chunk_size=3)

            self.assertEqual(n_cells, 11)
            self.assertEqual(len(cells), 11)
            self.assertTrue(seen_batch_sizes, "expected loompy.scan to be used")
            self.assertTrue(all(b == 3 for b in seen_batch_sizes))
            # Each cell token list non-empty (all genes expressed after +1).
            self.assertTrue(all(len(c) > 0 for c in cells))

    def test_native_path_also_passes_batch_size(self):
        import loompy
        from geneformer.tokenizer import TranscriptomeTokenizer

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            loom_path = tmp_path / "tiny.loom"
            gene_ids = self._write_tiny_loom(loom_path, n_genes=4, n_cells=7)
            med, tok = self._write_vocab(gene_ids, tmp_path)
            tk = TranscriptomeTokenizer(
                gene_median_file=med,
                token_dictionary_file=tok,
                chunk_size=2,
                max_cells=1000,
            )
            self.assertFalse(tk._species_convert)

            seen: list[int] = []
            real_scan = loompy.LoomConnection.scan

            def tracked_scan(self, *args, **kwargs):
                seen.append(int(kwargs.get("batch_size") or 0))
                yield from real_scan(self, *args, **kwargs)

            with mock.patch.object(loompy.LoomConnection, "scan", tracked_scan):
                cells, _meta, n_cells = tk.tokenize_loom(loom_path)

            self.assertEqual(n_cells, 7)
            self.assertEqual(len(cells), 7)
            self.assertTrue(seen)
            self.assertTrue(all(b == 2 for b in seen))


if __name__ == "__main__":
    unittest.main()
