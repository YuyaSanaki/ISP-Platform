"""10x→loom failure aggregation, loom custom_attr back-fill, cross-species max_cells."""
from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "contracts"))


def _purge_import_stubs() -> None:
    for key in list(sys.modules):
        if key == "geneformer" or key.startswith("geneformer."):
            del sys.modules[key]
        if key == "execute_tokenizer_pipeline":
            del sys.modules[key]


class TestBackfillLoomCustomAttrs(unittest.TestCase):
    def setUp(self) -> None:
        _purge_import_stubs()

    def test_fills_missing_column_attrs(self):
        import loompy
        from execute_tokenizer_pipeline import backfill_loom_custom_attrs

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.loom"
            mat = np.ones((3, 4), dtype=np.float32)
            loompy.create(
                str(path),
                mat,
                row_attrs={"ensembl_id": np.asarray([f"E{i}" for i in range(3)], dtype=object)},
                col_attrs={"n_counts": mat.sum(axis=0)},
            )
            filled = backfill_loom_custom_attrs(
                tmp, {"disease": "disease", "sample_id": "sample_id"}
            )
            self.assertEqual(sorted(filled), ["s.loom:disease", "s.loom:sample_id"])
            with loompy.connect(str(path)) as ds:
                self.assertIn("disease", ds.ca)
                self.assertEqual(list(ds.ca["disease"]), ["", "", "", ""])


class TestProcessSingleCellFailures(unittest.TestCase):
    def setUp(self) -> None:
        _purge_import_stubs()

    def test_conversion_error_exits_nonzero(self):
        from execute_tokenizer_pipeline import process_single_cell_to_loom

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "1w-WT-1st"
            sample.mkdir()
            # Minimal names so discover_sample_dirs finds the folder; read is mocked.
            for name in ("matrix.mtx.gz", "barcodes.tsv.gz", "features.tsv.gz"):
                (sample / name).write_bytes(b"stub")
            with mock.patch(
                "execute_tokenizer_pipeline.sc.read_10x_mtx",
                side_effect=RuntimeError("corrupt mtx"),
            ):
                with self.assertRaises(SystemExit) as ctx:
                    process_single_cell_to_loom(
                        str(root),
                        str(root / "loom_files"),
                        {"extract_metadata_from_path": True},
                        {"custom_attr_name_dict": {"disease": "disease"}},
                    )
            self.assertNotEqual(ctx.exception.code, 0)
            self.assertIn("failed 10x", str(ctx.exception))


class TestCrossSpeciesMaxCellsTruncate(unittest.TestCase):
    def setUp(self) -> None:
        _purge_import_stubs()

    def _write_tiny_loom(self, path: Path, *, n_genes: int, n_cells: int) -> list[str]:
        import loompy

        gene_ids = [f"ENSG{i:011d}" for i in range(n_genes)]
        mat = np.random.default_rng(0).integers(0, 20, size=(n_genes, n_cells)).astype(np.float32)
        mat[:, :] += 1
        loompy.create(
            str(path),
            mat,
            row_attrs={"ensembl_id": np.asarray(gene_ids, dtype=object)},
            col_attrs={
                "n_counts": mat.sum(axis=0),
                "cell_id": np.asarray([f"c{i}" for i in range(n_cells)], dtype=object),
            },
        )
        return gene_ids

    def _vocab(self, gene_ids: list[str], tmp: Path) -> tuple[Path, Path]:
        med_path = tmp / "median.pkl"
        tok_path = tmp / "token.pkl"
        with med_path.open("wb") as fh:
            pickle.dump({g: 1.0 for g in gene_ids}, fh)
        with tok_path.open("wb") as fh:
            pickle.dump({g: i + 5 for i, g in enumerate(gene_ids)}, fh)
        return med_path, tok_path

    def test_default_truncates_and_records(self):
        from geneformer.tokenizer import TranscriptomeTokenizer

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            loom_path = tmp_path / "big.loom"
            gene_ids = self._write_tiny_loom(loom_path, n_genes=4, n_cells=12)
            med, tok = self._vocab(gene_ids, tmp_path)
            tk = TranscriptomeTokenizer(
                gene_median_file=med,
                token_dictionary_file=tok,
                max_cells=5,
                truncate_cross_species_to_max_cells=True,
            )
            tk._species_convert = True
            tk._maybe_convert_adata = lambda adata: adata  # type: ignore[method-assign]
            cells, _meta, n_cells = tk.tokenize_loom(loom_path)
            self.assertEqual(n_cells, 5)
            self.assertEqual(len(cells), 5)
            self.assertEqual(len(tk.cross_species_file_truncations), 1)
            self.assertEqual(tk.cross_species_file_truncations[0]["kept"], 5)

    def test_opt_out_keeps_all_cells(self):
        from geneformer.tokenizer import TranscriptomeTokenizer

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            loom_path = tmp_path / "big.loom"
            gene_ids = self._write_tiny_loom(loom_path, n_genes=4, n_cells=12)
            med, tok = self._vocab(gene_ids, tmp_path)
            tk = TranscriptomeTokenizer(
                gene_median_file=med,
                token_dictionary_file=tok,
                max_cells=5,
                truncate_cross_species_to_max_cells=False,
            )
            tk._species_convert = True
            tk._maybe_convert_adata = lambda adata: adata  # type: ignore[method-assign]
            cells, _meta, n_cells = tk.tokenize_loom(loom_path)
            self.assertEqual(n_cells, 12)
            self.assertEqual(len(cells), 12)
            self.assertEqual(tk.cross_species_file_truncations, [])


if __name__ == "__main__":
    unittest.main()
