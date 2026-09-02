"""Normalize AnnData tokenization by coding-gene sums (match loom path).

Cross-species conversion drops unmapped genes / aggregates many-to-one orthologs.
Using a stale obs['n_counts'] (pre-conversion library size) changes relative
expression and therefore rank encodings. tokenize_loom already divides by the
sum of coding/miRNA genes in the cell; _tokenize_adata_matrix must do the same.
"""
from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "core", ROOT / "contracts"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


def _purge_import_stubs() -> None:
    for key in list(sys.modules):
        if key == "geneformer" or key.startswith("geneformer."):
            del sys.modules[key]


class TestAdataCodingSumNorm(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _purge_import_stubs()

    def setUp(self) -> None:
        _purge_import_stubs()

    def _tokenizer(self, gene_ids, tmp: Path):
        from geneformer.tokenizer import TranscriptomeTokenizer

        median = {g: 1.0 for g in gene_ids}
        tokens = {g: i + 5 for i, g in enumerate(gene_ids)}
        med_path = tmp / "median.pkl"
        tok_path = tmp / "token.pkl"
        with med_path.open("wb") as fh:
            pickle.dump(median, fh)
        with tok_path.open("wb") as fh:
            pickle.dump(tokens, fh)
        return TranscriptomeTokenizer(
            gene_median_file=med_path,
            token_dictionary_file=tok_path,
            chunk_size=8,
            max_cells=1000,
        )

    def test_stale_n_counts_does_not_change_ranks(self):
        import anndata as ad

        from geneformer.tokenizer import TranscriptomeTokenizer

        gene_ids = [f"ENSG{i:011d}" for i in range(4)]
        # Two genes dominate; two are low. Inflated n_counts mimics pre-conversion
        # library size (dropped non-coding / unmapped genes still counted).
        X = np.array(
            [
                [40.0, 10.0, 1.0, 1.0],
                [5.0, 30.0, 2.0, 1.0],
            ],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tk = self._tokenizer(gene_ids, tmp_path)

            adata_stale = ad.AnnData(X=X.copy())
            adata_stale.var["ensembl_id"] = gene_ids
            adata_stale.obs["n_counts"] = [1000.0, 1000.0]  # wrong / pre-conversion

            adata_fresh = ad.AnnData(X=X.copy())
            adata_fresh.var["ensembl_id"] = gene_ids
            adata_fresh.obs["n_counts"] = X.sum(axis=1)

            cells_stale, _ = tk._tokenize_adata_matrix(adata_stale)
            cells_fresh, _ = tk._tokenize_adata_matrix(adata_fresh)

            stale = [np.asarray(c).tolist() for c in cells_stale]
            fresh = [np.asarray(c).tolist() for c in cells_fresh]
            self.assertEqual(stale, fresh)

            # Sanity: ranks follow relative expression within coding genes.
            # Cell 0: gene0 (40) > gene1 (10) > ...
            self.assertEqual(stale[0][0], tk.gene_token_dict[gene_ids[0]])
            self.assertEqual(stale[1][0], tk.gene_token_dict[gene_ids[1]])

    def test_maybe_convert_refreshes_n_counts(self):
        import anndata as ad

        gene_ids = [f"ENSG{i:011d}" for i in range(3)]
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            tk = self._tokenizer(gene_ids, Path(tmp))
            tk._species_convert = True

            def fake_remap(adata, species):
                # Drop last gene (unmapped ortholog).
                slim = adata[:, :2].copy()
                return slim, object()

            import geneformer.species_context as sc

            adata = ad.AnnData(X=X.copy())
            adata.var["ensembl_id"] = gene_ids
            adata.obs["n_counts"] = [100.0, 100.0]

            with mock.patch.object(sc, "remap_adata_ensembl_ids", side_effect=fake_remap):
                out = tk._maybe_convert_adata(adata)

            np.testing.assert_allclose(out.obs["n_counts"].to_numpy(), [3.0, 9.0])
            self.assertEqual(out.n_vars, 2)


if __name__ == "__main__":
    unittest.main()
