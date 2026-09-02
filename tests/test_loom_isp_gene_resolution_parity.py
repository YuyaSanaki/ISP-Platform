"""Loom remap and ISP must share the same gene-resolution path."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "core", ROOT / "contracts", ROOT / "webui"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from geneformer import gene_converter


MOUSE_HUMAN = {
    "model_organism": "mouse",
    "model": "human_geneformer",
    "human_variant": "v2_104m",
    "ortholog_policy": "legacy_sum",
}
MOUSE_ENS = "ENSMUSG00000039323"
HUMAN_ENS = "ENSG00000115457"
# Symbol present only via symbol→Ensembl→ortholog (not as an ortholog-table key).
SYMBOL_ONLY = "GapdhProxy"
PROXY_ENS = "ENSMUSG00000057654"


class TestLoomIspGeneResolutionParity(unittest.TestCase):
    def setUp(self):
        gene_converter._TABLE_CACHE.clear()

    def test_convert_gene_ids_uses_symbol_table_like_isp(self):
        """Symbol absent from ortholog keys still resolves when symbol table is provided."""
        table = {PROXY_ENS: HUMAN_ENS, MOUSE_ENS: HUMAN_ENS}
        symbol_table = {SYMBOL_ONLY: PROXY_ENS, "Igfbp2": MOUSE_ENS}

        without = gene_converter.convert_gene_ids(
            [SYMBOL_ONLY],
            model_organism="mouse",
            model_id="human_geneformer",
            ortholog_table=table,
            species=MOUSE_HUMAN,
        )
        self.assertIn(SYMBOL_ONLY, without.unmapped)

        with_sym = gene_converter.convert_gene_ids(
            [SYMBOL_ONLY, MOUSE_ENS],
            model_organism="mouse",
            model_id="human_geneformer",
            ortholog_table=table,
            species=MOUSE_HUMAN,
            input_symbol_table=symbol_table,
        )
        self.assertEqual(with_sym.mapped[SYMBOL_ONLY], HUMAN_ENS)
        self.assertEqual(with_sym.mapped[MOUSE_ENS], HUMAN_ENS)

        isp_sym = gene_converter.resolve_gene_for_model(
            SYMBOL_ONLY,
            MOUSE_HUMAN,
            input_symbol_table=symbol_table,
            ortholog_table=table,
        )
        isp_ens = gene_converter.resolve_gene_for_model(
            MOUSE_ENS,
            MOUSE_HUMAN,
            input_symbol_table=symbol_table,
            ortholog_table=table,
        )
        self.assertEqual(isp_sym, with_sym.mapped[SYMBOL_ONLY])
        self.assertEqual(isp_ens, with_sym.mapped[MOUSE_ENS])

    @unittest.skipUnless(
        importlib.util.find_spec("anndata") and importlib.util.find_spec("scipy"),
        "anndata/scipy not installed",
    )
    def test_remap_adata_matches_isp_for_symbol_and_ensembl(self):
        import anndata as ad
        import numpy as np

        table = {PROXY_ENS: HUMAN_ENS, MOUSE_ENS: HUMAN_ENS}
        symbol_table = {SYMBOL_ONLY: PROXY_ENS}

        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        var = ad.AnnData(np.zeros((1, 2))).var
        var["ensembl_id"] = [SYMBOL_ONLY, MOUSE_ENS]
        adata = ad.AnnData(X=X, var=var)
        adata.obs["n_counts"] = [3.0, 7.0]

        with mock.patch(
            "geneformer.species_context.load_input_symbol_table",
            return_value=symbol_table,
        ), mock.patch.object(
            gene_converter,
            "load_ortholog_table",
            return_value=table,
        ):
            out, conv = gene_converter.remap_adata_ensembl_ids(adata, MOUSE_HUMAN)

        self.assertEqual(conv.mapped[SYMBOL_ONLY], HUMAN_ENS)
        self.assertEqual(conv.mapped[MOUSE_ENS], HUMAN_ENS)
        self.assertEqual(list(out.var["ensembl_id"]), [HUMAN_ENS])

        isp_sym = gene_converter.resolve_gene_for_model(
            SYMBOL_ONLY,
            MOUSE_HUMAN,
            input_symbol_table=symbol_table,
            ortholog_table=table,
        )
        isp_ens = gene_converter.resolve_gene_for_model(
            MOUSE_ENS,
            MOUSE_HUMAN,
            input_symbol_table=symbol_table,
            ortholog_table=table,
        )
        self.assertEqual(isp_sym, HUMAN_ENS)
        self.assertEqual(isp_ens, HUMAN_ENS)
        self.assertEqual(isp_sym, conv.mapped[SYMBOL_ONLY])
        self.assertEqual(isp_ens, conv.mapped[MOUSE_ENS])


if __name__ == "__main__":
    unittest.main()
