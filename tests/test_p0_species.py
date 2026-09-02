"""P0 species / ortholog conversion tests (stdlib + minimal deps)."""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "core", ROOT / "contracts", ROOT / "webui"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


def _load_package_modules():
    """Load geneformer submodules without importing heavy tokenizer dependencies."""
    geneformer_pkg = types.ModuleType("geneformer")
    geneformer_pkg.__path__ = [str(ROOT / "core" / "geneformer")]
    backends_pkg = types.ModuleType("geneformer.backends")
    sys.modules["geneformer"] = geneformer_pkg
    sys.modules["geneformer.backends"] = backends_pkg

    def _load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    # Lightweight torch stub for auto_batch_size import during pipeline_lib load.
    # Remove after load so later tests can import the real torch package.
    installed_torch_stub = False
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            OutOfMemoryError=RuntimeError,
        )
        sys.modules["torch"] = torch_stub
        installed_torch_stub = True

    try:
        registry = _load("geneformer.backends.registry", ROOT / "core" / "geneformer" / "backends" / "registry.py")
        backends_pkg.registry = registry
        for name in (
            "BackendSpec",
            "HumanVariant",
            "ModelId",
            "MouseVariant",
            "Organism",
            "get_backend",
            "native_organism",
            "needs_gene_conversion",
            "parse_species_config",
            "resolve_pretrained_path",
        ):
            setattr(backends_pkg, name, getattr(registry, name))
        gene_converter = _load("geneformer.gene_converter", ROOT / "core" / "geneformer" / "gene_converter.py")
        species_context = _load("geneformer.species_context", ROOT / "core" / "geneformer" / "species_context.py")
        auto_batch_size = _load("geneformer.auto_batch_size", ROOT / "core" / "geneformer" / "auto_batch_size.py")
        geneformer_pkg.species_context = species_context
        geneformer_pkg.auto_batch_size = auto_batch_size
        pipeline_lib = _load("pipeline_lib", ROOT / "core" / "pipeline_lib.py")
        return registry, gene_converter, pipeline_lib, species_context
    finally:
        if installed_torch_stub:
            sys.modules.pop("torch", None)


registry, gene_converter, pipeline_lib, species_context = _load_package_modules()


class TestSpeciesRegistry(unittest.TestCase):
    def test_mouse_defaults(self):
        species = registry.parse_species_config({})
        self.assertEqual(species["model_organism"], "mouse")
        self.assertEqual(species["model"], "mouse_geneformer")
        self.assertEqual(species["mouse_variant"], "base")
        self.assertEqual(species["ortholog_policy"], "one2one")
        backend = registry.get_backend(species)
        self.assertEqual(backend.max_input_size, 2048)
        self.assertTrue(str(backend.pretrained_dir).endswith("mouse-Geneformer"))

    def test_mouse_12l_e20(self):
        species = {
            "model_organism": "mouse",
            "model": "mouse_geneformer",
            "mouse_variant": "12l_e20",
        }
        backend = registry.get_backend(species)
        self.assertEqual(backend.max_input_size, 2048)
        self.assertIn("12L-E20", str(backend.pretrained_dir))
        self.assertEqual(backend.mouse_variant, registry.MouseVariant.L12_E20)

    def test_human_data_mouse_12l_resolves_same_backend(self):
        species = {
            "model_organism": "human",
            "model": "mouse_geneformer",
            "mouse_variant": "12l_e20",
        }
        backend = registry.get_backend(species)
        self.assertTrue(registry.needs_gene_conversion("human", "mouse_geneformer"))
        self.assertIn("12L-E20", str(backend.pretrained_dir))

    def test_invalid_mouse_variant(self):
        with self.assertRaises(ValueError):
            registry.parse_species_config(
                {"model": "mouse_geneformer", "mouse_variant": "96l"}
            )

    def test_human_v2_104m(self):
        species = {
            "model_organism": "human",
            "model": "human_geneformer",
            "human_variant": "v2_104m",
        }
        backend = registry.get_backend(species)
        self.assertEqual(backend.max_input_size, 4096)
        self.assertIn("V2-104M", str(backend.pretrained_dir))
        self.assertEqual(species.get("mouse_variant", ""), "")

    def test_invalid_organism(self):
        with self.assertRaises(ValueError):
            registry.parse_species_config({"model_organism": "zebrafish"})

    def test_needs_conversion(self):
        self.assertFalse(registry.needs_gene_conversion("mouse", "mouse_geneformer"))
        self.assertTrue(registry.needs_gene_conversion("mouse", "human_geneformer"))


class TestOrthologConversion(unittest.TestCase):
    def setUp(self):
        gene_converter._TABLE_CACHE.clear()

    def test_mouse_to_human_ensembl(self):
        result = gene_converter.convert_gene_ids(
            ["ENSMUSG00000039323"],
            model_organism="mouse",
            model_id="human_geneformer",
        )
        self.assertEqual(result.mapped["ENSMUSG00000039323"], "ENSG00000115457")
        self.assertEqual(result.dropped_count, 0)

    def test_mouse_symbol_igfbp2(self):
        result = gene_converter.convert_gene_ids(
            ["Igfbp2"],
            model_organism="mouse",
            model_id="human_geneformer",
        )
        self.assertEqual(result.mapped["Igfbp2"], "ENSG00000115457")

    def test_ensembl_version_suffix_stripped(self):
        result = gene_converter.convert_gene_ids(
            ["ENSMUSG00000039323.2"],
            model_organism="mouse",
            model_id="human_geneformer",
        )
        self.assertEqual(result.mapped["ENSMUSG00000039323.2"], "ENSG00000115457")

    def test_no_conversion_passthrough(self):
        pair = gene_converter.conversion_pair("mouse", "mouse_geneformer")
        self.assertIsNone(pair)
        result = gene_converter.convert_gene_ids(
            ["ENSMUSG00000039323"],
            model_organism="mouse",
            model_id="mouse_geneformer",
        )
        self.assertEqual(result.mapped["ENSMUSG00000039323"], "ENSMUSG00000039323")

    def test_resolve_gene_for_model_cross_species(self):
        resolved = gene_converter.resolve_gene_for_model(
            "Igfbp2",
            {"model_organism": "mouse", "model": "human_geneformer", "human_variant": "v2_104m"},
        )
        self.assertEqual(resolved, "ENSG00000115457")

    def test_resolve_gene_with_symbol_table_prefers_curated_symbol(self):
        symbol_table = {"Igfbp2": "ENSMUSG00000039323"}
        resolved = gene_converter.resolve_gene_for_model(
            "Igfbp2",
            {"model_organism": "mouse", "model": "human_geneformer", "human_variant": "v2_104m"},
            input_symbol_table=symbol_table,
        )
        self.assertEqual(resolved, "ENSG00000115457")

    def test_resolve_ensembl_matches_symbol_curated_not_biomart(self):
        """Ensembl input must use same curated target as symbol (not BioMart-only ID)."""
        resolved = gene_converter.resolve_gene_for_model(
            "ENSMUSG00000039323",
            {"model_organism": "mouse", "model": "human_geneformer", "human_variant": "v2_104m"},
            input_symbol_table={"Igfbp2": "ENSMUSG00000039323"},
        )
        self.assertEqual(resolved, "ENSG00000115457")
        self.assertNotEqual(resolved, "ENSG00000017427")

    def test_resolve_already_converted_human_ensembl(self):
        resolved = gene_converter.resolve_gene_for_model(
            "ENSG00000115457",
            {"model_organism": "mouse", "model": "human_geneformer", "human_variant": "v2_104m"},
            input_symbol_table={"Igfbp2": "ENSMUSG00000039323"},
        )
        self.assertEqual(resolved, "ENSG00000115457")

    def test_convert_perturbation_genes(self):
        genes, result = gene_converter.convert_perturbation_genes(
            ["Igfbp2", "UnknownGene"],
            {"model_organism": "mouse", "model": "human_geneformer", "human_variant": "v2_104m"},
        )
        self.assertEqual(genes, ["ENSG00000115457"])
        self.assertIn("UnknownGene", result.unmapped)


class TestPipelineLibSpecies(unittest.TestCase):
    def test_resolve_paths_includes_species(self):
        pipeline = {
            "data": {"input_dir": "/app/data/MyStudy/"},
            "species": {
                "model_organism": "mouse",
                "model": "human_geneformer",
                "human_variant": "v2_104m",
            },
            "paths": {"output_root": "/app/output"},
        }
        resolved = pipeline_lib.resolve_pipeline_paths(pipeline, Path("/app/output/run1"))
        self.assertEqual(resolved["species"]["model"], "human_geneformer")
        self.assertEqual(resolved["max_input_size"], 4096)
        self.assertIn("human-Geneformer", resolved["pretrained_model"])

    def test_resolve_paths_mouse_12l(self):
        pipeline = {
            "data": {"input_dir": "/app/data/MyStudy/"},
            "species": {
                "model_organism": "human",
                "model": "mouse_geneformer",
                "mouse_variant": "12l_e20",
            },
            "paths": {"output_root": "/app/output"},
        }
        resolved = pipeline_lib.resolve_pipeline_paths(pipeline, Path("/app/output/run1"))
        self.assertEqual(resolved["species"]["mouse_variant"], "12l_e20")
        self.assertEqual(resolved["max_input_size"], 2048)
        self.assertIn("12L-E20", resolved["pretrained_model"])

    def test_stage_configs_carry_species(self):
        pipeline = {
            "data": {"input_dir": "/app/data/MyStudy/"},
            "species": {"model_organism": "mouse", "model": "mouse_geneformer"},
            "paths": {"output_root": "/app/output"},
        }
        resolved = pipeline_lib.resolve_pipeline_paths(pipeline, Path("/app/output/run1"))
        tokenize = pipeline_lib.build_tokenize_config(pipeline, resolved, {"data": {}, "tokenizer": {}})
        self.assertEqual(tokenize["species"]["model"], "mouse_geneformer")

    def test_tokenize_report_conversion_default_fly(self):
        pipeline = {
            "data": {"input_dir": "/app/data/fly/"},
            "species": {
                "model_organism": "drosophila",
                "model": "human_geneformer",
                "human_variant": "v2_104m",
            },
            "paths": {"output_root": "/app/output"},
        }
        resolved = pipeline_lib.resolve_pipeline_paths(pipeline, Path("/app/output/run1"))
        tokenize = pipeline_lib.build_tokenize_config(pipeline, resolved, {"data": {}, "tokenizer": {}})
        self.assertTrue(tokenize["tokenizer"]["report_conversion"])

    def test_tokenize_report_conversion_default_mouse_cross_species_on(self):
        pipeline = {
            "data": {"input_dir": "/app/data/1w/"},
            "species": {
                "model_organism": "mouse",
                "model": "human_geneformer",
                "human_variant": "v2_104m",
            },
            "paths": {"output_root": "/app/output"},
        }
        resolved = pipeline_lib.resolve_pipeline_paths(pipeline, Path("/app/output/run1"))
        tokenize = pipeline_lib.build_tokenize_config(pipeline, resolved, {"data": {}, "tokenizer": {}})
        self.assertTrue(tokenize["tokenizer"]["report_conversion"])

    def test_tokenize_report_conversion_default_mouse_native_off(self):
        pipeline = {
            "data": {"input_dir": "/app/data/1w/"},
            "species": {
                "model_organism": "mouse",
                "model": "mouse_geneformer",
            },
            "paths": {"output_root": "/app/output"},
        }
        resolved = pipeline_lib.resolve_pipeline_paths(pipeline, Path("/app/output/run1"))
        tokenize = pipeline_lib.build_tokenize_config(pipeline, resolved, {"data": {}, "tokenizer": {}})
        self.assertFalse(tokenize["tokenizer"]["report_conversion"])


class TestRemapAdata(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("anndata") and importlib.util.find_spec("scipy"),
        "anndata/scipy not installed",
    )
    def test_remap_aggregates_duplicates(self):
        import anndata as ad
        import numpy as np
        from unittest import mock

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        var = ad.AnnData(np.zeros((2, 2))).var
        var["ensembl_id"] = ["ENSMUSG00000025903", "ENSMUSG00000069516"]
        adata = ad.AnnData(X=X, var=var)
        adata.obs["n_counts"] = [10.0, 10.0]

        # Synthetic N→1 orthologs: both mouse IDs map to human TP53.
        fake_table = {
            "ENSMUSG00000025903": "ENSG00000141510",
            "ENSMUSG00000069516": "ENSG00000141510",
        }
        with mock.patch.object(gene_converter, "load_ortholog_table", return_value=fake_table), mock.patch(
            "geneformer.species_context.load_input_symbol_table",
            return_value={},
        ):
            out, result = gene_converter.remap_adata_ensembl_ids(
                adata,
                {
                    "model_organism": "mouse",
                    "model": "human_geneformer",
                    "human_variant": "v2_104m",
                    "ortholog_policy": "legacy_sum",
                },
            )
        self.assertEqual(out.n_vars, 1)
        self.assertEqual(out.var["ensembl_id"].iloc[0], "ENSG00000141510")
        self.assertEqual(float(out.X[0, 0]), 3.0)
        self.assertEqual(result.kept_count, 2)

    @unittest.skipUnless(
        importlib.util.find_spec("anndata") and importlib.util.find_spec("scipy"),
        "anndata/scipy not installed",
    )
    def test_best_of_n_keeps_highest_median(self):
        import anndata as ad
        import numpy as np
        from unittest import mock

        X = np.array([[1.0, 10.0], [1.0, 10.0], [1.0, 10.0]])
        var = ad.AnnData(np.zeros((3, 2))).var
        var["ensembl_id"] = ["ENSMUSG00000025903", "ENSMUSG00000069516"]
        adata = ad.AnnData(X=X, var=var)
        fake_table = {
            "ENSMUSG00000025903": "ENSG00000141510",
            "ENSMUSG00000069516": "ENSG00000141510",
        }
        with mock.patch.object(gene_converter, "load_ortholog_table", return_value=fake_table):
            out, _ = gene_converter.remap_adata_ensembl_ids(
                adata,
                {
                    "model_organism": "mouse",
                    "model": "human_geneformer",
                    "human_variant": "v2_104m",
                    "ortholog_policy": "best_of_n",
                },
            )
        self.assertEqual(out.n_vars, 1)
        self.assertEqual(float(out.X[0, 0]), 10.0)

    @unittest.skipUnless(
        importlib.util.find_spec("anndata") and importlib.util.find_spec("scipy"),
        "anndata/scipy not installed",
    )
    def test_one2one_keeps_first_not_sum(self):
        import anndata as ad
        import numpy as np
        from unittest import mock

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        var = ad.AnnData(np.zeros((2, 2))).var
        var["ensembl_id"] = ["ENSMUSG00000025903", "ENSMUSG00000069516"]
        adata = ad.AnnData(X=X, var=var)
        fake_table = {
            "ENSMUSG00000025903": "ENSG00000141510",
            "ENSMUSG00000069516": "ENSG00000141510",
        }
        with mock.patch.object(gene_converter, "load_ortholog_table", return_value=fake_table):
            out, _ = gene_converter.remap_adata_ensembl_ids(
                adata,
                {
                    "model_organism": "mouse",
                    "model": "human_geneformer",
                    "human_variant": "v2_104m",
                    "ortholog_policy": "one2one",
                },
            )
        self.assertEqual(out.n_vars, 1)
        self.assertEqual(float(out.X[0, 0]), 1.0)


class TestOrthologPolicyConfig(unittest.TestCase):
    def setUp(self):
        gene_converter._TABLE_CACHE.clear()

    def test_invalid_policy(self):
        with self.assertRaises(ValueError):
            registry.parse_species_config({"ortholog_policy": "sum_all"})

    def test_one2one_table_drops_ambiguous_sources(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "mouse_to_human.tsv").write_text(
                "source_id\ttarget_id\n"
                "ENSMUSG00000000001\tENSG00000000001\n"
                "ENSMUSG00000000002\tENSG00000000002\n"
                "ENSMUSG00000000002\tENSG00000000003\n"
                "ENSMUSG00000000004\tENSG00000000004\n"
                "ENSMUSG00000000005\tENSG00000000004\n",
                encoding="utf-8",
            )
            table = gene_converter.load_ortholog_table(
                gene_converter.ConversionPair.MOUSE_TO_HUMAN,
                orthologs_dir=base,
                policy="one2one",
            )
            self.assertIn("ENSMUSG00000000001", table)
            self.assertNotIn("ENSMUSG00000000002", table)
            self.assertNotIn("ENSMUSG00000000004", table)
            self.assertNotIn("ENSMUSG00000000005", table)

    def test_legacy_sum_last_write_wins(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "mouse_to_human.tsv").write_text(
                "source_id\ttarget_id\n"
                "ENSMUSG00000000002\tENSG00000000002\n"
                "ENSMUSG00000000002\tENSG00000000003\n",
                encoding="utf-8",
            )
            table = gene_converter.load_ortholog_table(
                gene_converter.ConversionPair.MOUSE_TO_HUMAN,
                orthologs_dir=base,
                policy="legacy_sum",
            )
            self.assertEqual(table["ENSMUSG00000000002"], "ENSG00000000003")


class TestBatchDefaults(unittest.TestCase):
    def test_species_aware_batch_defaults(self):
        self.assertEqual(species_context.default_isp_forward_batch_size(2048), 100)
        self.assertEqual(species_context.default_isp_forward_batch_size(4096), 25)
        self.assertEqual(species_context.default_finetune_batch_size(2048), 6)
        self.assertEqual(species_context.default_finetune_batch_size(4096), 2)


if __name__ == "__main__":
    unittest.main()
