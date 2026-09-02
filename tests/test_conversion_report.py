"""Tests for pre-tokenize conversion reporting."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "core", ROOT / "contracts", ROOT / "webui"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

FIXTURES = ROOT / "tests" / "fixtures" / "orthologs"
sys.path.insert(0, str(ROOT))


def _load_modules():
    geneformer_pkg = types.ModuleType("geneformer")
    backends_pkg = types.ModuleType("geneformer.backends")
    sys.modules["geneformer"] = geneformer_pkg
    sys.modules["geneformer.backends"] = backends_pkg

    def _load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

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
    brief = _load(
        "geneformer.gene_brief_function",
        ROOT / "core" / "geneformer" / "gene_brief_function.py",
    )
    geneformer_pkg.gene_brief_function = brief
    gc = _load("geneformer.gene_converter", ROOT / "core" / "geneformer" / "gene_converter.py")
    sc = _load("geneformer.species_context", ROOT / "core" / "geneformer" / "species_context.py")
    geneformer_pkg.species_context = sc
    report = _load("geneformer.conversion_report", ROOT / "core" / "geneformer" / "conversion_report.py")
    return gc, report


gc, report = _load_modules()

FLY_HUMAN = {"model_organism": "drosophila", "model": "human_geneformer", "human_variant": "v2_104m"}
MOUSE_HUMAN = {"model_organism": "mouse", "model": "human_geneformer", "human_variant": "v2_104m"}
MOUSE_NATIVE = {"model_organism": "mouse", "model": "mouse_geneformer"}


class TestConversionReportDefaults(unittest.TestCase):
    def test_default_on_for_fly(self):
        self.assertTrue(report.default_report_conversion(FLY_HUMAN))

    def test_default_on_for_mouse_cross_species(self):
        self.assertTrue(report.default_report_conversion(MOUSE_HUMAN))

    def test_default_off_for_mouse_native(self):
        self.assertFalse(report.default_report_conversion(MOUSE_NATIVE))

    def test_cli_overrides_config(self):
        self.assertFalse(
            report.resolve_report_conversion(
                FLY_HUMAN,
                {"report_conversion": True},
                cli_flag=False,
            )
        )
        self.assertTrue(
            report.resolve_report_conversion(
                MOUSE_NATIVE,
                {"report_conversion": False},
                cli_flag=True,
            )
        )


class TestConversionReportAnalysis(unittest.TestCase):
    def setUp(self):
        gc._TABLE_CACHE.clear()

    def test_fly_fixture_mapping_counts(self):
        genes = ["FBgn0039044", "FBgn0050169", "FBgn9999999"]
        result = report.analyze_gene_conversion(genes, FLY_HUMAN, orthologs_dir=FIXTURES)
        self.assertEqual(result.input_genes, 3)
        self.assertEqual(result.mapped, 2)
        self.assertEqual(result.unmapped, 1)
        self.assertIn("FBgn9999999", result.unmapped_gene_ids)

    def test_writes_report_files(self):
        genes = ["FBgn0039044", "FBgn0050169", "FBgn9999999"]
        summary = report.analyze_gene_conversion(genes, FLY_HUMAN, orthologs_dir=FIXTURES)
        with tempfile.TemporaryDirectory() as tmp:
            paths = report.write_conversion_report(summary, tmp)
            self.assertTrue(Path(paths["json"]).is_file())
            self.assertTrue(Path(paths["text"]).is_file())
            self.assertTrue(Path(paths["unmapped_tsv"]).is_file())
            text = Path(paths["text"]).read_text(encoding="utf-8")
            self.assertIn("Gene conversion report", text)
            self.assertIn("unmapped (dropped):   1", text)
            tsv = Path(paths["unmapped_tsv"]).read_text(encoding="utf-8")
            self.assertIn("source_id", tsv.splitlines()[0])
            self.assertIn("brief_function", tsv.splitlines()[0])
            self.assertIn("FBgn9999999", tsv)


if __name__ == "__main__":
    unittest.main()
