"""Project-scoped curated ortholog overlays (do not edit global *_curated.tsv)."""
from __future__ import annotations

import importlib.util
import os
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
sys.path.insert(0, str(ROOT))


def _load_gene_converter():
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

    registry = _load(
        "geneformer.backends.registry",
        ROOT / "core" / "geneformer" / "backends" / "registry.py",
    )
    backends_pkg.registry = registry
    for name in (
        "BackendSpec",
        "HumanVariant",
        "ModelId",
        "MouseVariant",
        "Organism",
        "OrthologPolicy",
        "get_backend",
        "native_organism",
        "needs_gene_conversion",
        "parse_species_config",
        "resolve_pretrained_path",
    ):
        setattr(backends_pkg, name, getattr(registry, name))
    return _load(
        "geneformer.gene_converter",
        ROOT / "core" / "geneformer" / "gene_converter.py",
    )


gc = _load_gene_converter()

POU5F1 = "ENSG00000204531"
POU5F1B = "ENSG00000212993"
POU5F1_MOUSE = "ENSMUSG00000024406"
ORTHOLOGS = ROOT / "core" / "geneformer" / "dicts" / "orthologs"


@unittest.skipUnless(
    (ORTHOLOGS / "human_to_mouse.tsv").is_file(),
    "production ortholog TSV missing",
)
class TestCuratedOverlay(unittest.TestCase):
    def setUp(self):
        gc._TABLE_CACHE.clear()
        self._prev_env = os.environ.pop("GENEFORMER_ORTHOLOG_CURATED_OVERLAY", None)

    def tearDown(self):
        gc._TABLE_CACHE.clear()
        if self._prev_env is None:
            os.environ.pop("GENEFORMER_ORTHOLOG_CURATED_OVERLAY", None)
        else:
            os.environ["GENEFORMER_ORTHOLOG_CURATED_OVERLAY"] = self._prev_env

    def test_one2one_drops_pou5f1_without_overlay(self):
        table = gc.load_ortholog_table(
            gc.ConversionPair.HUMAN_TO_MOUSE, policy="one2one"
        )
        self.assertNotIn(POU5F1, table)
        table_m = gc.load_ortholog_table(
            gc.ConversionPair.MOUSE_TO_HUMAN, policy="one2one"
        )
        self.assertNotIn(POU5F1_MOUSE, table_m)

    def test_overlay_file_adds_ensembl_bridge_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            overlay = Path(tmp) / "bridge.tsv"
            overlay.write_text(
                f"source_id\ttarget_id\n{POU5F1}\t{POU5F1_MOUSE}\n",
                encoding="utf-8",
            )
            table = gc.load_ortholog_table(
                gc.ConversionPair.HUMAN_TO_MOUSE,
                policy="one2one",
                curated_overlay=overlay,
            )
            self.assertEqual(table[POU5F1], POU5F1_MOUSE)
            self.assertNotIn("POU5F1", table)
            self.assertNotIn(POU5F1B, table)

    def test_overlay_directory_resolves_pair_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "curated_bridge_human_to_mouse.tsv").write_text(
                f"source_id\ttarget_id\n{POU5F1}\t{POU5F1_MOUSE}\n",
                encoding="utf-8",
            )
            (root / "curated_bridge_mouse_to_human.tsv").write_text(
                f"source_id\ttarget_id\n{POU5F1_MOUSE}\t{POU5F1}\n",
                encoding="utf-8",
            )
            h2m = gc.load_ortholog_table(
                gc.ConversionPair.HUMAN_TO_MOUSE,
                policy="one2one",
                curated_overlay=root,
            )
            m2h = gc.load_ortholog_table(
                gc.ConversionPair.MOUSE_TO_HUMAN,
                policy="one2one",
                curated_overlay=root,
            )
            self.assertEqual(h2m[POU5F1], POU5F1_MOUSE)
            self.assertEqual(m2h[POU5F1_MOUSE], POU5F1)

    def test_env_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            overlay = Path(tmp) / "curated_bridge_human_to_mouse.tsv"
            overlay.write_text(
                f"source_id\ttarget_id\n{POU5F1}\t{POU5F1_MOUSE}\n",
                encoding="utf-8",
            )
            os.environ["GENEFORMER_ORTHOLOG_CURATED_OVERLAY"] = str(Path(tmp))
            table = gc.load_ortholog_table(
                gc.ConversionPair.HUMAN_TO_MOUSE, policy="one2one"
            )
            self.assertEqual(table[POU5F1], POU5F1_MOUSE)

    def test_species_config_preserves_overlay_path(self):
        from geneformer.backends.registry import parse_species_config

        parsed = parse_species_config(
            {
                "model_organism": "human",
                "model": "mouse_geneformer",
                "ortholog_policy": "one2one",
                "ortholog_curated_overlay": "/tmp/policy_v1",
            }
        )
        self.assertEqual(parsed["ortholog_curated_overlay"], "/tmp/policy_v1")


if __name__ == "__main__":
    unittest.main()
