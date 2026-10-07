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
NANOG = "ENSG00000111704"
NANOGP8 = "ENSG00000255192"
NANOG_MOUSE = "ENSMUSG00000012396"
ORTHOLOGS = ROOT / "core" / "geneformer" / "dicts" / "orthologs"
# The platform curated tables restore POU5F1 and NANOG; the overlay tests need a gene they drop.
CURATED_PLURIPOTENCY_SOURCES = {
    POU5F1, "POU5F1", NANOG, "NANOG",
    POU5F1_MOUSE, "Pou5f1", NANOG_MOUSE, "Nanog",
}


def orthologs_without_curated_pluripotency(dest: Path) -> Path:
    """Platform ortholog dir with the POU5F1 / NANOG rows removed from ``*_curated.tsv``."""
    for src in ORTHOLOGS.iterdir():
        if not src.is_file():
            continue
        out = dest / src.name
        if src.name.endswith("_curated.tsv"):
            lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
            kept = [ln for ln in lines if ln.split("\t", 1)[0] not in CURATED_PLURIPOTENCY_SOURCES]
            out.write_text("".join(kept), encoding="utf-8")
        else:
            out.symlink_to(src)
    return dest


@unittest.skipUnless(
    (ORTHOLOGS / "human_to_mouse.tsv").is_file(),
    "production ortholog TSV missing",
)
class TestCuratedOverlay(unittest.TestCase):
    def setUp(self):
        gc._TABLE_CACHE.clear()
        self._prev_env = os.environ.pop("GENEFORMER_ORTHOLOG_CURATED_OVERLAY", None)
        self._orthologs_tmp = tempfile.TemporaryDirectory()
        self._prev_dir = gc.ORTHOLOGS_DIR
        gc.ORTHOLOGS_DIR = orthologs_without_curated_pluripotency(Path(self._orthologs_tmp.name))

    def tearDown(self):
        gc.ORTHOLOGS_DIR = self._prev_dir
        self._orthologs_tmp.cleanup()
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


@unittest.skipUnless(
    (ORTHOLOGS / "human_to_mouse.tsv").is_file(),
    "production ortholog TSV missing",
)
class TestDefaultCuratedPluripotency(unittest.TestCase):
    """POU5F1 and NANOG are one2many in Ensembl; the platform curated tables pair them 1:1."""

    def setUp(self):
        gc._TABLE_CACHE.clear()
        self._prev_env = os.environ.pop("GENEFORMER_ORTHOLOG_CURATED_OVERLAY", None)

    def tearDown(self):
        gc._TABLE_CACHE.clear()
        if self._prev_env is not None:
            os.environ["GENEFORMER_ORTHOLOG_CURATED_OVERLAY"] = self._prev_env

    def test_one2one_maps_pou5f1_and_nanog_both_directions(self):
        h2m = gc.load_ortholog_table(gc.ConversionPair.HUMAN_TO_MOUSE, policy="one2one")
        m2h = gc.load_ortholog_table(gc.ConversionPair.MOUSE_TO_HUMAN, policy="one2one")
        self.assertEqual(h2m[POU5F1], POU5F1_MOUSE)
        self.assertEqual(h2m[NANOG], NANOG_MOUSE)
        self.assertEqual(m2h[POU5F1_MOUSE], POU5F1)
        self.assertEqual(m2h[NANOG_MOUSE], NANOG)

    def test_paralogues_stay_unmapped(self):
        h2m = gc.load_ortholog_table(gc.ConversionPair.HUMAN_TO_MOUSE, policy="one2one")
        self.assertNotIn(POU5F1B, h2m)
        self.assertNotIn(NANOGP8, h2m)

    def test_curated_rows_add_only_these_genes(self):
        h2m = gc.load_ortholog_table(gc.ConversionPair.HUMAN_TO_MOUSE, policy="one2one")
        m2h = gc.load_ortholog_table(gc.ConversionPair.MOUSE_TO_HUMAN, policy="one2one")
        with tempfile.TemporaryDirectory() as tmp:
            gc._TABLE_CACHE.clear()
            stripped = orthologs_without_curated_pluripotency(Path(tmp))
            h2m_base = gc.load_ortholog_table(
                gc.ConversionPair.HUMAN_TO_MOUSE, policy="one2one", orthologs_dir=stripped
            )
            m2h_base = gc.load_ortholog_table(
                gc.ConversionPair.MOUSE_TO_HUMAN, policy="one2one", orthologs_dir=stripped
            )
        self.assertEqual(
            {k for k in set(h2m) - set(h2m_base) if k.startswith("ENS")}, {POU5F1, NANOG}
        )
        self.assertEqual(
            {k for k in set(m2h) - set(m2h_base) if k.startswith("ENS")},
            {POU5F1_MOUSE, NANOG_MOUSE},
        )


if __name__ == "__main__":
    unittest.main()
