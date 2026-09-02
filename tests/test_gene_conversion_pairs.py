"""Cross-species ortholog resolution for all four conversion pairs."""
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

FIXTURES = ROOT / "tests" / "fixtures" / "orthologs"
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
    gc = _load("geneformer.gene_converter", ROOT / "core" / "geneformer" / "gene_converter.py")
    return gc


gc = _load_gene_converter()

MOUSE_HUMAN = {
    "model_organism": "mouse",
    "model": "human_geneformer",
    "human_variant": "v2_104m",
    "ortholog_policy": "legacy_sum",
}
HUMAN_MOUSE = {
    "model_organism": "human",
    "model": "mouse_geneformer",
    "ortholog_policy": "legacy_sum",
}
FLY_HUMAN = {
    "model_organism": "drosophila",
    "model": "human_geneformer",
    "human_variant": "v2_104m",
    "ortholog_policy": "legacy_sum",
}
FLY_MOUSE = {
    "model_organism": "drosophila",
    "model": "mouse_geneformer",
    "ortholog_policy": "legacy_sum",
}

MOUSE_IGFBP2 = "ENSMUSG00000039323"
HUMAN_IGFBP2 = "ENSG00000115457"
MOUSE_TP53 = "ENSMUSG00000025903"
HUMAN_TP53 = "ENSG00000141510"
# True fly p53 / Brca2 (fixtures mirror BioMart-like / smoke-safe pairs).
FLY_P53 = "FBgn0039044"
HUMAN_TP73 = "ENSG00000078900"
MOUSE_TRP63 = "ENSMUSG00000022510"
FLY_BRCA2 = "FBgn0050169"
HUMAN_BRCA2 = "ENSG00000139618"
MOUSE_BRCA2 = "ENSMUSG00000041147"
FLY_SYMBOLS = {"Tp53": FLY_P53, "p53": FLY_P53, "Brca2": FLY_BRCA2}


class TestAllConversionPairs(unittest.TestCase):
    def setUp(self):
        gc._TABLE_CACHE.clear()

    def _table(self, pair, policy="legacy_sum"):
        # Fixtures are tiny and include intentional BioMart-vs-curated collisions;
        # legacy_sum matches historical LWW+curated-override behavior used by these tests.
        return gc.load_ortholog_table(pair, orthologs_dir=FIXTURES, policy=policy)

    def test_mouse_to_human_ensembl_and_symbol_agree(self):
        table = self._table(gc.ConversionPair.MOUSE_TO_HUMAN)
        self.assertEqual(table[MOUSE_IGFBP2], HUMAN_IGFBP2)
        self.assertEqual(table["Igfbp2"], HUMAN_IGFBP2)
        self.assertEqual(
            gc.resolve_gene_for_model("Igfbp2", MOUSE_HUMAN, input_symbol_table={"Igfbp2": MOUSE_IGFBP2}),
            HUMAN_IGFBP2,
        )
        self.assertEqual(
            gc.resolve_gene_for_model(MOUSE_IGFBP2, MOUSE_HUMAN, input_symbol_table={"Igfbp2": MOUSE_IGFBP2}),
            HUMAN_IGFBP2,
        )

    def test_human_to_mouse_ensembl_and_symbol(self):
        table = self._table(gc.ConversionPair.HUMAN_TO_MOUSE)
        self.assertEqual(table[HUMAN_IGFBP2], MOUSE_IGFBP2)
        self.assertEqual(gc.resolve_gene_for_model(HUMAN_IGFBP2, HUMAN_MOUSE), MOUSE_IGFBP2)
        self.assertEqual(gc.resolve_gene_for_model("IGFBP2", HUMAN_MOUSE), MOUSE_IGFBP2)

    def test_fly_to_human(self):
        table = self._table(gc.ConversionPair.DROSOPHILA_TO_HUMAN)
        self.assertEqual(table[FLY_P53], HUMAN_TP73)
        self.assertEqual(table[FLY_BRCA2], HUMAN_BRCA2)
        self.assertEqual(
            gc.resolve_gene_for_model(FLY_P53, FLY_HUMAN, ortholog_table=table),
            HUMAN_TP73,
        )
        self.assertEqual(
            gc.resolve_gene_for_model(
                "Tp53", FLY_HUMAN, input_symbol_table=FLY_SYMBOLS, ortholog_table=table
            ),
            HUMAN_TP73,
        )

    def test_fly_to_mouse(self):
        table = self._table(gc.ConversionPair.DROSOPHILA_TO_MOUSE)
        self.assertEqual(table[FLY_P53], MOUSE_TRP63)
        self.assertEqual(table[FLY_BRCA2], MOUSE_BRCA2)
        self.assertEqual(
            gc.resolve_gene_for_model(FLY_P53, FLY_MOUSE, ortholog_table=table),
            MOUSE_TRP63,
        )

    def test_curated_overrides_main_table(self):
        """Main table has wrong BioMart-like mapping; curated must win."""
        table = self._table(gc.ConversionPair.MOUSE_TO_HUMAN)
        self.assertEqual(table[MOUSE_IGFBP2], HUMAN_IGFBP2)
        self.assertNotEqual(table[MOUSE_IGFBP2], "ENSG00000017427")

    def test_convert_gene_ids_bulk_mouse_to_human(self):
        result = gc.convert_gene_ids(
            [MOUSE_IGFBP2, MOUSE_TP53],
            model_organism="mouse",
            model_id="human_geneformer",
            ortholog_table=self._table(gc.ConversionPair.MOUSE_TO_HUMAN),
            species=MOUSE_HUMAN,
        )
        self.assertEqual(result.mapped[MOUSE_IGFBP2], HUMAN_IGFBP2)
        self.assertEqual(result.mapped[MOUSE_TP53], HUMAN_TP53)

    def test_passthrough_already_converted_ids(self):
        self.assertEqual(gc.resolve_gene_for_model(HUMAN_IGFBP2, MOUSE_HUMAN), HUMAN_IGFBP2)
        self.assertEqual(gc.resolve_gene_for_model(MOUSE_IGFBP2, HUMAN_MOUSE), MOUSE_IGFBP2)

    def test_one_target_per_fly_gene_in_table(self):
        for pair in (gc.ConversionPair.DROSOPHILA_TO_HUMAN, gc.ConversionPair.DROSOPHILA_TO_MOUSE):
            path = FIXTURES / gc._PAIR_TO_FILE[pair]
            rows = path.read_text(encoding="utf-8").strip().splitlines()[1:]
            sources = [line.split("\t", 1)[0] for line in rows if line.strip()]
            self.assertEqual(len(sources), len(set(sources)), f"duplicate FBgn in {path.name}")


if __name__ == "__main__":
    unittest.main()
