"""Dropped-gene preview for Web UI + brief-function lookup."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "core", ROOT / "webui"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)


def _load_brief():
    path = ROOT / "core" / "geneformer" / "gene_brief_function.py"
    spec = importlib.util.spec_from_file_location("gene_brief_function_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


brief = _load_brief()
from dropped_genes_ui import load_dropped_gene_table  # noqa: E402


class TestBriefFunctionLookup(unittest.TestCase):
    def test_curated_pou5f1(self):
        note = brief.lookup_brief_function("ENSG00000204531", "POU5F1")
        self.assertIn("pluripotency", note.lower())

    def test_drop_reason_one2many(self):
        self.assertIn("one-to-many", brief.explain_drop_reason("ortholog_one2many").lower())

    def test_symbol_for_curated_id(self):
        self.assertEqual(brief.lookup_symbol("ENSG00000204531"), "POU5F1")


class TestDroppedGeneTable(unittest.TestCase):
    def test_loads_unmapped_tsv_and_enriches(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            tok = run / "tokenized_dataset"
            tok.mkdir()
            (tok / "conversion_report.json").write_text(
                json.dumps(
                    {
                        "conversion_pair": "human_to_mouse",
                        "ortholog_policy": "one2one",
                        "input_genes": 3,
                        "mapped": 2,
                        "unmapped": 1,
                        "mapped_pct": 66.67,
                        "unmapped_gene_ids": ["ENSG00000204531"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with open(tok / "conversion_unmapped_genes.tsv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["source_id"])
                w.writerow(["ENSG00000204531"])
            table = load_dropped_gene_table([run])
            self.assertIsNotNone(table)
            assert table is not None
            self.assertEqual(table.unmapped, 1)
            self.assertEqual(table.rows[0]["Gene ID"], "ENSG00000204531")
            self.assertEqual(table.rows[0]["Symbol"], "POU5F1")
            self.assertIn("pluripotency", table.rows[0]["Brief function"].lower())

    def test_missing_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_dropped_gene_table([tmp]))


if __name__ == "__main__":
    unittest.main()
