"""E2E TOP1 ISP UMAP gene selection must use TOP1 significant (not raw shift)."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_auto_batch_and_isp_validation import _load_run_pipeline_validators


class TestPickE2ETop1SignificantGene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rp = _load_run_pipeline_validators()

    def test_picks_top_significant_from_csv_not_raw_shifter(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats = Path(tmp)
            # Non-significant high-shift gene that old fallback would prefer.
            with open(stats / "top100_positive_shifters.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=["Gene_name", "Ensembl_ID", "Shift_to_goal_end", "Sig"],
                )
                w.writeheader()
                w.writerow(
                    {
                        "Gene_name": "HighShiftNoise",
                        "Ensembl_ID": "ENSG_NOISE",
                        "Shift_to_goal_end": "0.99",
                        "Sig": "0",
                    }
                )
                w.writerow(
                    {
                        "Gene_name": "Igfbp2",
                        "Ensembl_ID": "ENSG00000115457",
                        "Shift_to_goal_end": "0.24",
                        "Sig": "1",
                    }
                )
            with open(stats / "significant_genes.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=["Gene_name", "Ensembl_ID", "Shift_to_goal_end", "Sig"],
                )
                w.writeheader()
                w.writerow(
                    {
                        "Gene_name": "Igfbp2",
                        "Ensembl_ID": "ENSG00000115457",
                        "Shift_to_goal_end": "0.24",
                        "Sig": "1",
                    }
                )
                w.writerow(
                    {
                        "Gene_name": "Bmx",
                        "Ensembl_ID": "ENSG00000102010",
                        "Shift_to_goal_end": "0.06",
                        "Sig": "1",
                    }
                )

            gene = self.rp.pick_e2e_top1_significant_gene(stats)
            self.assertEqual(gene, "ENSG00000115457")

    def test_empty_significant_skips_even_if_top100_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats = Path(tmp)
            with open(stats / "significant_genes.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=["Gene_name", "Ensembl_ID", "Shift_to_goal_end", "Sig"],
                )
                w.writeheader()
            with open(stats / "top100_positive_shifters.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=["Gene_name", "Ensembl_ID", "Shift_to_goal_end", "Sig"],
                )
                w.writeheader()
                w.writerow(
                    {
                        "Gene_name": "HighShiftNoise",
                        "Ensembl_ID": "ENSG_NOISE",
                        "Shift_to_goal_end": "0.99",
                        "Sig": "0",
                    }
                )
            self.assertIsNone(self.rp.pick_e2e_top1_significant_gene(stats))

    def test_parquet_sig_filter_fallback(self):
        class _FakeDF:
            def __init__(self, rows):
                self._rows = list(rows)
                self.empty = not self._rows
                self.columns = ["Gene_name", "Ensembl_ID", "Shift_to_goal_end", "Sig"]

            def __getitem__(self, key):
                if isinstance(key, str):
                    raise TypeError("column getitem unused")
                # boolean mask from Sig==1 comparison via overloaded ops below
                return _FakeDF([r for r, keep in zip(self._rows, key) if keep])

            def sort_values(self, col, ascending=True):
                rows = sorted(self._rows, key=lambda r: r[col], reverse=not ascending)
                return _FakeSorted(rows)

        class _FakeSorted:
            def __init__(self, rows):
                self._rows = rows

            @property
            def iloc(self):
                return self

            def __getitem__(self, idx):
                return self._rows[idx]

        class _SigSeries:
            def __init__(self, values):
                self._values = values

            def __eq__(self, other):
                return [v == other for v in self._values]

        # Patch pandas.DataFrame-like read_parquet path used inside the helper.
        rows = [
            {
                "Gene_name": "Noise",
                "Ensembl_ID": "ENSG_NOISE",
                "Shift_to_goal_end": 0.9,
                "Sig": 0,
            },
            {
                "Gene_name": "Igfbp2",
                "Ensembl_ID": "ENSG00000115457",
                "Shift_to_goal_end": 0.2,
                "Sig": 1,
            },
        ]

        class _DF(_FakeDF):
            def __getitem__(self, key):
                if key == "Sig":
                    return _SigSeries([r["Sig"] for r in self._rows])
                if isinstance(key, list) and all(isinstance(x, bool) for x in key):
                    return _FakeDF([r for r, keep in zip(self._rows, key) if keep])
                raise KeyError(key)

        with tempfile.TemporaryDirectory() as tmp:
            stats = Path(tmp)
            (stats / "stats.parquet").write_bytes(b"unused")
            fake_pd = mock.MagicMock()
            fake_pd.read_parquet.return_value = _DF(rows)
            with mock.patch.dict("sys.modules", {"pandas": fake_pd}):
                gene = self.rp.pick_e2e_top1_significant_gene(stats)
            self.assertEqual(gene, "ENSG00000115457")


if __name__ == "__main__":
    unittest.main()
