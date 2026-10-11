"""Anchor-gene combination batch pairs the anchor with every other gene in the cell."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "core", ROOT / "contracts"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


class TestAnchorComboBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Not at module level: importing the perturber before test_p0_species stubs the
        # geneformer package breaks datasets.map pickling in later ISP tests.
        try:
            from geneformer import in_silico_perturber as isp
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest(f"deps missing: {exc}")
        cls.isp = isp

    def test_delete_pairs_anchor_with_each_gene(self):
        example_cell = {"input_ids": [[5, 6, 7, 8]], "length": [4]}
        batch, indices = self.isp.make_perturbation_batch(
            example_cell,
            "delete",
            [7],
            None,
            None,
            [7],
            1,
            1,
        )
        self.assertEqual(indices, [[0, 2], [1, 2], [2, 3]])
        self.assertEqual(batch["input_ids"], [[6, 8], [5, 8], [5, 6]])


if __name__ == "__main__":
    unittest.main()
