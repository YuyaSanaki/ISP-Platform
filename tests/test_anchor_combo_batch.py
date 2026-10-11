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

try:
    from geneformer.in_silico_perturber import make_perturbation_batch

    DEPS_MISSING = None
except Exception as exc:  # pragma: no cover
    DEPS_MISSING = str(exc)


@unittest.skipIf(DEPS_MISSING is not None, f"deps missing: {DEPS_MISSING}")
class TestAnchorComboBatch(unittest.TestCase):
    def test_delete_pairs_anchor_with_each_gene(self):
        example_cell = {"input_ids": [[5, 6, 7, 8]], "length": [4]}
        batch, indices = make_perturbation_batch(
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
