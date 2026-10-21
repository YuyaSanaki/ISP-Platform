"""Unit tests for State-feedback Phase 0 oracle rerank helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))

from state_feedback.oracle_rerank import (
    build_pseudobulk_rank_priority,
    rerank_fixed_gene_set,
    spearman_rank_correlation,
)


class TestOracleRerank(unittest.TestCase):
    def test_fixed_gene_set_preserved(self):
        ids = [10, 20, 30, 40]
        # Prefer 40 then 10 then 20 then 30
        priority = {40: 0.0, 10: 1.0, 20: 2.0, 30: 3.0}
        out = rerank_fixed_gene_set(ids, priority)
        self.assertEqual(sorted(out), sorted(ids))
        self.assertEqual(out, [40, 10, 20, 30])

    def test_missing_priority_sinks_right_stable(self):
        ids = [1, 2, 3]
        priority = {2: 0.0}
        out = rerank_fixed_gene_set(ids, priority)
        self.assertEqual(out[0], 2)
        self.assertEqual(set(out), {1, 2, 3})

    def test_pseudobulk_priority(self):
        rows = [
            [1, 2, 3],
            [1, 3, 2],
        ]
        pr = build_pseudobulk_rank_priority(rows)
        self.assertLess(pr[1], pr[2])
        self.assertLess(pr[1], pr[3])

    def test_spearman_identical(self):
        a = [5, 4, 3, 2]
        self.assertAlmostEqual(spearman_rank_correlation(a, list(a)), 1.0)

    def test_spearman_reversed(self):
        a = [1, 2, 3, 4]
        b = [4, 3, 2, 1]
        self.assertAlmostEqual(spearman_rank_correlation(a, b), -1.0)


try:
    import run_state_feedback_oracle as sfo

    ORACLE_DEPS_MISSING = None
except Exception as exc:  # pragma: no cover
    ORACLE_DEPS_MISSING = str(exc)


@unittest.skipIf(ORACLE_DEPS_MISSING is not None, f"deps missing: {ORACLE_DEPS_MISSING}")
class TestOraclePermuteKeepsOverexpressedFront(unittest.TestCase):
    def test_pinned_oe_genes_stay_first_in_their_order(self):
        # OE genes 7 and 9 lead the cell; the oracle prefers 30 and 40, then the OE genes.
        ex = {"input_ids": [9, 7, 10, 20, 30, 40]}
        priority = {30: 0.0, 40: 1.0, 7: 2.0, 9: 3.0, 10: 4.0, 20: 5.0}
        out = sfo._apply_oracle(ex, priority, (7, 9))["input_ids"]
        self.assertEqual(out[:2], [9, 7])
        self.assertEqual(out[2:], [30, 40, 10, 20])
        self.assertEqual(sorted(out), sorted(ex["input_ids"]))

    def test_unpinned_permute_matches_earlier_behaviour(self):
        ex = {"input_ids": [9, 7, 10, 20, 30, 40]}
        priority = {30: 0.0, 40: 1.0, 7: 2.0, 9: 3.0, 10: 4.0, 20: 5.0}
        out = sfo._apply_oracle(ex, priority)["input_ids"]
        self.assertEqual(out, rerank_fixed_gene_set(ex["input_ids"], priority))
        self.assertNotEqual(out[:2], [9, 7])


if __name__ == "__main__":
    unittest.main()
