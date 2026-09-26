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


if __name__ == "__main__":
    unittest.main()
