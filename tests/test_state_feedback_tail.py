"""State-feedback reranks keep the genes beyond the model input window."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))

try:
    from datasets import Dataset

    from state_feedback import feedback as fb

    DEPS_MISSING = None
except Exception as exc:  # pragma: no cover
    DEPS_MISSING = str(exc)


@unittest.skipIf(DEPS_MISSING is not None, f"deps missing: {DEPS_MISSING}")
class TestKeepTail(unittest.TestCase):
    def test_tail_after_reranked_window_is_kept(self):
        self.assertEqual(fb.keep_tail([1, 2, 3, 4, 5], [3, 1, 2]), [3, 1, 2, 4, 5])

    def test_rows_within_the_window_are_unchanged(self):
        self.assertEqual(fb.keep_tail([1, 2, 3], [3, 2, 1]), [3, 2, 1])

    def test_new_gene_set_is_not_padded_with_old_tail(self):
        self.assertEqual(fb.keep_tail([1, 2, 3, 4], [9, 1]), [9, 1])

    def test_replace_input_ids_keeps_tail_and_length(self):
        ds = Dataset.from_dict(
            {
                "input_ids": [[1, 2, 3, 4], [5, 6]],
                "length": [4, 2],
                "attention_mask": [[1] * 4, [1] * 2],
            }
        )
        out = fb.replace_input_ids(ds, [[2, 1], [6, 5]])
        self.assertEqual(out["input_ids"], [[2, 1, 3, 4], [6, 5]])
        self.assertEqual(out["length"], [4, 2])


if __name__ == "__main__":
    unittest.main()
