"""Length-preserving OE: absent genes insert at front and drop lowest-ranked tails."""
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
    from geneformer.in_silico_perturber import overexpress_tokens

    DEPS_MISSING = None
except Exception as exc:  # pragma: no cover
    DEPS_MISSING = str(exc)


@unittest.skipIf(DEPS_MISSING is not None, f"deps missing: {DEPS_MISSING}")
class TestOverexpressLengthPreserve(unittest.TestCase):
    def test_all_absent_drops_tail(self):
        # Four new OE tokens; lowest four ranks (97–100) must be removed.
        example = {
            "input_ids": list(range(1, 101)),
            "perturb_index": [-100],
            "tokens_to_perturb": [1001, 1002, 1003, 1004],
            "length": 100,
        }
        out = overexpress_tokens(example)
        self.assertEqual(len(out["input_ids"]), 100)
        self.assertEqual(out["length"], 100)
        self.assertEqual(out["input_ids"][:4], [1001, 1002, 1003, 1004])
        self.assertEqual(out["input_ids"][4:], list(range(1, 97)))
        self.assertNotIn(97, out["input_ids"])
        self.assertNotIn(100, out["input_ids"])

    def test_all_present_move_preserves_length(self):
        ids = list(range(1, 21))
        # Move tokens 5,10,15,20 to front (already in encoding).
        example = {
            "input_ids": ids,
            "perturb_index": [4, 9, 14, 19],
            "tokens_to_perturb": [5, 10, 15, 20],
            "length": 20,
        }
        out = overexpress_tokens(example)
        self.assertEqual(len(out["input_ids"]), 20)
        self.assertEqual(out["input_ids"][:4], [5, 10, 15, 20])
        # Remaining genes keep relative order without the moved ones.
        self.assertEqual(
            out["input_ids"][4:],
            [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19],
        )

    def test_partial_present_net_growth_trimmed(self):
        # Two present (deleted then re-inserted) + two new → net +2 → drop 2 tail.
        example = {
            "input_ids": list(range(1, 11)),
            "perturb_index": [0, 1],  # tokens 1 and 2
            "tokens_to_perturb": [1, 2, 101, 102],
            "length": 10,
        }
        out = overexpress_tokens(example)
        self.assertEqual(len(out["input_ids"]), 10)
        self.assertEqual(out["input_ids"][:4], [1, 2, 101, 102])
        # After deleting 1,2 the body was 3..10; insert 4 → length 12; drop 9,10.
        self.assertEqual(out["input_ids"][4:], [3, 4, 5, 6, 7, 8])


if __name__ == "__main__":
    unittest.main()
