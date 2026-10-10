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
    import torch

    from geneformer.in_silico_perturber import (
        oe_indices_to_remove_for_alignment,
        overexpress_tokens,
        remove_indices_from_emb,
    )

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


def _group_oe_gene_lengths(orig_len, n_absent, n_present):
    """Gene counts compared after group OE: (perturbed minus OE genes, original minus removed).

    Present OE genes sit at the tail of the cell, the case where trailing-index
    padding used to collide with them.
    """
    absent = [100_000 + i for i in range(n_absent)]
    present_idx = list(range(orig_len - n_present, orig_len))
    ids = list(range(1, orig_len + 1))
    present = [ids[i] for i in present_idx]
    tokens = absent + present
    example = {
        "input_ids": list(ids),
        "perturb_index": present_idx if present_idx else [-100],
        "tokens_to_perturb": tokens,
        "length": orig_len,
    }
    out = overexpress_tokens(example)
    removed = oe_indices_to_remove_for_alignment(example["perturb_index"], orig_len, len(tokens))
    return len(out["input_ids"]) - len(tokens), orig_len - len(set(removed)), out


@unittest.skipIf(DEPS_MISSING is not None, f"deps missing: {DEPS_MISSING}")
class TestUpstreamOverflowBand(unittest.TestCase):
    """Official Geneformer 1f7fbae misaligns for max_len - 2k < length < max_len (k absent OE genes).

    See docs/upstream_overexpression.md. The length-preserving rule has no model
    window, so the same inputs must align.
    """

    MAX_LEN = 4096

    def test_lane_b_case2_shapes(self):
        # Pegasus reproduction: V2-104M, length 4094, OSKM all absent.
        # Official Geneformer compared 4092 with 4090 genes and raised.
        orig_len, hidden = self.MAX_LEN - 2, 8
        pert_len, orig_kept, out = _group_oe_gene_lengths(orig_len, n_absent=4, n_present=0)
        self.assertEqual(len(out["input_ids"]), orig_len)
        self.assertEqual(out["length"], orig_len)
        self.assertEqual(pert_len, orig_kept)
        self.assertEqual(pert_len, orig_len - 4)

        perturbed_emb = torch.zeros(orig_len, hidden)[4:, :]
        removed = oe_indices_to_remove_for_alignment([-100], orig_len, 4)
        original_emb = remove_indices_from_emb(torch.zeros(orig_len, hidden), removed, 0)
        self.assertEqual(perturbed_emb.shape, original_emb.shape)

    def test_whole_band_aligns(self):
        for n_absent, n_present in [(1, 0), (2, 2), (3, 1), (4, 0)]:
            lo = self.MAX_LEN - 2 * n_absent - 1
            for orig_len in range(lo, self.MAX_LEN + 1):
                with self.subTest(n_absent=n_absent, n_present=n_present, orig_len=orig_len):
                    pert_len, orig_kept, out = _group_oe_gene_lengths(orig_len, n_absent, n_present)
                    self.assertEqual(len(out["input_ids"]), orig_len)
                    self.assertEqual(pert_len, orig_kept)


if __name__ == "__main__":
    unittest.main()
