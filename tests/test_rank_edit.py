"""Tests for the rank-edit step operators (length-preserving OE / KD on rank-value encodings)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))

from rank_edit import (
    apply_rank_edits,
    apply_single_step_delete,
    apply_single_step_overexpress,
    apply_step,
    legacy_steps_block,
    normalize_step_type,
    parse_steps,
    perturb_index_for_tokens,
)


def _fake_example(tokens: list[int]) -> dict:
    return {"input_ids": list(tokens), "length": len(tokens)}


class TestOverexpress(unittest.TestCase):
    def test_later_step_is_leftmost(self):
        ex = _fake_example(list(range(10, 20)))
        out = apply_rank_edits(ex, [("overexpress", [101]), ("overexpress", [102])])
        self.assertEqual(out["input_ids"][:2], [102, 101])
        self.assertEqual(len(out["input_ids"]), len(ex["input_ids"]))

    def test_group_step_keeps_listed_order(self):
        out = apply_single_step_overexpress(_fake_example(list(range(50, 70))), [201, 202])
        self.assertEqual(out["input_ids"][:2], [201, 202])

    def test_length_preserved_when_inserting_absent(self):
        ex = _fake_example(list(range(100, 120)))
        out = apply_single_step_overexpress(ex, [999])
        self.assertEqual(len(out["input_ids"]), len(ex["input_ids"]))
        self.assertEqual(out["input_ids"][0], 999)
        self.assertEqual(out["input_ids"][-1], 118)

    def test_present_gene_moves_to_front(self):
        out = apply_single_step_overexpress(_fake_example([10, 11, 12, 13]), [12])
        self.assertEqual(out["input_ids"], [12, 10, 11, 13])

    def test_perturb_index(self):
        self.assertEqual(perturb_index_for_tokens([5, 6, 7], [7, 9]), [2])
        self.assertEqual(perturb_index_for_tokens([5, 6, 7], [9]), [-100])


class TestDelete(unittest.TestCase):
    def test_delete_removes_token_and_shortens(self):
        out = apply_single_step_delete(_fake_example([10, 20, 30, 40]), [20, 40])
        self.assertEqual(out["input_ids"], [10, 30])
        self.assertEqual(out["length"], 2)
        self.assertEqual(out["attention_mask"], [1, 1])

    def test_delete_absent_is_noop(self):
        out = apply_single_step_delete(_fake_example([10, 20, 30]), [99])
        self.assertEqual(out["input_ids"], [10, 20, 30])
        self.assertEqual(out["length"], 3)

    def test_oe_then_kd(self):
        ex = _fake_example(list(range(10, 20)))
        out = apply_rank_edits(ex, [("overexpress", [101]), ("delete", [12])])
        self.assertEqual(out["input_ids"][0], 101)
        self.assertNotIn(12, out["input_ids"])
        self.assertEqual(len(out["input_ids"]), len(ex["input_ids"]) - 1)

    def test_kd_then_oe_inserts_at_front(self):
        ex = _fake_example(list(range(10, 20)))
        out = apply_rank_edits(ex, [("delete", [12]), ("overexpress", [101])])
        self.assertEqual(out["input_ids"][0], 101)
        self.assertNotIn(12, out["input_ids"])
        self.assertEqual(len(out["input_ids"]), len(ex["input_ids"]) - 1)

    def test_apply_step_aliases(self):
        self.assertEqual(normalize_step_type("OE"), "overexpress")
        self.assertEqual(normalize_step_type("kd"), "delete")
        self.assertEqual(apply_step(_fake_example([1, 2, 3]), [2], "KD")["input_ids"], [1, 3])
        with self.assertRaises(ValueError):
            normalize_step_type("activate")


class TestParseSteps(unittest.TestCase):
    def test_parse_steps_yaml(self):
        steps = parse_steps(
            {
                "steps": [
                    {"name": "oe1", "type": "OE", "genes": ["Pou5f1", "Sox2"]},
                    {"type": "delete", "genes_to_perturb": "Igfbp2"},
                ]
            }
        )
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]["type"], "overexpress")
        self.assertEqual(steps[0]["genes"], ["Pou5f1", "Sox2"])
        self.assertEqual(steps[1]["type"], "delete")
        self.assertEqual(steps[1]["name"], "step02_delete")
        self.assertEqual(steps[1]["genes"], ["Igfbp2"])

    def test_empty_steps(self):
        self.assertEqual(parse_steps({}), [])
        self.assertEqual(parse_steps({"steps": []}), [])

    def test_step_without_genes_raises(self):
        with self.assertRaises(ValueError):
            parse_steps({"steps": [{"type": "OE", "genes": []}]})


class TestLegacyStepsBlock(unittest.TestCase):
    def test_ordered_rank_edit_key(self):
        block = legacy_steps_block({"ordered_rank_edit": {"steps": [{"type": "OE", "genes": ["A"]}]}})
        self.assertEqual(parse_steps(block)[0]["genes"], ["A"])

    def test_sequential_key(self):
        block = legacy_steps_block({"sequential": {"steps": [{"type": "OE", "genes": ["B"]}]}})
        self.assertEqual(parse_steps(block)[0]["genes"], ["B"])

    def test_ordered_rank_edit_wins_over_sequential(self):
        block = legacy_steps_block({"ordered_rank_edit": {"x": 1}, "sequential": {"x": 2}})
        self.assertEqual(block["x"], 1)

    def test_missing(self):
        self.assertEqual(legacy_steps_block({}), {})
        self.assertEqual(legacy_steps_block(None), {})


if __name__ == "__main__":
    unittest.main()
