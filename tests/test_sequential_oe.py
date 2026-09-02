"""Tests for sequential length-preserving OE on rank-value encodings."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))

from sequential_oe import (
    OSKM_FACTOR_KEYS,
    all_oskm_orders,
    apply_sequential_overexpress,
    apply_single_step_overexpress,
    front_token_order_after_steps,
    order_label,
    simultaneous_token_order,
    tokens_for_order,
)


def _fake_example(tokens: list[int]) -> dict:
    return {"input_ids": list(tokens), "length": len(tokens)}


class TestSequentialOE(unittest.TestCase):
    def test_order_label(self):
        self.assertEqual(order_label(("O", "S", "K", "M")), "O-S-K-M")

    def test_all_orders_count(self):
        self.assertEqual(len(all_oskm_orders()), 24)

    def test_later_step_is_leftmost(self):
        tok_o, tok_s, tok_k = 101, 102, 103
        ex = _fake_example(list(range(10, 20)))
        steps = [[tok_o], [tok_s], [tok_k]]
        out = apply_sequential_overexpress(ex, steps)
        self.assertEqual(out["input_ids"][:3], [tok_k, tok_s, tok_o])
        self.assertEqual(len(out["input_ids"]), len(ex["input_ids"]))

    def test_sequential_differs_from_simultaneous(self):
        tok_o, tok_s = 201, 202
        base = list(range(50, 70))
        ex = _fake_example(base)
        seq = apply_sequential_overexpress(ex, [[tok_o], [tok_s]])
        sim = apply_sequential_overexpress(ex, [[tok_o, tok_s]])
        self.assertNotEqual(seq["input_ids"][:2], sim["input_ids"][:2])
        self.assertEqual(seq["input_ids"][:2], [tok_s, tok_o])
        self.assertEqual(sim["input_ids"][:2], [tok_o, tok_s])

    def test_length_preserved_when_inserting_absent(self):
        ex = _fake_example(list(range(100, 120)))
        absent = 999
        out = apply_single_step_overexpress(ex, [absent])
        self.assertEqual(len(out["input_ids"]), len(ex["input_ids"]))
        self.assertEqual(out["input_ids"][0], absent)

    def test_front_order_helpers(self):
        steps = [[1], [2], [3]]
        self.assertEqual(front_token_order_after_steps(steps), [3, 2, 1])
        self.assertEqual(simultaneous_token_order(steps), [1, 2, 3])

    def test_tokens_for_order(self):
        mapping = {k: i + 1 for i, k in enumerate(OSKM_FACTOR_KEYS)}
        steps = tokens_for_order(("O", "S", "K", "M"), mapping)
        self.assertEqual(steps, [[1], [2], [3], [4]])


class TestSequentialKD(unittest.TestCase):
    def test_delete_removes_token_and_shortens(self):
        from sequential_oe import apply_single_step_delete

        ex = _fake_example([10, 20, 30, 40])
        out = apply_single_step_delete(ex, [20, 40])
        self.assertEqual(out["input_ids"], [10, 30])
        self.assertEqual(out["length"], 2)
        self.assertEqual(out["attention_mask"], [1, 1])

    def test_delete_absent_is_noop(self):
        from sequential_oe import apply_single_step_delete

        ex = _fake_example([10, 20, 30])
        out = apply_single_step_delete(ex, [99])
        self.assertEqual(out["input_ids"], [10, 20, 30])
        self.assertEqual(out["length"], 3)

    def test_oe_then_kd(self):
        from sequential_oe import apply_sequential_perturb

        ex = _fake_example(list(range(10, 20)))
        oe_tok, kd_tok = 101, 12
        out = apply_sequential_perturb(
            ex,
            [("overexpress", [oe_tok]), ("delete", [kd_tok])],
        )
        self.assertEqual(out["input_ids"][0], oe_tok)
        self.assertNotIn(kd_tok, out["input_ids"])
        self.assertEqual(len(out["input_ids"]), len(ex["input_ids"]) - 1)

    def test_kd_then_oe_inserts_at_front(self):
        from sequential_oe import apply_sequential_perturb

        ex = _fake_example(list(range(10, 20)))
        kd_tok, oe_tok = 12, 101
        out = apply_sequential_perturb(
            ex,
            [("delete", [kd_tok]), ("overexpress", [oe_tok])],
        )
        self.assertEqual(out["input_ids"][0], oe_tok)
        self.assertNotIn(kd_tok, out["input_ids"])
        self.assertEqual(len(out["input_ids"]), len(ex["input_ids"]) - 1)

    def test_apply_step_aliases(self):
        from sequential_oe import apply_step, normalize_step_type

        self.assertEqual(normalize_step_type("OE"), "overexpress")
        self.assertEqual(normalize_step_type("kd"), "delete")
        ex = _fake_example([1, 2, 3])
        out = apply_step(ex, [2], "KD")
        self.assertEqual(out["input_ids"], [1, 3])


class TestParseSequentialSteps(unittest.TestCase):
    def test_parse_steps_yaml(self):
        from sequential_oe import parse_sequential_steps

        steps = parse_sequential_steps(
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
        from sequential_oe import parse_sequential_steps

        self.assertEqual(parse_sequential_steps({}), [])
        self.assertEqual(parse_sequential_steps({"steps": []}), [])


if __name__ == "__main__":
    unittest.main()
