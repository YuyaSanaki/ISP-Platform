"""Tests for the State-feedback policy: feedback after every step, no cap.

The runner-level tests need torch / datasets (Docker); the config check is pure Python.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))

from state_feedback.multistep import check_feedback_config  # noqa: E402

try:
    import datasets  # noqa: F401
    import torch  # noqa: F401

    HAS_ML = True
except Exception:  # pragma: no cover - host python
    HAS_ML = False


class TestCheckFeedbackConfig(unittest.TestCase):
    def test_clean_config_passes_silently(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_feedback_config({"pin_overexpressed": True}, 4)

    def test_other_schedules_are_rejected(self):
        for key, value in (
            ("feedback_every_step", False),
            ("feedback_after_step", 2),
            ("feedback_after_last_step", False),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_feedback_config({key: value}, 4)

    def test_every_step_values_are_ignored_with_warning(self):
        cfg = {"feedback_every_step": True, "feedback_after_step": 1,
               "feedback_after_last_step": True}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_feedback_config(cfg, 4)
        self.assertEqual(len(caught), 3)

    def test_cap_below_step_count_is_rejected(self):
        with self.assertRaises(ValueError):
            check_feedback_config({"multi_step": {"max_feedback_events": 3}}, 4)

    def test_cap_at_or_above_step_count_is_ignored_with_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_feedback_config({"multi_step": {"max_feedback_events": 5}}, 4)
        self.assertEqual(len(caught), 1)
        self.assertIn("equals the number of steps", str(caught[0].message))


@unittest.skipUnless(HAS_ML, "torch / datasets not available")
class TestRunConditionFeedback(unittest.TestCase):
    """``run_condition`` with forwards mocked: one feedback event per step."""

    def _run(self, rerank_fn, *, n_steps=4, pin=True):
        import pandas as pd
        from datasets import Dataset

        import run_state_feedback_isp as rsf

        ids = [list(range(10, 20)), list(range(30, 40))]
        start = Dataset.from_dict({"input_ids": ids, "length": [len(r) for r in ids]})
        steps = [{"name": f"s{i}", "type": "overexpress"} for i in range(1, n_steps + 1)]
        tokens = [[11] if i % 2 else [12] for i in range(n_steps)]

        def fake_score(model, start_ds, working, *a, **k):
            return pd.DataFrame({"Shift_to_goal_end": [0.1] * len(working)})

        saved = rsf._score_cell_mean
        rsf._score_cell_mean = fake_score
        try:
            with tempfile.TemporaryDirectory() as tmp:
                rows = rsf.run_condition(
                    "test", None, start, steps, tokens, "goal", {}, Path(tmp),
                    layer_to_quant=0, pad_token_id=0, model_input_size=4096,
                    forward_batch_size=2, nproc=1, rerank_fn=rerank_fn,
                    pin_overexpressed=pin,
                )
        finally:
            rsf._score_cell_mean = saved
        return rows

    @staticmethod
    def _identity_rerank(ctrl_ds, pert_ds):
        return [list(map(int, r)) for r in pert_ds["input_ids"]], {}

    def test_feedback_count_equals_step_count(self):
        for n in (2, 4, 7):
            with self.subTest(n_steps=n):
                rows = self._run(self._identity_rerank, n_steps=n)
                fb_rows = [r for r in rows if r["step_name"] == "feedback"]
                self.assertEqual([r["step"] for r in fb_rows], list(range(1, n + 1)))
                self.assertEqual([r["feedback_event"] for r in fb_rows], list(range(1, n + 1)))

    def test_without_rerank_there_is_no_feedback(self):
        rows = self._run(None)
        self.assertFalse([r for r in rows if r["step_name"] == "feedback"])
        self.assertEqual(len(rows), 4)

    def _capture_reverse(self, *, pin):
        seen: list[list[list[int]]] = []

        def rerank(ctrl_ds, pert_ds):
            before = [list(map(int, r)) for r in pert_ds["input_ids"]]
            seen.append(before)
            return [list(reversed(r)) for r in before], {}

        self._run(rerank, n_steps=2, pin=pin)
        return seen

    # _run overexpresses 12 at step 1 and 11 at step 2
    def test_overexpressed_genes_stay_pinned_at_front(self):
        seen = self._capture_reverse(pin=True)
        after_step1 = seen[0][0]
        self.assertEqual(after_step1[0], 12)
        rest = [t for t in reversed(after_step1) if t not in (11, 12)]
        # step 2 overexpresses 11 on the pinned encoding [12, *reversed rest]
        self.assertEqual(seen[1][0], [11, 12] + rest)

    def test_without_pin_rerank_moves_overexpressed_genes(self):
        seen = self._capture_reverse(pin=False)
        after_step1 = seen[0][0]
        reversed_ids = list(reversed(after_step1))
        self.assertEqual(seen[1][0], [11] + [t for t in reversed_ids if t != 11])
        self.assertEqual(seen[1][0][-1], 12)


if __name__ == "__main__":
    unittest.main()
