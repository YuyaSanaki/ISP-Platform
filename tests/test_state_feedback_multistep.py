"""Tests for the multi-step State-feedback event cap.

The runner-level test needs torch / datasets (Docker); config and guard unit tests are
pure Python.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))

from state_feedback.multistep import FeedbackGuard, MultiStepConfig  # noqa: E402

try:
    import datasets  # noqa: F401
    import torch  # noqa: F401

    HAS_ML = True
except Exception:  # pragma: no cover - host python
    HAS_ML = False


class TestMultiStepConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = MultiStepConfig.from_mapping(None)
        self.assertEqual(cfg.to_dict(), {"max_feedback_events": 5})

    def test_rejects_unknown_and_invalid(self):
        with self.assertRaises(ValueError):
            MultiStepConfig.from_mapping({"max_events": 3})
        with self.assertRaises(ValueError):
            MultiStepConfig.from_mapping({"max_feedback_events": 0})

    def test_removed_stop_keys_are_ignored_with_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = MultiStepConfig.from_mapping(
                {
                    "max_feedback_events": 3,
                    "converge_spearman": 0.995,
                    "converge_patience": 2,
                    "halt_on_cycle": True,
                }
            )
        self.assertEqual(cfg.to_dict(), {"max_feedback_events": 3})
        self.assertEqual(len(caught), 1)
        self.assertIn("halt_on_cycle", str(caught[0].message))


class TestFeedbackGuard(unittest.TestCase):
    def test_cap_and_summary(self):
        guard = FeedbackGuard(MultiStepConfig(max_feedback_events=2))
        self.assertEqual(guard.record_event(), {"guard_event": 1.0})
        self.assertFalse(guard.cap_reached)
        guard.record_event()
        self.assertTrue(guard.cap_reached)
        self.assertTrue(guard.mark_cap(3))
        self.assertFalse(guard.mark_cap(4))
        self.assertEqual(
            guard.summary(),
            {"feedback_events": 2, "max_feedback_events": 2, "capped_before_step": 3},
        )


@unittest.skipUnless(HAS_ML, "torch / datasets not available")
class TestRunConditionGuard(unittest.TestCase):
    """``run_condition`` with forwards mocked: cap wiring and outputs."""

    def _run(self, rerank_fn, *, every_step, n_steps=4, multi_step=None):
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
        summaries: dict = {}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                rows = rsf.run_condition(
                    "test",
                    None,
                    start,
                    steps,
                    tokens,
                    "goal",
                    {},
                    Path(tmp),
                    layer_to_quant=0,
                    pad_token_id=0,
                    model_input_size=4096,
                    forward_batch_size=2,
                    nproc=1,
                    rerank_fn=rerank_fn,
                    feedback_after_step=1,
                    feedback_every_step=every_step,
                    multi_step=multi_step,
                    guard_summaries=summaries,
                )
        finally:
            rsf._score_cell_mean = saved
        return rows, summaries

    @staticmethod
    def _reverse_rerank(ctrl_ds, pert_ds):
        before = [list(map(int, r)) for r in pert_ds["input_ids"]]
        return [list(reversed(r)) for r in before], {}

    @staticmethod
    def _identity_rerank(ctrl_ds, pert_ds):
        before = [list(map(int, r)) for r in pert_ds["input_ids"]]
        return before, {}

    def test_single_feedback_has_no_guard(self):
        rows, summaries = self._run(self._reverse_rerank, every_step=False)
        fb_rows = [r for r in rows if r["step_name"] == "feedback"]
        self.assertEqual(len(fb_rows), 1)
        self.assertNotIn("guard_event", fb_rows[0])
        self.assertEqual(summaries, {})

    def test_identity_rerank_runs_every_event(self):
        rows, summaries = self._run(self._identity_rerank, every_step=True)
        fb_rows = [r for r in rows if r["step_name"] == "feedback"]
        self.assertEqual([r["step"] for r in fb_rows], [1, 2, 3])
        self.assertEqual(summaries["test"]["feedback_events"], 3)
        self.assertIsNone(summaries["test"]["capped_before_step"])

    def test_cap_limits_feedback_events(self):
        cfg = MultiStepConfig(max_feedback_events=2)
        rows, summaries = self._run(
            self._reverse_rerank, every_step=True, n_steps=5, multi_step=cfg
        )
        fb_rows = [r for r in rows if r["step_name"] == "feedback"]
        self.assertEqual([r["step"] for r in fb_rows], [1, 2])
        self.assertEqual([r["guard_event"] for r in fb_rows], [1.0, 2.0])
        self.assertEqual(summaries["test"]["capped_before_step"], 3)
        # guard columns sit on feedback rows only, so the endpoint gate still reads step rows
        step_rows = [r for r in rows if r["step_name"].startswith("s")]
        self.assertEqual(len(step_rows), 5)
        self.assertTrue(all("guard_event" not in r for r in step_rows))


if __name__ == "__main__":
    unittest.main()
