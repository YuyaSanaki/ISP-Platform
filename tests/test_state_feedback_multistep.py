"""Tests for multi-step State-feedback guardrails (cycle halt, convergence stop, cap).

The runner-level test needs torch / datasets (Docker); guard unit tests are pure Python.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))

from state_feedback.multistep import (  # noqa: E402
    HALT_CAP,
    HALT_CONVERGED,
    HALT_CYCLE,
    FeedbackGuard,
    MultiStepConfig,
)

try:
    import datasets  # noqa: F401
    import torch  # noqa: F401

    HAS_ML = True
except Exception:  # pragma: no cover - host python
    HAS_ML = False

A = [1, 2, 3, 4, 5, 6]
B = [2, 1, 3, 4, 5, 6]
REVERSED = list(reversed(A))


class TestMultiStepConfig(unittest.TestCase):
    def test_defaults_match_guardrail_spec(self):
        cfg = MultiStepConfig.from_mapping(None)
        self.assertEqual(cfg.max_feedback_events, 5)
        self.assertEqual(cfg.converge_spearman, 0.995)
        self.assertEqual(cfg.converge_topk, 1000)
        self.assertEqual(cfg.converge_jaccard, 0.99)
        self.assertEqual(cfg.converge_patience, 2)
        self.assertTrue(cfg.halt_on_cycle)

    def test_rejects_unknown_and_invalid(self):
        with self.assertRaises(ValueError):
            MultiStepConfig.from_mapping({"max_events": 3})
        with self.assertRaises(ValueError):
            MultiStepConfig.from_mapping({"max_feedback_events": 0})
        with self.assertRaises(ValueError):
            MultiStepConfig.from_mapping({"converge_spearman": 1.5})
        with self.assertRaises(ValueError):
            MultiStepConfig.from_mapping({"converge_patience": 0})


class TestFeedbackGuard(unittest.TestCase):
    def _cfg(self, **kw):
        base = {"converge_jaccard": 1.0, "converge_topk": 3}
        base.update(kw)
        return MultiStepConfig.from_mapping(base)

    def test_convergence_after_patience(self):
        guard = FeedbackGuard(1, self._cfg(converge_patience=2))
        out, stats = guard.apply(1, [A], [A])  # no move: small event 1
        self.assertEqual(out, [A])
        self.assertIsNone(guard.halt_reason[0])
        out, stats = guard.apply(2, [A], [A])  # small event 2 -> converged
        self.assertEqual(guard.halt_reason[0], HALT_CONVERGED)
        self.assertEqual(guard.halt_step[0], 2)
        self.assertEqual(stats["guard_new_converged"], 1.0)
        self.assertTrue(guard.exhausted)

    def test_large_move_resets_streak(self):
        guard = FeedbackGuard(1, self._cfg(converge_patience=2))
        guard.apply(1, [A], [A])
        guard.apply(2, [A], [REVERSED])
        guard.apply(3, [REVERSED], [REVERSED])
        self.assertIsNone(guard.halt_reason[0])

    def test_jaccard_criterion(self):
        guard = FeedbackGuard(1, self._cfg(converge_spearman=1.0, converge_jaccard=0.99, converge_patience=1))
        guard.apply(1, [A], [B])  # top-3 set {1,2,3} unchanged -> small by Jaccard
        self.assertEqual(guard.halt_reason[0], HALT_CONVERGED)

    def test_jaccard_ignored_when_cell_has_at_most_k_genes(self):
        guard = FeedbackGuard(
            1, self._cfg(converge_topk=6, converge_jaccard=0.99, converge_patience=1)
        )
        guard.apply(1, [A], [REVERSED])  # top-6 is the whole fixed gene set
        self.assertIsNone(guard.halt_reason[0])

    def test_two_cycle_is_rejected_and_halts(self):
        guard = FeedbackGuard(1, self._cfg())
        guard.apply(1, [A], [REVERSED])
        guard.apply(2, [REVERSED], [B])
        pre = [6, 5, 4, 3, 1, 2]
        out, stats = guard.apply(3, [pre], [REVERSED])  # back to post of two events ago
        self.assertEqual(out, [pre])
        self.assertEqual(guard.halt_reason[0], HALT_CYCLE)
        self.assertEqual(guard.halt_step[0], 3)
        self.assertEqual(stats["guard_new_cycle_halts"], 1.0)
        self.assertEqual(guard.n_applied[0], 2)

    def test_cycle_check_can_be_disabled(self):
        guard = FeedbackGuard(1, self._cfg(halt_on_cycle=False))
        guard.apply(1, [A], [REVERSED])
        guard.apply(2, [REVERSED], [B])
        out, _ = guard.apply(3, [B], [REVERSED])
        self.assertEqual(out, [REVERSED])
        self.assertIsNone(guard.halt_reason[0])

    def test_halted_cells_pass_through_and_others_continue(self):
        guard = FeedbackGuard(2, self._cfg(converge_patience=1))
        out, _ = guard.apply(1, [A, A], [A, REVERSED])
        self.assertEqual(guard.halt_reason, [HALT_CONVERGED, None])
        out, stats = guard.apply(2, [A, REVERSED], [REVERSED, A])
        self.assertEqual(out[0], A)  # halted: proposal ignored
        self.assertEqual(out[1], A)
        self.assertEqual(stats["guard_active_before"], 1.0)

    def test_cap_and_summary(self):
        guard = FeedbackGuard(2, self._cfg(max_feedback_events=1))
        guard.apply(1, [A, A], [REVERSED, REVERSED])
        self.assertTrue(guard.cap_reached)
        self.assertEqual(guard.mark_cap(2), 2)
        self.assertEqual(guard.halt_reason, [HALT_CAP, HALT_CAP])
        self.assertEqual(
            guard.summary(),
            {
                "feedback_events": 1,
                "n_cells": 2,
                "halted_cycle": 0,
                "halted_converged": 0,
                "halted_cap": 2,
                "ran_to_end": 0,
            },
        )
        rows = guard.cell_rows()
        self.assertEqual(rows[0], {"cell": 0, "halt_reason": HALT_CAP, "halt_step": 2, "n_feedback_applied": 1})

    def test_rejects_wrong_cell_count(self):
        guard = FeedbackGuard(2, self._cfg())
        with self.assertRaises(ValueError):
            guard.apply(1, [A], [A])


@unittest.skipUnless(HAS_ML, "torch / datasets not available")
class TestRunConditionGuard(unittest.TestCase):
    """``run_condition`` with forwards mocked: guard wiring, halts and outputs."""

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
                guard_csv = Path(tmp) / "feedback_guard.csv"
                guard_rows = pd.read_csv(guard_csv) if guard_csv.exists() else None
        finally:
            rsf._score_cell_mean = saved
        return rows, summaries, guard_rows

    @staticmethod
    def _reverse_rerank(ctrl_ds, pert_ds):
        before = [list(map(int, r)) for r in pert_ds["input_ids"]]
        return [list(reversed(r)) for r in before], {}

    @staticmethod
    def _identity_rerank(ctrl_ds, pert_ds):
        before = [list(map(int, r)) for r in pert_ds["input_ids"]]
        return before, {}

    def test_single_feedback_has_no_guard(self):
        rows, summaries, guard_rows = self._run(self._reverse_rerank, every_step=False)
        fb_rows = [r for r in rows if r["step_name"] == "feedback"]
        self.assertEqual(len(fb_rows), 1)
        self.assertNotIn("guard_event", fb_rows[0])
        self.assertEqual(summaries, {})
        self.assertIsNone(guard_rows)

    def test_identity_rerank_converges(self):
        rows, summaries, guard_rows = self._run(self._identity_rerank, every_step=True)
        fb_rows = [r for r in rows if r["step_name"] == "feedback"]
        self.assertEqual(len(fb_rows), 2)  # patience 2, then every cell stopped
        self.assertEqual(summaries["test"]["halted_converged"], 2)
        self.assertEqual(list(guard_rows["halt_reason"]), [HALT_CONVERGED, HALT_CONVERGED])
        self.assertEqual(list(guard_rows["halt_step"]), [2, 2])

    def test_cap_limits_feedback_events(self):
        cfg = MultiStepConfig.from_mapping({"max_feedback_events": 2, "halt_on_cycle": False})
        rows, summaries, guard_rows = self._run(
            self._reverse_rerank, every_step=True, n_steps=5, multi_step=cfg
        )
        fb_rows = [r for r in rows if r["step_name"] == "feedback"]
        self.assertEqual([r["step"] for r in fb_rows], [1, 2])
        self.assertEqual(summaries["test"]["halted_cap"], 2)
        self.assertEqual(list(guard_rows["halt_step"]), [3, 3])
        # guard columns sit on feedback rows only, so the endpoint gate still reads step rows
        step_rows = [r for r in rows if r["step_name"].startswith("s")]
        self.assertEqual(len(step_rows), 5)
        self.assertTrue(all("guard_event" not in r for r in step_rows))


if __name__ == "__main__":
    unittest.main()
