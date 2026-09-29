"""Tests for multi-step State-feedback stability metrics (pure Python + numpy)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))

from state_feedback.stability import (  # noqa: E402
    ChainTracker,
    coefficient_of_variation,
    cross_seed_agreement,
    pair_metrics,
    stability_verdict,
)

try:
    import datasets  # noqa: F401
    import torch  # noqa: F401

    HAS_ML = True
except Exception:  # pragma: no cover - host python
    HAS_ML = False


class TestPairMetrics(unittest.TestCase):
    def test_identity(self):
        ids = list(range(100, 120))
        m = pair_metrics(ids, ids, topk=(5,))
        self.assertAlmostEqual(m["spearman"], 1.0)
        self.assertEqual(m["disp_max"], 0.0)
        self.assertEqual(m["frac_moved"], 0.0)
        self.assertEqual(m["jaccard_5"], 1.0)
        self.assertEqual(m["n_entered"], 0.0)

    def test_reverse(self):
        ids = list(range(100, 120))
        m = pair_metrics(ids, ids[::-1], topk=(5,))
        self.assertAlmostEqual(m["spearman"], -1.0)
        self.assertEqual(m["disp_max"], 19.0)
        self.assertEqual(m["jaccard_5"], 0.0)

    def test_overexpression_uses_shared_tokens(self):
        before = [1, 2, 3, 4, 5]
        after = [9, 1, 2, 3, 4]  # 9 inserted at front, 5 dropped
        m = pair_metrics(before, after, topk=(2,))
        self.assertEqual(m["n_shared"], 4.0)
        self.assertEqual(m["n_entered"], 1.0)
        self.assertAlmostEqual(m["spearman"], 1.0)
        self.assertEqual(m["disp_median"], 1.0)
        self.assertAlmostEqual(m["jaccard_2"], 1 / 3)

    def test_single_swap(self):
        m = pair_metrics([1, 2, 3, 4], [2, 1, 3, 4], topk=(1,))
        self.assertAlmostEqual(m["spearman"], 0.8)
        self.assertEqual(m["frac_moved"], 0.5)


class TestChainTracker(unittest.TestCase):
    def test_event_and_reference_rows(self):
        start = [[1, 2, 3, 4], [5, 6, 7, 8]]
        ref = {1: [[9, 1, 2, 3], [9, 5, 6, 7]]}
        tr = ChainTracker("c", start, reference=ref, topk=(2,))
        tr.observe("step", 1, ref[1])
        row = tr.observe("feedback", 1, [[9, 2, 1, 3], [9, 5, 6, 7]], keep=True)
        self.assertEqual(row["label"], "feedback1")
        self.assertEqual(row["event_index"], 2)
        # cell 0 swaps one pair (rho 0.8), cell 1 is unchanged (rho 1.0)
        self.assertAlmostEqual(row["event_spearman_median"], 0.9)
        self.assertIn("vs_ref_spearman_median", row)
        idle = tr.observe("idle", 1, [[9, 2, 1, 3], [9, 5, 6, 7]], ref_step=1)
        self.assertAlmostEqual(idle["event_spearman_median"], 1.0)
        self.assertEqual(idle["event_disp_max_median"], 0.0)
        self.assertIn("feedback1", tr.snapshots)
        self.assertEqual(tr.last[0].tolist(), [9, 2, 1, 3])

    def test_no_reference_row_without_reference(self):
        tr = ChainTracker("c", [[1, 2, 3]], topk=(2,))
        row = tr.observe("step", 1, [[1, 2, 3]])
        self.assertNotIn("vs_ref_spearman_median", row)


class TestCrossSeed(unittest.TestCase):
    def test_pairs_and_cv(self):
        finals = {0: [[1, 2, 3, 4]], 1: [[1, 2, 3, 4]], 2: [[4, 3, 2, 1]]}
        rows = cross_seed_agreement(finals, topk=(2,))
        self.assertEqual([(r["seed_a"], r["seed_b"]) for r in rows], [(0, 1), (0, 2), (1, 2)])
        self.assertAlmostEqual(rows[0]["spearman_median"], 1.0)
        self.assertAlmostEqual(rows[1]["spearman_median"], -1.0)
        self.assertAlmostEqual(coefficient_of_variation([1.0, 1.0, 1.0]), 0.0)
        self.assertTrue(coefficient_of_variation([1.0]) != coefficient_of_variation([1.0]))


def _rows(fb_rho, fb_disp, idle_rho, idle_disp):
    rows = [
        {"kind": "feedback", "event_spearman_median": r, "event_disp_median_median": d}
        for r, d in zip(fb_rho, fb_disp)
    ]
    rows += [
        {"kind": "idle", "event_spearman_median": r, "event_disp_median_median": d}
        for r, d in zip(idle_rho, idle_disp)
    ]
    return rows


class TestVerdict(unittest.TestCase):
    AGREE = [{"spearman_median": 0.99}]

    def test_stable_chain_passes(self):
        v = stability_verdict(
            {0: _rows([0.95, 0.95, 0.94], [40, 42, 45], [0.995, 0.998], [8, 3])},
            {0: {"configured": 0.05, "random0": 0.01, "random1": 0.02}},
            {0: 0.10, 1: 0.101},
            self.AGREE,
        )
        self.assertEqual(v["passed"], {"S1": True, "S2": True, "S3": True, "S4": True})
        self.assertTrue(v["multi_step_supported"])

    def test_drift_and_teacher_pull_fail(self):
        v = stability_verdict(
            {0: _rows([0.95, 0.90], [40, 80], [0.95, 0.94], [30, 35])},
            {0: {"configured": 0.05, "random0": 0.06}},
            {0: 0.10, 1: 0.20},
            [{"spearman_median": 0.80}],
        )
        self.assertEqual(v["passed"], {"S1": False, "S2": False, "S3": False, "S4": False})
        self.assertFalse(v["multi_step_supported"])

    def test_s1_fails_when_first_event_is_still(self):
        v = stability_verdict(
            {0: _rows([1.0, 0.99], [0, 30], [0.999], [0])},
            {0: {"configured": 0.05, "random0": 0.0}},
            {0: 0.1, 1: 0.1},
            self.AGREE,
        )
        self.assertFalse(v["passed"]["S1"])


@unittest.skipUnless(HAS_ML, "torch / datasets not available")
class TestAggregate(unittest.TestCase):
    """Seed jobs written to separate run directories combine into one verdict."""

    def _write_run(self, root: Path, seed: int, final: list[list[int]], end: float) -> Path:
        import json
        import pickle

        import pandas as pd

        run = root / f"run{seed}"
        (run / f"seed{seed}").mkdir(parents=True)
        (run / "run_manifest.json").write_text(json.dumps({"topk": [2]}))
        with open(run / f"seed{seed}" / "final_configured.pkl", "wb") as fh:
            pickle.dump(final, fh)
        ev = [
            {"seed": seed, "mode": "multi_step", "chain": "multi_configured",
             "event_index": i, "kind": k, "event_spearman_median": r,
             "event_disp_median_median": d}
            for i, (k, r, d) in enumerate(
                [("step", 0.99, 1), ("feedback", 0.95, 40), ("feedback", 0.95, 41),
                 ("idle", 0.995, 5), ("idle", 0.999, 2)], start=1)
        ]
        pd.DataFrame(ev).to_csv(run / "events.csv", index=False)
        pd.DataFrame([
            {"seed": None, "chain": "configured", "mode": "ordered_rank_edit",
             "endpoint_median": 0.01},
            {"seed": seed, "chain": "configured", "mode": "multi_step",
             "endpoint_median": end, "gain_over_ordered_rank_edit": end - 0.01},
            {"seed": seed, "chain": "random0", "mode": "multi_step",
             "endpoint_median": 0.02, "gain_over_ordered_rank_edit": 0.01},
        ]).to_csv(run / "endpoints.csv", index=False)
        return run

    def test_two_seed_runs(self):
        import tempfile

        from run_state_feedback_stability import aggregate

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = list(range(1, 21))
            a = self._write_run(root, 0, [ids], 0.25)
            b = self._write_run(root, 1, [ids[:-2] + ids[:-3:-1]], 0.26)
            verdict = aggregate([a, b], root / "agg")
            self.assertEqual(verdict["seeds"], [0, 1])
            self.assertEqual(verdict["passed"],
                             {"S1": True, "S2": True, "S3": True, "S4": True})
            self.assertTrue((root / "agg" / "seed_agreement.csv").exists())
            with self.assertRaises(ValueError):
                aggregate([a, a], root / "dup")


@unittest.skipUnless(HAS_ML, "torch / datasets not available")
class TestRunConditionHook(unittest.TestCase):
    def test_hook_sees_every_encoding(self):
        import tempfile

        import pandas as pd
        from datasets import Dataset

        import run_state_feedback_isp as rsf

        ids = [list(range(10, 20)), list(range(30, 40))]
        start = Dataset.from_dict({"input_ids": ids, "length": [len(r) for r in ids]})
        steps = [{"name": f"s{i}", "type": "overexpress"} for i in (1, 2)]
        tokens = [[12], [11]]
        seen: list[tuple[str, int, list[int]]] = []

        def hook(kind, step, ds):
            seen.append((kind, step, list(map(int, ds["input_ids"][0]))))

        def reverse(ctrl_ds, pert_ds):
            return [list(reversed(list(map(int, r)))) for r in pert_ds["input_ids"]], {}

        saved = rsf._score_cell_mean
        rsf._score_cell_mean = lambda m, s, w, *a, **k: pd.DataFrame(
            {"Shift_to_goal_end": [0.1] * len(w)}
        )
        try:
            with tempfile.TemporaryDirectory() as tmp:
                rsf.run_condition(
                    "t", None, start, steps, tokens, "goal", {}, Path(tmp),
                    layer_to_quant=0, pad_token_id=0, model_input_size=4096,
                    forward_batch_size=2, nproc=1, rerank_fn=reverse,
                    feedback_every_step=True, on_encoding=hook,
                )
        finally:
            rsf._score_cell_mean = saved
        self.assertEqual([(k, s) for k, s, _ in seen],
                         [("step", 1), ("feedback", 1), ("step", 2), ("feedback", 2)])
        self.assertEqual(seen[0][2][0], 12)
        self.assertEqual(seen[1][2][0], 12)  # pinned through the rerank
        self.assertEqual(seen[3][2][:2], [11, 12])


if __name__ == "__main__":
    unittest.main()
