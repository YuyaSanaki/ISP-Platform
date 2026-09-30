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
    running_random_mean,
    specific_gain,
    stability_verdict,
)
from state_feedback import random_chains  # noqa: E402

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


class TestSpecificGain(unittest.TestCase):
    def test_configured_minus_mean_random(self):
        r = specific_gain(
            [0.30, 0.32], [0.01, 0.01],
            [[0.20, 0.22], [0.24, 0.26]], [[0.0, 0.0], [0.02, 0.02]],
            n_boot=200,
        )
        # configured gain 0.29/0.31, random gain mean 0.21/0.23 -> specific 0.08
        self.assertAlmostEqual(r["specific_gain_mean"], 0.08)
        self.assertAlmostEqual(r["configured_gain_mean"], 0.30)
        self.assertAlmostEqual(r["specific_fraction"], 0.08 / 0.30)
        self.assertGreater(r["specific_gain_ci_low"], 0.0)
        self.assertEqual(r["frac_cells_positive"], 1.0)

    def test_needs_random(self):
        with self.assertRaises(ValueError):
            specific_gain([0.1], [0.0], [], [])

    def test_chain_spread_widens_ci(self):
        conf, ref = [0.3] * 50, [0.0] * 50
        tight = [[0.2] * 50 for _ in range(10)]
        spread = [[0.2 + 0.05 * ((-1) ** r)] * 50 for r in range(10)]
        refs = [[0.0] * 50] * 10
        a = specific_gain(conf, ref, tight, refs, n_boot=500)
        b = specific_gain(conf, ref, spread, refs, n_boot=500)
        self.assertAlmostEqual(a["specific_gain_mean"], b["specific_gain_mean"])
        self.assertEqual(a["specific_gain_ci_high"] - a["specific_gain_ci_low"], 0.0)
        self.assertGreater(b["specific_gain_ci_high"] - b["specific_gain_ci_low"], 0.01)
        self.assertAlmostEqual(b["random_gain_sd_chains"], 0.05 * (10 / 9) ** 0.5)

    def test_null_places_configured_among_random_chains(self):
        randoms = [[g] * 5 for g in (0.10, 0.12, 0.14, 0.16)]
        refs = [[0.0] * 5] * 4
        inside = specific_gain([0.13] * 5, [0.0] * 5, randoms, refs, n_boot=50)
        outside = specific_gain([0.50] * 5, [0.0] * 5, randoms, refs, n_boot=50)
        self.assertGreater(inside["null_empirical_p"], 0.2)
        self.assertAlmostEqual(outside["null_empirical_p"], 1 / 5)
        self.assertNotIn("null_sd", specific_gain([0.1], [0.0], [[0.0]] * 2, [[0.0]] * 2))

    def test_running_mean(self):
        rows = running_random_mean([[0.1, 0.1], [0.3, 0.3], [0.2, 0.2]], [[0.0, 0.0]] * 3)
        self.assertEqual([r["n_chains"] for r in rows], [1, 2, 3])
        self.assertAlmostEqual(rows[1]["running_mean"], 0.2)
        self.assertAlmostEqual(rows[2]["running_mean"], 0.2)
        self.assertAlmostEqual(rows[2]["running_se"], 0.1 / 3 ** 0.5)


class TestRandomChains(unittest.TestCase):
    def _profiles(self):
        # configured: 1 (absent), 2 (rare, near the bottom). Population: 3-19 absent,
        # 20-29 rare near the bottom, 30-39 common near the top.
        det = {**{t: 0.05 for t in range(20, 30)}, **{t: 0.6 for t in range(30, 40)}, 2: 0.04}
        rank = {**{t: 0.9 for t in range(20, 30)}, **{t: 0.2 for t in range(30, 40)}, 2: 0.88}
        return list(range(1, 40)), det, rank

    def test_strata_match_start_position(self):
        pop, det, rank = self._profiles()
        absent, info = random_chains.stratum(1, pop, det, rank, min_stratum=5)
        self.assertEqual(info["rule"], "absent")
        self.assertTrue(set(range(10, 20)) <= set(absent))
        self.assertFalse(set(absent) & set(range(20, 40)))
        rare, info = random_chains.stratum(2, pop, det, rank, min_stratum=5)
        self.assertEqual(sorted(set(rare) - {2}), list(range(20, 30)))
        self.assertEqual(info["rank_window"], 0.1)

    def test_draw_keeps_structure_and_is_order_invariant(self):
        pop, det, rank = self._profiles()
        steps = [{"type": "overexpress"}, {"type": "knockdown"}]
        a, rec = random_chains.draw_random_chains(
            steps, [[1, 2], [2]], pop, det, rank, n_chains=4, seed=3, min_stratum=5)
        b, _ = random_chains.draw_random_chains(
            steps[::-1], [[2], [2, 1]], pop, det, rank, n_chains=4, seed=3, min_stratum=5)
        self.assertEqual(len(a), 4)
        for name, (sl, tbs) in a.items():
            self.assertEqual([len(t) for t in tbs], [2, 1])
            self.assertEqual([s["type"] for s in sl], ["overexpress", "knockdown"])
            self.assertEqual(tbs[0][1], tbs[1][0])  # same configured gene -> same random gene
            self.assertEqual(sorted(b[name][1][1]), sorted(tbs[0]))
        self.assertEqual(len(set(rec["picks"][1])), 4)  # without replacement
        self.assertFalse(set(rec["picks"][1]) & {1, 2})

    def test_describe_reports_balance(self):
        pop, det, rank = self._profiles()
        _, rec = random_chains.draw_random_chains(
            [{"type": "overexpress"}], [[2]], pop, det, rank, n_chains=5, seed=0, min_stratum=5)
        d = random_chains.describe(rec, {"goal": {t: float(t) for t in pop}}, {2: "KLF4"})
        self.assertEqual(d["n_picks"], 5)
        self.assertEqual(d["balance"][0]["configured_gene"], "KLF4")
        self.assertIn("goal_smd", d["balance"][0])


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

    @staticmethod
    def _cells(path: Path, values):
        import pandas as pd

        path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"cell_index": [1, 0], "Shift_to_goal_end": values[::-1]}).to_csv(
            path / "per_cell_shifts.csv", index=False
        )

    def test_specific_gain_per_random_set_from_per_cell_files(self):
        import tempfile

        import pandas as pd

        from run_state_feedback_stability import aggregate

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._write_run(root, 0, [list(range(1, 21))], 0.30)
            random_end = {"random_s0_0": 0.26, "random_s0_1": 0.24, "random_s0_10": 0.28,
                          "random_s1_0": 0.22}
            for chain in ("configured", *random_end):
                self._cells(run / "ordered_rank_edit" / chain / "step02_x", [0.01, 0.01])
                self._cells(run / "ordered_rank_edit" / chain / "step01_x", [9.0, 9.0])
            s = run / "seed0"
            self._cells(s / "multi_configured" / "step02_x", [9.0, 9.0])
            self._cells(s / "multi_configured" / "step02_feedback", [0.31, 0.31])
            self._cells(s / "multi_configured" / "step01_feedback", [9.0, 9.0])
            for chain, end in random_end.items():
                self._cells(s / f"multi_{chain}" / "step02_feedback", [end, end])
            aggregate([run], run)
            spec = pd.read_csv(run / "specificity.csv").set_index("set")
            self.assertEqual(list(spec.index), ["all", "s0", "s1"])
            self.assertAlmostEqual(spec.loc["all", "specific_gain_mean"], 0.30 - 0.24)
            self.assertAlmostEqual(spec.loc["s0", "specific_gain_mean"], 0.30 - 0.25)
            self.assertEqual(spec.loc["all", "n_random_chains"], 4)
            running = pd.read_csv(run / "random_running_mean.csv")
            s0 = running[running["set"] == "s0"]
            # draw order 0, 1, 10 (not string order)
            self.assertEqual(list(s0["chain_gain"].round(4)), [0.25, 0.23, 0.27])


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
                    forward_batch_size=2, nproc=1, rerank_fn=reverse, on_encoding=hook,
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
