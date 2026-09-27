"""Unit tests for State-feedback ISP Phase 1-2 (baselines + Delta-rank decoder).

Pure-function tests run without torch; decoder and hidden-state tests are skipped
when torch is unavailable (host python has no ML deps; Docker has them).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))

from state_feedback.metrics import (
    bootstrap_delta_rho,
    calibration_bins,
    calibration_monotonicity,
    displacement_summary,
    gap_closed_fraction,
    is_two_cycle,
    median_iqr,
    ndcg_at_k,
    paired_sign_flip_test,
    precision_at_k,
    rank_displacement,
    spearman_values,
    topk_jaccard,
)
from state_feedback.rerank import (
    apply_priority_order,
    base_rank_norm,
    bounded_priority,
    delta_mlm_priority,
    norm_priority,
    quantize,
)
from state_feedback.teacher import (
    observed_delta_rank,
    pseudobulk_positions,
    split_tokens,
)

try:
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover - host python
    HAS_TORCH = False


class TestRerank(unittest.TestCase):
    def test_base_rank_norm_spans_unit_interval(self):
        self.assertEqual(base_rank_norm(1), [0.0])
        self.assertEqual(base_rank_norm(3), [0.0, 0.5, 1.0])

    def test_priority_convention_lower_is_leftmost(self):
        ids = [10, 20, 30]
        out = apply_priority_order(ids, [2.0, 0.0, 1.0])
        self.assertEqual(out, [20, 30, 10])

    def test_gene_set_is_preserved(self):
        ids = [7, 8, 9, 10]
        out = apply_priority_order(ids, [0.3, 0.3, 0.1, 0.9])
        self.assertEqual(sorted(out), sorted(ids))

    def test_exact_ties_keep_original_order(self):
        ids = [1, 2, 3]
        self.assertEqual(apply_priority_order(ids, [0.5, 0.5, 0.5]), ids)

    def test_hysteresis_suppresses_near_tie_swaps(self):
        ids = [1, 2]
        # 2 is marginally more favourable than 1.
        priority = [0.500, 0.499]
        self.assertEqual(apply_priority_order(ids, priority), [2, 1])
        self.assertEqual(apply_priority_order(ids, priority, hysteresis=0.01), [1, 2])

    def test_hysteresis_still_allows_real_moves(self):
        ids = [1, 2]
        self.assertEqual(apply_priority_order(ids, [0.9, 0.1], hysteresis=0.01), [2, 1])

    def test_quantize_grid(self):
        self.assertAlmostEqual(quantize(0.504, 0.01), 0.5)
        self.assertAlmostEqual(quantize(0.123, 0.0), 0.123)

    def test_bounded_priority_clamps_shift(self):
        out = bounded_priority([0.5, 0.5], [10.0, -10.0], max_shift=0.1)
        self.assertAlmostEqual(out[0], 0.6)
        self.assertAlmostEqual(out[1], 0.4)

    def test_norm_priority_shifts_in_the_right_direction(self):
        deltas = [0.0, 5.0, -5.0, 1.0]
        base = base_rank_norm(4)
        priority = norm_priority(deltas, alpha=1.0, max_shift=0.5)
        self.assertLess(priority[1], base[1])  # largest increase moves left
        self.assertGreater(priority[2], base[2])  # largest decrease moves right

    def test_norm_priority_can_reorder(self):
        ids = [1, 2, 3, 4]
        priority = norm_priority([0.0, 0.0, 0.0, 10.0], alpha=1.0, max_shift=1.0)
        self.assertEqual(apply_priority_order(ids, priority)[0], 4)

    def test_norm_priority_constant_input_is_identity(self):
        base = base_rank_norm(4)
        priority = norm_priority([1.0] * 4, alpha=1.0, max_shift=0.5)
        for got, want in zip(priority, base):
            self.assertAlmostEqual(got, want)

    def test_delta_mlm_priority_keeps_inertia(self):
        # Equal evidence -> order must follow base rank (inertia term).
        priority = delta_mlm_priority([0.0] * 5, alpha=1.0, max_shift=0.2)
        self.assertEqual(priority, sorted(priority))

    def test_delta_mlm_bounded_by_max_shift(self):
        priority = delta_mlm_priority([0.0, 100.0, -100.0], alpha=5.0, max_shift=0.05)
        base = base_rank_norm(3)
        for got, want in zip(priority, base):
            self.assertLessEqual(abs(got - want), 0.05 + 1e-9)


class TestMetrics(unittest.TestCase):
    def test_spearman_monotonic(self):
        self.assertAlmostEqual(spearman_values([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(spearman_values([1, 2, 3], [30, 20, 10]), -1.0)

    def test_spearman_handles_ties(self):
        rho = spearman_values([1.0, 1.0, 2.0], [5.0, 5.0, 9.0])
        self.assertAlmostEqual(rho, 1.0)

    def test_topk_jaccard(self):
        self.assertAlmostEqual(topk_jaccard([1, 2, 3, 4], [1, 2, 9, 9], 2), 1.0)
        self.assertAlmostEqual(topk_jaccard([1, 2], [3, 4], 2), 0.0)

    def test_rank_displacement_signs(self):
        # Aligned with `after`: token 3 went from index 2 to index 0 -> -2.
        self.assertEqual(rank_displacement([1, 2, 3], [3, 1, 2]), [-2, 1, 1])

    def test_displacement_summary_identity(self):
        summary = displacement_summary([1, 2, 3], [1, 2, 3])
        self.assertEqual(summary["mean_abs_displacement"], 0.0)
        self.assertEqual(summary["frac_moved"], 0.0)

    def test_two_cycle_detection(self):
        self.assertTrue(is_two_cycle([1, 2], [2, 1], [1, 2]))
        self.assertFalse(is_two_cycle([1, 2], [2, 1], [2, 1]))

    def test_gap_closed_fraction(self):
        self.assertAlmostEqual(gap_closed_fraction(0.0, 0.8, 1.0), 0.8)
        self.assertAlmostEqual(gap_closed_fraction(0.0, -0.2, 1.0), -0.2)
        self.assertTrue(gap_closed_fraction(1.0, 1.0, 1.0) != gap_closed_fraction(1.0, 1.0, 1.0))


class TestTeacher(unittest.TestCase):
    def test_pseudobulk_positions(self):
        pos, counts, mean_len = pseudobulk_positions([[1, 2, 3], [1, 3, 2]])
        self.assertAlmostEqual(pos[1], 0.0)
        self.assertAlmostEqual(pos[2], 1.5)
        self.assertEqual(counts[1], 2)
        self.assertAlmostEqual(mean_len, 3.0)

    def test_observed_delta_rank_sign(self):
        # Token 2 is last in ctrl and first in obs -> negative (moves left/up).
        ctrl = [[1, 2]] * 5
        obs = [[2, 1]] * 5
        delta = observed_delta_rank(ctrl, obs, min_detection_count=5)
        self.assertLess(delta[2], 0.0)
        self.assertGreater(delta[1], 0.0)

    def test_min_detection_count_filters_noisy_tokens(self):
        ctrl = [[1, 2], [1, 2], [1, 2]]
        obs = [[1, 2], [1, 2], [1, 2]]
        self.assertEqual(observed_delta_rank(ctrl, obs, min_detection_count=5), {})
        self.assertNotEqual(observed_delta_rank(ctrl, obs, min_detection_count=3), {})

    def test_token_split_is_deterministic_and_disjoint(self):
        tokens = list(range(500))
        train_a, val_a = split_tokens(tokens, val_fraction=0.2, seed=7)
        train_b, val_b = split_tokens(tokens, val_fraction=0.2, seed=7)
        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertEqual(train_a & val_a, set())
        self.assertEqual(len(train_a) + len(val_a), len(tokens))
        self.assertGreater(len(val_a), 0)


class TestDirectionFidelityMetrics(unittest.TestCase):
    def test_median_iqr(self):
        stats = median_iqr([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(stats["median"], 2.5)
        self.assertAlmostEqual(stats["q25"], 1.75)
        self.assertAlmostEqual(stats["q75"], 3.25)
        self.assertAlmostEqual(stats["iqr"], 1.5)
        self.assertEqual(stats["n"], 4.0)

    def test_median_iqr_drops_nan(self):
        self.assertEqual(median_iqr([float("nan"), 5.0])["n"], 1.0)

    def test_precision_at_k_perfect_and_worst(self):
        obs = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
        self.assertAlmostEqual(precision_at_k(obs, obs, 2, up=True), 1.0)
        self.assertAlmostEqual(precision_at_k(obs, obs, 2, up=False), 1.0)
        flipped = [-v for v in obs]
        self.assertAlmostEqual(precision_at_k(flipped, obs, 2, up=True), 0.0)

    def test_precision_at_k_direction_matters(self):
        # up = most negative; down = most positive
        obs = [-5.0, 0.0, 5.0]
        self.assertAlmostEqual(precision_at_k(obs, obs, 1, up=True), 1.0)
        self.assertAlmostEqual(precision_at_k([5.0, 0.0, -5.0], obs, 1, up=True), 0.0)

    def test_precision_at_k_nan_when_k_too_large(self):
        self.assertTrue(precision_at_k([1.0, 2.0], [1.0, 2.0], 5, up=True) != precision_at_k([1.0, 2.0], [1.0, 2.0], 5, up=True))

    def test_ndcg_perfect_is_one(self):
        obs = [-3.0, -2.0, -1.0, 0.0]
        self.assertAlmostEqual(ndcg_at_k(obs, obs, 3, up=True), 1.0)

    def test_ndcg_penalises_wrong_order(self):
        obs = [-3.0, -2.0, -1.0, 0.0]
        good = ndcg_at_k(obs, obs, 2, up=True)
        bad = ndcg_at_k([0.0, -1.0, -2.0, -3.0], obs, 2, up=True)
        self.assertGreater(good, bad)

    def test_ndcg_nan_when_no_relevance(self):
        obs = [1.0, 2.0, 3.0]  # nothing moved up
        self.assertTrue(ndcg_at_k(obs, obs, 2, up=True) != ndcg_at_k(obs, obs, 2, up=True))

    def test_calibration_bins_monotone(self):
        pred = [float(i) for i in range(100)]
        obs = [float(i) for i in range(100)]
        bins = calibration_bins(pred, obs, 10)
        self.assertEqual(len(bins), 10)
        self.assertAlmostEqual(calibration_monotonicity(bins), 1.0)
        self.assertEqual(sum(b["n"] for b in bins), 100.0)

    def test_calibration_monotonicity_inverted(self):
        pred = [float(i) for i in range(50)]
        obs = [float(-i) for i in range(50)]
        self.assertAlmostEqual(calibration_monotonicity(calibration_bins(pred, obs, 5)), -1.0)

    def test_bootstrap_delta_rho_detects_clear_winner(self):
        groups = [i // 10 for i in range(200)]
        obs = [float((i * 37) % 200) for i in range(200)]
        good = list(obs)
        bad = [float((i * 91) % 200) for i in range(200)]
        out = bootstrap_delta_rho(groups, good, bad, obs, n_boot=200, seed=1)
        self.assertGreater(out["delta_rho"], 0.5)
        self.assertEqual(out["excludes_zero"], 1.0)
        self.assertGreater(out["ci_low"], 0.0)

    def test_bootstrap_delta_rho_ties_include_zero(self):
        groups = [i // 10 for i in range(200)]
        obs = [float((i * 37) % 200) for i in range(200)]
        out = bootstrap_delta_rho(groups, list(obs), list(obs), obs, n_boot=200, seed=1)
        self.assertAlmostEqual(out["delta_rho"], 0.0)
        self.assertEqual(out["excludes_zero"], 0.0)

    def test_paired_sign_flip_detects_shift(self):
        out = paired_sign_flip_test([0.3] * 40, n_perm=2000, seed=0)
        self.assertAlmostEqual(out["mean_diff"], 0.3)
        self.assertEqual(out["frac_positive"], 1.0)
        self.assertLess(out["p_value"], 0.01)

    def test_paired_sign_flip_null(self):
        diffs = [0.2, -0.2] * 20
        out = paired_sign_flip_test(diffs, n_perm=2000, seed=0)
        self.assertAlmostEqual(out["mean_diff"], 0.0)
        self.assertGreater(out["p_value"], 0.5)


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestEvaluate(unittest.TestCase):
    def _samples(self, n_cells=6, n_genes=20, d=4):
        from state_feedback.samples import EvalSamples

        torch.manual_seed(0)
        n = n_cells * n_genes
        cell_index = [c for c in range(n_cells) for _ in range(n_genes)]
        tokens = [1000 + g for _ in range(n_cells) for g in range(n_genes)]
        target = torch.tensor(
            [0.01 * (g - n_genes / 2) for _ in range(n_cells) for g in range(n_genes)],
            dtype=torch.float32,
        )
        return EvalSamples(
            cell_index=cell_index,
            tokens=tokens,
            base_rank=torch.rand(n),
            target=target,
            delta_h=torch.randn(n, d),
            delta_norm=target.clone(),  # norm perfectly informative in this fixture
            delta_self_logit=torch.randn(n),
        )

    def test_identity_and_random_are_uninformative(self):
        from state_feedback.evaluate import evaluate_method, predicted_delta_rank

        s = self._samples()
        ident = predicted_delta_rank("identity", s, alpha=1.0, max_shift=0.1)
        self.assertEqual(set(ident), {0.0})
        row = evaluate_method("random", predicted_delta_rank("random", s, alpha=1.0, max_shift=0.1, seed=3), s)
        self.assertLess(abs(row["pooled_spearman"]), 0.3)

    def test_norm_sign_convention(self):
        # delta_norm == target here, and a norm *increase* must predict a move left
        # (negative displacement), so the correlation is negative by construction.
        from state_feedback.evaluate import predicted_delta_rank
        from state_feedback.metrics import spearman_values

        s = self._samples()
        pred = predicted_delta_rank("norm", s, alpha=1.0, max_shift=0.5)
        self.assertLess(spearman_values(pred, s.target.tolist()), -0.9)

    def test_bounded_by_max_shift(self):
        from state_feedback.evaluate import predicted_delta_rank

        s = self._samples()
        for method in ("random", "norm", "delta_mlm"):
            pred = predicted_delta_rank(method, s, alpha=10.0, max_shift=0.05, seed=0)
            self.assertLessEqual(max(abs(p) for p in pred), 0.05 + 1e-9)

    def test_delta_mlm_requires_features(self):
        from state_feedback.evaluate import predicted_delta_rank

        s = self._samples()
        s.delta_self_logit = None
        with self.assertRaises(ValueError):
            predicted_delta_rank("delta_mlm", s, alpha=1.0, max_shift=0.1)

    def test_linear_requires_decoder(self):
        from state_feedback.evaluate import predicted_delta_rank

        with self.assertRaises(ValueError):
            predicted_delta_rank("linear_deltarank", self._samples(), alpha=1.0, max_shift=0.1)

    def test_evaluate_method_reports_all_axes(self):
        from state_feedback.evaluate import evaluate_method

        s = self._samples()
        row = evaluate_method("oracleish", s.target.tolist(), s, topk=[5])
        self.assertAlmostEqual(row["pooled_spearman"], 1.0)
        self.assertAlmostEqual(row["cellwise_spearman_median"], 1.0)
        self.assertAlmostEqual(row["gene_aggregated_spearman"], 1.0)
        self.assertAlmostEqual(row["precision_at_5_up"], 1.0)
        self.assertAlmostEqual(row["precision_at_5_down"], 1.0)
        self.assertAlmostEqual(row["calibration_monotonicity"], 1.0)
        self.assertEqual(row["n_cells_scored"], 6.0)
        self.assertEqual(row["n_genes_scored"], 20.0)

    def test_compare_methods_skips_unavailable(self):
        from state_feedback.evaluate import compare_methods

        s = self._samples()
        s.delta_self_logit = None
        out = compare_methods(s, alpha=1.0, max_shift=0.1, decoder=None, n_boot=20, n_perm=50)
        names = {r["method"] for r in out["methods"]}
        self.assertEqual(names, {"identity", "random", "norm", "base_rank"})
        self.assertEqual(set(out["skipped"]), {"delta_mlm", "linear_deltarank"})
        self.assertEqual(out["contrasts"], [])
        self.assertNotIn("base_rank_control", out)

    def test_compare_methods_contrasts_primary(self):
        from state_feedback.decoder import DeltaRankDecoder
        from state_feedback.evaluate import compare_methods, direction_fidelity_verdict

        s = self._samples()
        dec = DeltaRankDecoder(s.delta_h.size(1), max_shift=0.1)
        out = compare_methods(s, alpha=1.0, max_shift=0.1, decoder=dec, n_boot=20, n_perm=50)
        self.assertEqual(
            {c["baseline"] for c in out["contrasts"]},
            {"norm", "delta_mlm", "base_rank", "identity", "random"},
        )
        self.assertIn("linear_partial", out["base_rank_control"])
        verdict = direction_fidelity_verdict(out)
        # untrained decoder predicts zero, so it cannot beat an informative baseline
        self.assertIn("norm", verdict["does_not_beat"])
        self.assertFalse(verdict["pass"])
        self.assertIn("adds_beyond_base_rank", verdict)

    def test_verdict_requires_signal_beyond_base_rank(self):
        from state_feedback.evaluate import direction_fidelity_verdict

        contrasts = [
            {"baseline": b, "boot_delta_rho": 0.1, "boot_ci_low": 0.05, "boot_ci_high": 0.15,
             "boot_excludes_zero": True}
            for b in ("norm", "delta_mlm")
        ]
        rows = [{"method": "linear_deltarank", "pooled_spearman": 0.4}]
        no_partial = {"methods": rows, "contrasts": contrasts,
                      "base_rank_control": {"linear_partial": {"partial": 0.01, "ci_low": -0.02,
                                                               "ci_high": 0.04}}}
        v = direction_fidelity_verdict(no_partial)
        self.assertTrue(v["beats_required_baselines"])
        self.assertFalse(v["adds_beyond_base_rank"])
        self.assertFalse(v["pass"])
        with_partial = dict(no_partial, base_rank_control={
            "linear_partial": {"partial": 0.2, "ci_low": 0.15, "ci_high": 0.25}})
        self.assertTrue(direction_fidelity_verdict(with_partial)["pass"])

    def test_base_rank_method_is_crossfitted(self):
        from state_feedback.evaluate import predicted_delta_rank
        from state_feedback.metrics import spearman_values

        s = self._samples()
        s.target = (s.base_rank - 0.5).clone()
        pred = predicted_delta_rank("base_rank", s, alpha=1.0, max_shift=1.0)
        self.assertGreater(spearman_values(pred, s.target.tolist()), 0.9)

    def test_zscore_is_within_cell(self):
        from state_feedback.evaluate import _zscore_within_cells

        # cell 0 and cell 1 have different scales; z-scores must match
        z = _zscore_within_cells([1.0, 2.0, 3.0, 100.0, 200.0, 300.0], [0, 0, 0, 1, 1, 1])
        self.assertAlmostEqual(z[0], z[3], places=6)
        self.assertAlmostEqual(z[2], z[5], places=6)


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestSampleSelection(unittest.TestCase):
    def test_selection_is_deterministic_across_passes(self):
        from state_feedback.samples import select_gene_indices

        ids = list(range(100))
        scorable = {i: 0.0 for i in range(100)}
        a = select_gene_indices(ids, scorable, max_genes=10, seed=42)
        b = select_gene_indices(ids, scorable, max_genes=10, seed=42)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 10)
        self.assertEqual(a, sorted(a))

    def test_selection_respects_scorable_and_keep(self):
        from state_feedback.samples import select_gene_indices

        ids = [5, 6, 7, 8]
        scorable = {5: 0.0, 7: 0.0, 8: 0.0}
        self.assertEqual(select_gene_indices(ids, scorable, max_genes=0, seed=0), [0, 2, 3])
        self.assertEqual(
            select_gene_indices(ids, scorable, max_genes=0, seed=0, keep_tokens={7}), [2]
        )

    def test_token_subset_keeps_all_features_aligned(self):
        from state_feedback.samples import EvalSamples

        s = EvalSamples(
            cell_index=[0, 0, 1],
            tokens=[10, 11, 10],
            base_rank=torch.tensor([0.0, 0.5, 1.0]),
            target=torch.tensor([1.0, 2.0, 3.0]),
            delta_h=torch.arange(6, dtype=torch.float32).reshape(3, 2),
            delta_norm=torch.tensor([7.0, 8.0, 9.0]),
            delta_self_logit=torch.tensor([-1.0, -2.0, -3.0]),
        )
        sub = s.token_subset({10})
        self.assertEqual(sub.tokens, [10, 10])
        self.assertEqual(sub.cell_index, [0, 1])
        self.assertEqual(sub.target.tolist(), [1.0, 3.0])
        self.assertEqual(sub.delta_norm.tolist(), [7.0, 9.0])
        self.assertEqual(sub.delta_self_logit.tolist(), [-1.0, -3.0])
        self.assertEqual(sub.delta_h.tolist(), [[0.0, 1.0], [4.0, 5.0]])


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestDecoder(unittest.TestCase):
    def _decoder(self, d_model=8, max_shift=0.1):
        from state_feedback.decoder import DeltaRankDecoder

        return DeltaRankDecoder(d_model, max_shift=max_shift)

    def test_shapes(self):
        dec = self._decoder()
        delta_h = torch.randn(17, 8)
        base = torch.linspace(0, 1, 17)
        self.assertEqual(dec.delta_rank(delta_h, base).shape, (17,))
        self.assertEqual(dec(delta_h, base).shape, (17,))
        self.assertEqual(dec.from_states(delta_h, torch.zeros_like(delta_h), base).shape, (17,))

    def test_untrained_decoder_is_identity(self):
        dec = self._decoder()
        delta_h = torch.randn(11, 8)
        base = torch.linspace(0, 1, 11)
        self.assertTrue(torch.allclose(dec(delta_h, base), base, atol=1e-6))

    def test_displacement_is_bounded(self):
        dec = self._decoder(max_shift=0.05)
        with torch.no_grad():
            dec.proj.weight.fill_(10.0)
            dec.proj.bias.fill_(5.0)
        out = dec.delta_rank(torch.randn(64, 8) * 100, torch.rand(64))
        self.assertLessEqual(float(out.abs().max()), 0.05 + 1e-6)

    def test_null_drift_zero_when_untrained(self):
        from state_feedback.decoder import null_drift

        drift = null_drift(self._decoder(), n_genes=128)
        self.assertAlmostEqual(drift["null_drift_mean_abs"], 0.0, places=6)

    def test_training_recovers_a_linear_teacher(self):
        from state_feedback.decoder import (
            TrainingSet,
            predict_delta_rank,
            train_delta_rank_decoder,
        )

        torch.manual_seed(0)
        n, d = 4096, 8
        delta_h = torch.randn(n, d)
        w = torch.zeros(d)
        w[0] = 1.0
        target = 0.05 * torch.tanh(delta_h @ w)
        data = TrainingSet(
            delta_h=delta_h,
            base_rank=torch.rand(n),
            target=target,
            tokens=list(range(n)),
        )
        dec, hist = train_delta_rank_decoder(
            data,
            d_model=d,
            max_shift=0.1,
            epochs=100,
            lr=0.05,
            batch_size=512,
            device="cpu",
        )
        pred = predict_delta_rank(dec, delta_h, data.base_rank)
        self.assertGreater(spearman_values(pred.tolist(), target.tolist()), 0.9)
        self.assertEqual(hist["n_samples"], n)

    def test_training_set_subset(self):
        from state_feedback.decoder import TrainingSet, token_mask

        data = TrainingSet(
            delta_h=torch.randn(4, 3),
            base_rank=torch.rand(4),
            target=torch.rand(4),
            tokens=[10, 11, 12, 13],
        )
        sub = data.subset(token_mask(data.tokens, {11, 13}))
        self.assertEqual(len(sub), 2)
        self.assertEqual(sub.tokens, [11, 13])

    def test_checkpoint_roundtrip(self):
        import tempfile

        from state_feedback.decoder import load_decoder, save_decoder

        dec = self._decoder(d_model=6, max_shift=0.2)
        with torch.no_grad():
            dec.proj.weight.normal_()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dec.pt"
            save_decoder(dec, path)
            restored = load_decoder(path)
        self.assertEqual(restored.d_model, 6)
        self.assertAlmostEqual(restored.max_shift, 0.2)
        self.assertTrue(torch.allclose(restored.proj.weight, dec.proj.weight))


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestGeneStateAlignment(unittest.TestCase):
    def test_align_ctrl_reindexes_by_token_identity(self):
        from state_feedback.gene_states import align_ctrl

        ctrl_ids = [10, 20, 30]
        pert_ids = [30, 10, 99]
        h_ctrl = torch.tensor([[1.0], [2.0], [3.0]])
        aligned, valid = align_ctrl(pert_ids, ctrl_ids, h_ctrl)
        self.assertEqual(aligned.squeeze(-1).tolist(), [3.0, 1.0, 0.0])
        self.assertEqual(valid.tolist(), [True, True, False])

    def test_absent_gene_gets_zero_delta(self):
        from state_feedback.gene_states import align_delta_h

        ctrl_ids = [10, 20]
        pert_ids = [99, 10]
        h_ctrl = torch.tensor([[1.0], [2.0]])
        h_pert = torch.tensor([[7.0], [5.0]])
        delta, valid = align_delta_h(ctrl_ids, pert_ids, h_ctrl, h_pert)
        self.assertEqual(delta.squeeze(-1).tolist(), [0.0, 4.0])
        self.assertEqual(valid.tolist(), [False, True])

    def test_batch_tensors_right_pads(self):
        from state_feedback.gene_states import batch_tensors

        ids, mask = batch_tensors([[1, 2, 3], [4, 5]], pad_token_id=0)
        self.assertEqual(ids.shape, (2, 3))
        self.assertEqual(ids[1].tolist(), [4, 5, 0])
        self.assertEqual(mask[1].tolist(), [1, 1, 0])


if __name__ == "__main__":
    unittest.main()
