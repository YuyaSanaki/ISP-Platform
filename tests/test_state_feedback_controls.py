"""Unit tests for state_feedback.controls (base-rank and perturbation-specificity controls).

Skipped when numpy / torch are unavailable (host python has no ML deps; Docker has them).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))

try:
    import numpy as np

    HAS_NUMPY = True
except Exception:  # pragma: no cover - host python
    HAS_NUMPY = False

try:
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover - host python
    HAS_TORCH = False


@unittest.skipUnless(HAS_NUMPY, "numpy not available")
class TestRanks(unittest.TestCase):
    def test_average_rank_matches_tie_convention(self):
        from state_feedback.controls import average_rank

        np.testing.assert_allclose(average_rank([3.0, 1.0, 2.0]), [2.0, 0.0, 1.0])
        np.testing.assert_allclose(average_rank([1.0, 2.0, 2.0, 5.0]), [0.0, 1.5, 1.5, 3.0])

    def test_spearman_perfect_and_constant(self):
        from state_feedback.controls import spearman

        x = np.arange(10.0)
        self.assertAlmostEqual(spearman(x, x ** 3), 1.0)
        self.assertAlmostEqual(spearman(x, -x), -1.0)
        self.assertNotEqual(spearman(x, np.ones(10)), spearman(x, np.ones(10)))  # nan


@unittest.skipUnless(HAS_NUMPY, "numpy not available")
class TestBaseRankControl(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.n = 4000
        self.base = rng.random(self.n)
        self.signal = rng.standard_normal(self.n)
        # target = strong base-rank trend + weaker base-independent signal
        self.target = -(self.base - 0.5) + 0.3 * self.signal
        self.tokens = np.arange(self.n)
        self.cells = np.repeat(np.arange(40), self.n // 40)

    def test_base_only_predictor_has_high_pooled_but_zero_partial(self):
        from state_feedback.controls import (
            crossfit_base_only,
            partial_spearman_given_base,
            spearman,
        )

        pred = crossfit_base_only(self.base, self.target, self.tokens)
        self.assertGreater(spearman(pred, self.target), 0.5)
        self.assertLess(abs(partial_spearman_given_base(pred, self.target, self.base)), 0.1)

    def test_partial_isolates_base_independent_signal(self):
        from state_feedback.controls import partial_spearman_given_base

        self.assertGreater(partial_spearman_given_base(self.signal, self.target, self.base), 0.5)
        noise = np.random.default_rng(1).standard_normal(self.n)
        self.assertLess(abs(partial_spearman_given_base(noise, self.target, self.base)), 0.1)

    def test_crossfit_never_uses_own_fold(self):
        from state_feedback.controls import crossfit_base_only

        target = np.where(self.tokens % 2 == 0, 1.0, -1.0)
        pred = crossfit_base_only(self.base, target, self.tokens)
        # even tokens are predicted from odd tokens only, and vice versa
        self.assertTrue(np.all(pred[self.tokens % 2 == 0] < 0))
        self.assertTrue(np.all(pred[self.tokens % 2 == 1] > 0))

    def test_bootstrap_partial_ci_brackets_point(self):
        from state_feedback.controls import bootstrap_partial

        out = bootstrap_partial(self.signal, self.target, self.base, self.cells, n_boot=50, seed=0)
        self.assertLessEqual(out["ci_low"], out["partial"])
        self.assertGreaterEqual(out["ci_high"], out["partial"])
        self.assertTrue(out["excludes_zero"])

    def test_bootstrap_partial_diff_same_predictor_is_zero(self):
        from state_feedback.controls import bootstrap_partial_diff

        out = bootstrap_partial_diff(
            self.signal, self.signal, self.target, self.base, self.cells, n_boot=20, seed=0
        )
        self.assertAlmostEqual(out["diff"], 0.0)
        self.assertFalse(out["excludes_zero"])

    def test_stratified_and_target_by_stratum_shapes(self):
        from state_feedback.controls import stratified_spearman, target_by_base_stratum

        strata = target_by_base_stratum(self.target, self.base)
        self.assertEqual(len(strata), 5)
        self.assertGreater(strata[0], strata[-1])  # top-ranked genes move down
        self.assertEqual(len(stratified_spearman(self.signal, self.target, self.base)), 5)


@unittest.skipUnless(HAS_NUMPY, "numpy not available")
class TestSpecificityHelpers(unittest.TestCase):
    def test_detection_rate_counts_cells_not_occurrences(self):
        from state_feedback.controls import detection_rate

        rate = detection_rate([[1, 2, 2], [2, 3], [4]])
        self.assertAlmostEqual(rate[2], 2 / 3)
        self.assertAlmostEqual(rate[1], 1 / 3)

    def test_detection_matched_pool(self):
        from state_feedback.controls import detection_matched_pool

        detection = {1: 0.05, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.0}
        self.assertEqual(detection_matched_pool([1, 2, 3, 4, 5], detection, 0.1), [1, 2, 3])

    def test_versus_random(self):
        from state_feedback.controls import versus_random

        out = versus_random(0.5, [0.0, 0.1, 0.2, 0.6])
        self.assertEqual(out["random_n"], 4)
        self.assertEqual(out["n_random_ge"], 1)
        self.assertAlmostEqual(out["empirical_p"], 2 / 5)
        self.assertEqual(versus_random(0.5, [])["random_n"], 0)


@unittest.skipUnless(HAS_NUMPY and HAS_TORCH, "numpy/torch not available")
class TestDecoderControls(unittest.TestCase):
    def _samples(self, n_cells=8, n_genes=50, d=6):
        from state_feedback.samples import EvalSamples

        torch.manual_seed(0)
        n = n_cells * n_genes
        delta_h = torch.randn(n, d)
        base = torch.rand(n)
        target = -(base - 0.5) + 0.3 * delta_h[:, 0]
        return EvalSamples(
            cell_index=[c for c in range(n_cells) for _ in range(n_genes)],
            tokens=[1000 + g for _ in range(n_cells) for g in range(n_genes)],
            base_rank=base,
            target=target,
            delta_h=delta_h,
            delta_norm=torch.zeros(n),
        )

    def _decoder(self, d=6):
        from state_feedback.decoder import DeltaRankDecoder

        dec = DeltaRankDecoder(d, max_shift=0.5)
        with torch.no_grad():
            dec.proj.weight.zero_()
            dec.proj.weight[0, 0] = 1.0   # reads delta_h[:, 0]
        return dec

    def test_decoder_variants_isolate_inputs(self):
        from state_feedback.controls import decoder_variants, partial_spearman_given_base

        s = self._samples()
        v = decoder_variants(self._decoder(), s, seed=0)
        self.assertEqual(set(v), {"linear_deltarank", "delta_h_shuffled"})
        tgt, base = s.target.numpy(), s.base_rank.numpy()
        self.assertGreater(partial_spearman_given_base(v["linear_deltarank"], tgt, base), 0.5)
        self.assertLess(abs(partial_spearman_given_base(v["delta_h_shuffled"], tgt, base)), 0.2)

    def test_base_rank_control_report(self):
        from state_feedback.controls import base_rank_control

        s = self._samples()
        rep = base_rank_control(s, self._decoder(), extra={"zero": [0.0] * len(s)}, n_boot=20)
        self.assertEqual(rep["n_samples"], len(s))
        self.assertIn("zero", rep["methods"])
        self.assertGreater(rep["linear_partial"]["partial"], 0.5)
        self.assertTrue(rep["linear_partial"]["excludes_zero"])
        self.assertGreater(rep["mean_delta_h_norm"], 0.0)

    def test_base_rank_control_without_decoder(self):
        from state_feedback.controls import base_rank_control

        rep = base_rank_control(self._samples(), None, n_boot=5)
        self.assertNotIn("linear_partial", rep)
        self.assertEqual(rep["methods"], {})


if __name__ == "__main__":
    unittest.main()
