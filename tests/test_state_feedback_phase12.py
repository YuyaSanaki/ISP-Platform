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
    displacement_summary,
    gap_closed_fraction,
    is_two_cycle,
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
