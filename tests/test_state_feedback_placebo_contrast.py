"""Tests for the potential-outcome placebo contrast of State-feedback reranking."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))

try:
    import datasets  # noqa: F401
    import torch

    HAS_ML = True
except Exception:  # pragma: no cover - host python
    HAS_ML = False

try:
    import numpy  # noqa: F401

    HAS_NUMPY = True
except Exception:  # pragma: no cover
    HAS_NUMPY = False

D = 6


def _emb(tok, alias):
    g = torch.Generator().manual_seed(int(alias.get(int(tok), int(tok))))
    return torch.randn(D, generator=g)


def fake_hidden_states(alias=None):
    """Gene state = own embedding + cell context (shared and gene-specific) + position:
    inserting any gene shifts every downstream position (generic part); the context
    depends on which gene was inserted, differently for each gene."""
    alias = alias or {}

    def _hidden(model, raw_ids, pad_token_id, layer_to_quant):
        width = max(len(r) for r in raw_ids)
        out = torch.zeros(len(raw_ids), width, D)
        for b, ids in enumerate(raw_ids):
            embs = [_emb(t, alias) for t in ids]
            ctx = torch.stack(embs).mean(0)
            for j, e in enumerate(embs):
                out[b, j] = e + 0.3 * ctx + 0.3 * e * ctx + 0.05 * j
        return out

    return _hidden


def _pair(ctrl, pert, hidden):
    from state_feedback.gene_states import GeneStatePair

    return GeneStatePair(
        cell_indices=list(range(len(ctrl))),
        ctrl_ids=[list(c) for c in ctrl],
        pert_ids=[list(p) for p in pert],
        h_ctrl=hidden(None, ctrl, 0, 0),
        h_pert=hidden(None, pert, 0, 0),
    )


def _ds(rows):
    from datasets import Dataset

    return Dataset.from_dict({"input_ids": [list(r) for r in rows],
                              "length": [len(r) for r in rows]})


def _decoder(seed=0, max_shift=0.5, scale=1.0):
    from state_feedback.decoder import DeltaRankDecoder

    torch.manual_seed(seed)
    dec = DeltaRankDecoder(D, max_shift=max_shift)
    with torch.no_grad():
        dec.proj.weight.normal_().mul_(scale)
    return dec


FWD = dict(layer_to_quant=0, pad_token_id=0, model_input_size=64)


@unittest.skipUnless(HAS_ML, "torch/datasets not available")
class TestSwapAndSlots(unittest.TestCase):
    def setUp(self):
        from state_feedback import placebo_contrast

        self.pc = placebo_contrast

    def test_swap_in_place_and_drop_old_placebo(self):
        new, excl = self.pc.swap_tokens([9, 1, 2, 7, 3], {9: 7})
        self.assertEqual(new, [7, 1, 2, 3])
        self.assertEqual(excl, {9, 7})

    def test_swap_only_present_chain_genes(self):
        new, excl = self.pc.swap_tokens([9, 1, 2, 5], {9: 50, 8: 7})
        self.assertEqual(new, [50, 1, 2, 5])
        self.assertEqual(excl, {9, 50})

    def test_swap_without_chain_genes_is_identity(self):
        ids = [1, 2, 3, 7]
        new, excl = self.pc.swap_tokens(ids, {9: 7, 8: 6})
        self.assertEqual(new, ids)
        self.assertEqual(excl, set())

    def test_swap_truncates(self):
        new, _ = self.pc.swap_tokens([9, 1, 2, 3], {9: 50}, max_len=3)
        self.assertEqual(new, [50, 1, 2])

    def test_slot_substitutions_by_step_and_position(self):
        subs = self.pc.slot_substitutions([[10], [11, 12]], [[[20], [21, 22]], [[30], [31, 32]]])
        self.assertEqual(subs, [{10: 20, 11: 21, 12: 22}, {10: 30, 11: 31, 12: 32}])

    def test_placebo_sharing_a_gene_is_left_out(self):
        subs = self.pc.slot_substitutions([[10], [11]], [[[20], [11]], [[30], [31]]])
        self.assertEqual(subs, [{10: 30, 11: 31}])
        with self.assertRaises(ValueError):
            self.pc.slot_substitutions([[10], [11]], [[[11], [20]]])

    def test_slot_count_must_match(self):
        with self.assertRaises(ValueError):
            self.pc.slot_substitutions([[10], [11]], [[[20]]])

    def test_delete_step_indices(self):
        steps = [{"name": "a", "type": "knockdown"}, {"name": "b", "type": "overexpress"},
                 {"name": "c", "type": "delete"}]
        self.assertEqual(self.pc.delete_step_indices(steps), frozenset({0, 2}))
        self.assertEqual(self.pc.delete_step_indices(steps[1:2]), frozenset())
        with self.assertRaises(ValueError):
            self.pc.delete_step_indices([{"name": "x", "type": "activate"}])

    def test_mixed_substitution_split_by_step_type(self):
        subs = self.pc.slot_substitutions([[10], [11], [12]], [[[20], [21], [22]]],
                                          frozenset({0}))
        self.assertEqual(subs, [self.pc.MixedSubstitution(oe={11: 21, 12: 22}, ko={10: 20})])


@unittest.skipUnless(HAS_ML, "torch/datasets not available")
class TestEstimationDraw(unittest.TestCase):
    def test_nested_and_seeded(self):
        from state_feedback import placebo_contrast as pc

        steps = [{"index": i + 1, "name": f"s{i + 1}", "type": "overexpress", "genes": [str(t)]}
                 for i, t in enumerate((5, 6))]
        tbs = [[5], [6]]
        population = list(range(1, 400))
        det = {t: 0.5 for t in population}
        rank = {t: 0.5 for t in population}
        a, _ = pc.draw_estimation_placebos(steps, tbs, population, det, rank, n=5, seed=1)
        b, _ = pc.draw_estimation_placebos(steps, tbs, population, det, rank, n=10, seed=1)
        c, _ = pc.draw_estimation_placebos(steps, tbs, population, det, rank, n=5, seed=2)
        self.assertEqual(a, b[:5])
        self.assertNotEqual(a, c)
        for p in b:
            self.assertEqual([len(s) for s in p], [1, 1])
            self.assertFalse({5, 6} & {t for s in p for t in s})


@unittest.skipUnless(HAS_ML, "torch/datasets not available")
class TestMediator(unittest.TestCase):
    def setUp(self):
        from state_feedback import placebo_contrast as pc

        self.pc = pc
        self.ctrl = [[1, 2, 3, 4, 5, 6], [2, 3, 1, 6, 5, 4]]
        # chain gene 9 inserted at the front (absent from the control)
        self.pert = [[9, 1, 2, 3, 4, 5, 6], [9, 2, 3, 1, 6, 5, 4]]
        self.subs = [{9: 40}, {9: 41}]

    def _run(self, hidden, ctrl, pert, subs, forward=None):
        pair = _pair(ctrl, pert, hidden)
        with mock.patch("state_feedback.gene_states._hidden_states", forward or hidden):
            m_g, counts = self.pc.placebo_mediator(None, pair, subs, **FWD)
        return pair, m_g, counts

    def test_matches_manual_counterfactual(self):
        hidden = fake_hidden_states()
        pair, m_g, counts = self._run(hidden, self.ctrl, self.pert, self.subs)
        for r in range(2):
            h_ctrl = hidden(None, [self.ctrl[r]], 0, 0)[0]
            pos_ctrl = {t: j for j, t in enumerate(self.ctrl[r])}
            expect = torch.zeros(len(self.pert[r]), D)
            for sub in self.subs:
                swapped = [sub.get(t, t) for t in self.pert[r]]
                h_sw = hidden(None, [swapped], 0, 0)[0]
                for j, t in enumerate(self.pert[r]):
                    if t in pos_ctrl:
                        expect[j] += h_sw[j] - h_ctrl[pos_ctrl[t]]
            expect /= len(self.subs)
            self.assertTrue(torch.allclose(m_g[r], expect, atol=1e-6))
            self.assertEqual(counts[r].tolist(), [0.0] + [2.0] * 6)

    def test_generic_chain_gives_zero_contrast(self):
        # a chain gene whose state is that of the placebo: no gene-specific effect left
        hidden = fake_hidden_states(alias={9: 40})
        pair, m_g, _ = self._run(hidden, self.ctrl, self.pert, [{9: 40}])
        for r in range(2):
            feat, _ = self.pc.contrast_delta_h(pair, r, m_g[r])
            self.assertEqual(float(feat.abs().max()), 0.0)
            delta, _ = pair.delta_h(r)
            self.assertGreater(float(delta.abs().max()), 0.0)

    def test_no_chain_gene_gives_exact_zero_without_reencoding(self):
        hidden = fake_hidden_states()
        calls = []

        def counting(model, raw_ids, pad, layer):
            calls.append(len(raw_ids))
            return hidden(model, raw_ids, pad, layer)

        other = [[8, 1, 2, 3, 4, 5, 6], [8, 2, 3, 1, 6, 5, 4]]
        pair, m_g, _ = self._run(hidden, self.ctrl, other, self.subs, forward=counting)
        self.assertEqual(calls, [])
        for r in range(2):
            feat, _ = self.pc.contrast_delta_h(pair, r, m_g[r])
            self.assertEqual(float(feat.abs().max()), 0.0)

    def test_only_cells_with_chain_genes_are_reencoded(self):
        hidden = fake_hidden_states()
        calls = []

        def counting(model, raw_ids, pad, layer):
            calls.append(len(raw_ids))
            return hidden(model, raw_ids, pad, layer)

        mixed = [self.pert[0], [8, 2, 3, 1, 6, 5, 4]]
        self._run(hidden, self.ctrl, mixed, self.subs, forward=counting)
        self.assertEqual(calls, [1, 1])


@unittest.skipUnless(HAS_ML, "torch/datasets not available")
class TestRerankAndTraining(unittest.TestCase):
    def setUp(self):
        from state_feedback import placebo_contrast as pc

        self.pc = pc
        self.ctrl = [[1, 2, 3, 4, 5, 6], [2, 3, 1, 6, 5, 4], [6, 5, 4, 3, 2, 1]]
        self.pert = [[9, 1, 2, 3, 4, 5, 6], [9, 2, 3, 1, 6, 5, 4], [9, 6, 5, 4, 3, 2, 1]]
        self.subs = [{9: 40}, {9: 41}, {9: 42}]

    def _rerank(self, ctrl, pert, decoder, hidden=None):
        with mock.patch("state_feedback.gene_states._hidden_states",
                        hidden or fake_hidden_states()):
            return self.pc.rerank_placebo_contrast(
                None, decoder, _ds(ctrl), _ds(pert), self.subs,
                forward_batch_size=2, **FWD,
            )

    def test_null_feedback_has_zero_displacement(self):
        after, diag = self._rerank(self.ctrl, self.ctrl, _decoder(max_shift=0.5, scale=50.0))
        self.assertEqual(after, self.ctrl)
        self.assertAlmostEqual(diag["rerank_spearman_before_after"], 1.0)

    def test_generic_chain_does_not_move_the_encoding(self):
        after, _ = self._rerank(self.ctrl, self.pert, _decoder(max_shift=0.5, scale=50.0),
                                hidden=fake_hidden_states(alias={9: 40, 41: 40, 42: 40}))
        self.assertEqual(after, self.pert)

    def test_overexpression_reference_output(self):
        # Fixed reference for the overexpression path: must not change when other
        # perturbation types are added.
        after, diag = self._rerank(self.ctrl, self.pert,
                                   _decoder(seed=3, max_shift=0.5, scale=50.0))
        self.assertNotEqual(after, self.pert)
        self.assertEqual(after, REFERENCE_AFTER)
        self.assertAlmostEqual(diag["contrast_placebos_mean"], 3.0)

    def test_training_samples(self):
        hidden = fake_hidden_states()
        teacher = {t: 0.1 * t for t in (1, 2, 3, 4, 5, 9)}
        with mock.patch("state_feedback.gene_states._hidden_states", hidden):
            data = self.pc.collect_contrast_training_samples(
                None, _ds(self.ctrl), _ds(self.pert), teacher, self.subs,
                forward_batch_size=2, max_genes_per_cell=10, **FWD,
            )
            pair = _pair(self.ctrl, self.pert, hidden)
            m_g, _ = self.pc.placebo_mediator(None, pair, self.subs, **FWD)
        self.assertEqual(sorted(set(data.tokens)), [1, 2, 3, 4, 5])
        self.assertEqual(len(data), 15)
        first = [j for j, t in enumerate(self.pert[0]) if t in (1, 2, 3, 4, 5)]
        feat, _ = self.pc.contrast_delta_h(pair, 0, m_g[0])
        got = {t: data.delta_h[i] for i, t in enumerate(data.tokens[:5])}
        for j in first:
            self.assertTrue(torch.allclose(got[self.pert[0][j]], feat[j], atol=1e-6))
        self.assertTrue(torch.allclose(
            data.target[:5], torch.tensor([teacher[t] for t in data.tokens[:5]])
        ))


REFERENCE_AFTER = [[2, 9, 5, 1, 3, 4, 6], [2, 9, 5, 3, 1, 6, 4], [5, 9, 2, 6, 4, 3, 1]]


def _ko(ko, oe=None):
    from state_feedback.placebo_contrast import MixedSubstitution

    return MixedSubstitution(oe=dict(oe or {}), ko=dict(ko))


@unittest.skipUnless(HAS_ML, "torch/datasets not available")
class TestUndoDeleteRedo(unittest.TestCase):
    def setUp(self):
        from state_feedback.placebo_contrast import undo_delete_redo

        self.f = undo_delete_redo

    def _deleted(self, ids, genes):
        return [t for t in ids if t not in genes]

    def test_before_feedback_equals_placebo_deleted_from_start(self):
        start = [1, 2, 50, 3, 4, 60, 5, 6, 7]
        cases = [
            {50: 2, 60: 6},    # placebo before / after the factor
            {50: 2, 60: 4},    # placebo right before the factor
            {50: 7, 60: 1},    # placebo at the end / at the front
        ]
        for ko in cases:
            x_t = self._deleted(start, {50, 60})
            new, excl = self.f(x_t, start, _ko(ko))
            self.assertEqual(new, self._deleted(start, set(ko.values())), ko)
            self.assertEqual(excl, set(ko.values()))

    def test_factor_first_in_start(self):
        start = [50, 1, 2, 3]
        new, _ = self.f([1, 2, 3], start, _ko({50: 2}))
        self.assertEqual(new, [50, 1, 3])

    def test_factor_follows_its_anchor_after_rerank(self):
        start = [1, 2, 50, 3, 4, 5]
        x_t = [3, 1, 4, 2, 5]           # reranked after 50 was deleted
        new, _ = self.f(x_t, start, _ko({50: 5}))
        self.assertEqual(new, [3, 1, 4, 2, 50])
        self.assertEqual(len(new), len(x_t))
        self.assertEqual(set(new), set(x_t) - {5} | {50})

    def test_factor_absent_from_start_gets_no_placebo_edit(self):
        start = [1, 2, 3, 4]
        new, excl = self.f([1, 2, 3, 4], start, _ko({50: 3}))
        self.assertEqual((new, excl), ([1, 2, 3, 4], set()))

    def test_missing_placebo_gene_gives_no_counterfactual(self):
        self.assertIsNone(self.f([1, 2, 3], [1, 50, 2, 3], _ko({50: 99})))

    def test_mixed_chain(self):
        # KO of 50, then OE of 9 (front-pinned): the OE slot is swapped as in
        # swap_tokens, the KO slot goes back after its anchor.
        from state_feedback.placebo_contrast import swap_tokens

        start = [1, 2, 50, 3, 40, 4]
        x_t = [9, 1, 2, 3, 40, 4]
        new, excl = self.f(x_t, start, _ko({50: 3}, oe={9: 40}))
        oe_only, _ = swap_tokens(x_t, {9: 40})
        self.assertEqual(oe_only, [40, 1, 2, 3, 4])
        self.assertEqual(new, [40, 1, 2, 50, 4])
        self.assertEqual(excl, {9, 40, 3})

    def test_fallback_after_front_overexpressed_placebos(self):
        start = [50, 1, 2, 3]
        new, _ = self.f([9, 1, 2, 3], start, _ko({50: 3}, oe={9: 70}))
        self.assertEqual(new, [70, 50, 1, 2])


@unittest.skipUnless(HAS_ML, "torch/datasets not available")
class TestDeleteContrast(unittest.TestCase):
    def setUp(self):
        from state_feedback import placebo_contrast as pc

        self.pc = pc
        # cell 2 has no factor 50: untreated
        self.ctrl = [[1, 2, 50, 3, 4, 5], [2, 50, 3, 1, 5, 4], [5, 4, 3, 2, 1, 6]]
        self.pert = [[1, 2, 3, 4, 5], [2, 3, 1, 5, 4], [5, 4, 3, 2, 1, 6]]
        self.subs = [_ko({50: 4}), _ko({50: 5}), _ko({50: 1})]

    def test_untreated_cell_has_zero_contrast_and_no_reencoding(self):
        hidden = fake_hidden_states()
        calls = []

        def counting(model, raw_ids, pad, layer):
            calls.append(len(raw_ids))
            return hidden(model, raw_ids, pad, layer)

        pair = _pair(self.ctrl, self.pert, hidden)
        with mock.patch("state_feedback.gene_states._hidden_states", counting):
            m_g, counts = self.pc.placebo_mediator(None, pair, self.subs, **FWD)
        self.assertEqual(calls, [2, 2, 2])
        feat, _ = self.pc.contrast_delta_h(pair, 2, m_g[2])
        self.assertEqual(float(feat.abs().max()), 0.0)

    def test_generic_deletion_gives_zero_contrast(self):
        # the factor's embedding equals the placebo's: deleting either is the same
        hidden = fake_hidden_states(alias={50: 4})
        ctrl = [[1, 2, 50, 3, 4, 5, 6, 7]]
        pert = [[1, 2, 3, 4, 5, 6, 7]]
        pair = _pair(ctrl, pert, hidden)
        with mock.patch("state_feedback.gene_states._hidden_states", hidden):
            m_g, _ = self.pc.placebo_mediator(None, pair, [_ko({50: 4})], **FWD)
        feat, _ = self.pc.contrast_delta_h(pair, 0, m_g[0])
        delta, _ = pair.delta_h(0)
        self.assertGreater(float(delta.abs().max()), 0.0)
        # genes between the restored factor and the deleted placebo shift by one
        # position; the genes before both edits see the same context
        self.assertLess(float(feat[:2].abs().max()), 1e-5)

    def test_untreated_cell_does_not_move(self):
        with mock.patch("state_feedback.gene_states._hidden_states", fake_hidden_states()):
            after, _ = self.pc.rerank_placebo_contrast(
                None, _decoder(seed=3, max_shift=0.5, scale=50.0), _ds(self.ctrl),
                _ds(self.pert), self.subs, forward_batch_size=3, **FWD,
            )
        self.assertEqual(after[2], self.pert[2])

    def test_training_uses_treated_cells_only(self):
        teacher = {t: 0.1 * t for t in (1, 2, 3, 4, 5, 6)}
        with mock.patch("state_feedback.gene_states._hidden_states", fake_hidden_states()):
            data = self.pc.collect_contrast_training_samples(
                None, _ds(self.ctrl), _ds(self.pert), teacher, self.subs,
                forward_batch_size=3, max_genes_per_cell=10, **FWD,
            )
        self.assertGreater(len(data), 0)
        self.assertNotIn(6, data.tokens)   # only cell 2 (untreated) has gene 6
        self.assertNotIn(50, data.tokens)


@unittest.skipUnless(HAS_NUMPY, "numpy not available")
class TestRandomizeStepsOnly(unittest.TestCase):
    def test_keeps_configured_steps(self):
        from state_feedback import random_chains

        steps = [{"index": i + 1, "name": g, "type": t, "genes": [g]}
                 for i, (g, t) in enumerate([("TP53", "delete"), ("KLF4", "overexpress")])]
        chains = {"r0": ([{"index": 1, "name": "r0_s1", "type": "delete", "genes": ["7"]},
                          {"index": 2, "name": "r0_s2", "type": "overexpress", "genes": ["8"]}],
                         [[7], [8]])}
        out = random_chains.randomize_steps_only(chains, steps, [[100], [200]], [1])
        self.assertEqual(out["r0"][1], [[7], [200]])
        self.assertEqual(out["r0"][0][1]["genes"], ["KLF4"])
        self.assertEqual(out["r0"][0][0]["genes"], ["7"])
        with self.assertRaises(ValueError):
            random_chains.randomize_steps_only(chains, steps, [[100], [200]], [3])


@unittest.skipUnless(HAS_NUMPY, "numpy not available")
class TestSpecificGainConditional(unittest.TestCase):
    def setUp(self):
        from state_feedback.stability import specific_gain, specific_gain_conditional

        self.full, self.cond = specific_gain, specific_gain_conditional
        self.refs = [[0.0] * 4] * 4
        self.randoms = [[0.1, 0.2, 0.1, 0.0], [0.2, 0.2, 0.3, 0.1],
                        [0.3, 0.1, 0.3, 0.2], [0.1, 0.1, 0.5, 0.1]]
        self.conf = [0.4, 0.3, 0.4, 0.2]

    def test_all_available_matches_specific_gain(self):
        a = self.full(self.conf, [0.0] * 4, self.randoms, self.refs, n_boot=200)
        b = self.cond(self.conf, [0.0] * 4, self.randoms, self.refs, [True] * 4,
                      [[True] * 4] * 4, n_boot=200)
        for k in ("specific_gain_mean", "random_gain_mean", "z_vs_random", "rank_among_random",
                  "frac_cells_above_all_random", "null_empirical_p", "specific_gain_ci_low"):
            self.assertAlmostEqual(a[k], b[k], places=10, msg=k)
        self.assertEqual(b["n_cells_dropped"], 0.0)

    def test_unavailable_chains_and_untreated_cells_are_left_out(self):
        avail = [[True, True, True, True], [True, True, True, True],
                 [True, True, True, True], [True, True, False, True]]
        r = self.cond(self.conf, [0.0] * 4, self.randoms, self.refs,
                      [True, True, True, False], avail, n_boot=50)
        self.assertEqual(r["n_cells"], 3.0)
        rand = numpy.array([[0.1, 0.2, 0.1], [0.2, 0.2, 0.3], [0.3, 0.1, 0.3],
                            [0.1, 0.1, numpy.nan]])
        expect = numpy.mean(numpy.array([0.4, 0.3, 0.4]) - numpy.nanmean(rand, axis=0))
        self.assertAlmostEqual(r["specific_gain_mean"], expect)
        r3 = self.cond(self.conf, [0.0] * 4, self.randoms, self.refs, [True] * 4, avail,
                       min_random_per_cell=4, n_boot=50)
        self.assertEqual(r3["n_cells_dropped"], 1.0)


@unittest.skipUnless(HAS_NUMPY, "numpy not available")
class TestSpecificGainRanks(unittest.TestCase):
    def test_rank_z_and_cells_above_all(self):
        from state_feedback.stability import specific_gain

        refs = [[0.0] * 4] * 4
        randoms = [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2],
                   [0.3, 0.3, 0.3, 0.3], [0.1, 0.1, 0.5, 0.1]]
        r = specific_gain([0.4, 0.4, 0.4, 0.4], [0.0] * 4, randoms, refs, n_boot=20)
        self.assertEqual(r["rank_among_random"], 1.0)
        self.assertAlmostEqual(r["frac_cells_above_all_random"], 0.75)
        means = numpy.array([0.1, 0.2, 0.3, 0.2])
        self.assertAlmostEqual(r["z_vs_random"], (0.4 - means.mean()) / means.std(ddof=1))
        low = specific_gain([0.15] * 4, [0.0] * 4, randoms, refs, n_boot=20)
        self.assertEqual(low["rank_among_random"], 4.0)


if __name__ == "__main__":
    unittest.main()
