"""Genome-wide gene-level ISP cos sims (no goal states) carry the genes they compare.

``make_comparison_batch`` removes the perturbed position from the original embedding, so cos-sim
position k of perturbation j compares the k-th gene of the encoding without that position, not
``input_ids[k]``. The stub model is context free: a position compares the same gene on both sides
exactly when its cos sim is 1.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "core", ROOT / "contracts"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

try:
    import torch
    from datasets import Dataset

    from geneformer import in_silico_perturber as isp

    DEPS_MISSING = None
except Exception as exc:  # pragma: no cover - exercised only outside the container
    DEPS_MISSING = str(exc)

HIDDEN = 8
MAX_LEN = 64
CELL = list(range(10, 30))


class _StubOutputs:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _StubModel:
    def __call__(self, input_ids, attention_mask=None):
        weights = torch.arange(1, HIDDEN + 1, device=input_ids.device, dtype=torch.float32)
        embs = torch.sin((input_ids.float().unsqueeze(-1) + 3.0) * weights / 7.0)
        return _StubOutputs([embs, embs])


@unittest.skipIf(DEPS_MISSING, f"torch/datasets/geneformer unavailable: {DEPS_MISSING}")
class TestRemainingGeneIds(unittest.TestCase):
    def test_removes_positions(self):
        self.assertEqual(isp.remaining_gene_ids([5, 6, 7, 8], [2, 0]), [6, 8])
        self.assertEqual(isp.remaining_gene_ids([5, 6, 7, 8], [[1], [3]]), [5, 7])
        self.assertEqual(isp.remaining_gene_ids(torch.tensor([5, 6, 7]), [1]), [5, 7])
        self.assertEqual(isp.remaining_gene_ids([5, 6, 7], [-100]), [5, 6, 7])


@unittest.skipIf(DEPS_MISSING, f"torch/datasets/geneformer unavailable: {DEPS_MISSING}")
class TestGenomeWideGeneLabels(unittest.TestCase):
    def _check(self, perturb_type):
        example_cell = Dataset.from_list([{"input_ids": CELL, "length": len(CELL)}])
        batch, indices = isp.make_perturbation_batch(
            example_cell, perturb_type, "all", None, None, None, 0, 1)
        original_emb = isp.forward_pass_single_cell(_StubModel(), example_cell, -1)
        cos_sims = isp.quant_cos_sims(
            _StubModel(), perturb_type, batch, None, None, 4, -1, original_emb, "all", indices,
            False, None, None, 0, MAX_LEN, 1,
        ).cpu()
        mislabelled = 0
        for j, row in enumerate(cos_sims):
            genes = isp.remaining_gene_ids(torch.tensor(CELL), indices[j])
            ids = batch[j]["input_ids"]
            scored = ids[1:] if perturb_type == "overexpress" else ids
            self.assertEqual(genes, scored)
            self.assertEqual(len(genes), row.shape[0])
            torch.testing.assert_close(row, torch.ones(row.shape[0]))
            mislabelled += CELL[:len(genes)] != genes
        # input_ids[k] labels are wrong for every perturbed gene except the last
        self.assertEqual(mislabelled, len(cos_sims) - 1)

    def test_overexpress(self):
        self._check("overexpress")

    def test_delete(self):
        self._check("delete")


if __name__ == "__main__":
    unittest.main()
