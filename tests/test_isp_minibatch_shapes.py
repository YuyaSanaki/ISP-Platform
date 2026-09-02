"""Regression tests for ISP embedding shapes when a forward minibatch holds a single cell.

`quant_cos_sims` used to collapse the cell dimension of `[cell, gene, hidden]` embeddings
whenever the trailing minibatch had exactly one cell, which aborted long ISP runs with
"Embedding shape mismatch". The batching arithmetic is unchanged, so runs that already
worked must keep producing the same numbers.
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
N_GENES = 12
# 13 perturbations with forward_batch_size 4: the "(N-1)/fbs is integer" guard drops the
# batch size to 3, so the run still ends on a minibatch of one.
N_PERTURB = 13
TRAILING_ONE_BATCH = 4
SINGLE_CHUNK_BATCH = 13


class _StubOutputs:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _StubModel:
    """Deterministic stand-in for BERT: embeddings depend only on the token values."""

    def __call__(self, input_ids, attention_mask=None):
        weights = torch.arange(1, HIDDEN + 1, device=input_ids.device, dtype=torch.float32)
        embs = input_ids.float().unsqueeze(-1) * weights
        embs = torch.sin(embs / 7.0)
        return _StubOutputs([embs, embs])


def _perturbation_batch():
    base = list(range(2, 2 + N_GENES))
    rows = []
    indices_to_perturb = []
    for i in range(N_PERTURB):
        gene_idx = i % N_GENES
        tokens = base[:gene_idx] + base[gene_idx + 1 :]
        rows.append({"input_ids": tokens})
        indices_to_perturb.append([gene_idx])
    return Dataset.from_list(rows), indices_to_perturb


def _original_emb():
    base = torch.arange(2, 2 + N_GENES, dtype=torch.float32)
    weights = torch.arange(1, HIDDEN + 1, dtype=torch.float32)
    emb = torch.sin(base.unsqueeze(-1) * weights / 7.0)
    return isp._tensor_to_device(emb)


def _run(forward_batch_size):
    perturbation_batch, indices_to_perturb = _perturbation_batch()
    state_emb = torch.ones(1, HIDDEN, device=isp.ISP_device) * 0.25
    return isp.quant_cos_sims(
        _StubModel(),
        "delete",
        perturbation_batch,
        None,
        None,
        forward_batch_size,
        -1,
        _original_emb(),
        [2],
        indices_to_perturb,
        False,
        {"state_key": "disease", "start_state": "AD", "goal_state": "WT"},
        {"AD": state_emb, "WT": state_emb},
        0,
        N_GENES,
        1,
    )


@unittest.skipIf(DEPS_MISSING, f"torch/datasets/geneformer unavailable: {DEPS_MISSING}")
class TestBatchedLayerEmbs(unittest.TestCase):
    def test_single_cell_minibatch_keeps_cell_dim(self):
        embs = torch.zeros(1, N_GENES, HIDDEN)
        self.assertEqual(
            tuple(isp.batched_layer_embs(_StubOutputs([embs, embs]), -1).shape),
            (1, N_GENES, HIDDEN),
        )

    def test_align_emb_ranks_restores_leading_cell_dim(self):
        original = torch.zeros(1, N_GENES, HIDDEN)
        collapsed = torch.zeros(N_GENES, HIDDEN)
        aligned_original, aligned_minibatch = isp._align_emb_ranks(original, collapsed)
        self.assertEqual(aligned_original.shape, aligned_minibatch.shape)
        self.assertEqual(tuple(aligned_minibatch.shape), (1, N_GENES, HIDDEN))


@unittest.skipIf(DEPS_MISSING, f"torch/datasets/geneformer unavailable: {DEPS_MISSING}")
class TestCosSimShift(unittest.TestCase):
    def test_single_cell_shift_is_concatenable(self):
        original = torch.rand(1, N_GENES, HIDDEN, device=isp.ISP_device)
        minibatch = torch.rand(1, N_GENES, HIDDEN, device=isp.ISP_device)
        end_emb = torch.rand(1, HIDDEN, device=isp.ISP_device)

        shift = isp.cos_sim_shift(original, minibatch, end_emb, False)

        self.assertEqual(len(shift), 1)
        self.assertEqual(shift[0].dim(), 1)
        self.assertEqual(torch.cat(shift).numel(), 1)


@unittest.skipIf(DEPS_MISSING, f"torch/datasets/geneformer unavailable: {DEPS_MISSING}")
class TestQuantCosSimsBatching(unittest.TestCase):
    def test_trailing_single_cell_minibatch_matches_single_chunk(self):
        trailing = _run(TRAILING_ONE_BATCH)
        single_chunk = _run(SINGLE_CHUNK_BATCH)

        for state in ("AD", "WT"):
            self.assertEqual(trailing[state].shape, (N_PERTURB,))
            torch.testing.assert_close(trailing[state], single_chunk[state])


if __name__ == "__main__":
    unittest.main()
