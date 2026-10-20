"""Group ISP without goal states scores each compared gene once, against the same gene.

With ``genes_to_perturb`` set and no ``cell_states_to_model``, each cell's row of cosine
similarities compares the original and perturbed embedding position by position, after both
sides have dropped the perturbed genes present in the cell (and, for overexpression, the
original has dropped one lowest-ranked gene per absent OE gene). The first L - k positions hold
the same genes in both contexts. The reference below finds each gene by token in separate
forward passes of the original and perturbed encodings, so it checks the gene labels and the
number of genes pooled into the cell score without relying on positions.
"""
from __future__ import annotations

import sys
import unittest
from collections import defaultdict
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
OE_TOKENS = [900, 901, 902]


class _StubOutputs:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _StubModel:
    """Embeddings depend on the token and its position, so a gene's context matters."""

    def __call__(self, input_ids, attention_mask=None):
        weights = torch.arange(1, HIDDEN + 1, device=input_ids.device, dtype=torch.float32)
        pos = torch.arange(input_ids.shape[-1], device=input_ids.device, dtype=torch.float32)
        embs = torch.sin((input_ids.float().unsqueeze(-1) + 3.0) * weights / 7.0)
        embs = embs + 0.5 * torch.cos(pos.unsqueeze(-1) * weights / 5.0)
        return _StubOutputs([embs, embs])


def _cells(perturb_type):
    # Mixed lengths, sorted longest first as downsample_and_sort leaves them.
    lengths = [30, 30, 27, 25, 22, 18]
    cells = []
    for i, n in enumerate(lengths):
        ids = list(range(10 + 40 * i, 10 + 40 * i + n))
        if perturb_type == "delete":
            ids = ids[:3] + OE_TOKENS + ids[3:-3]
        elif i == 2:
            ids[5] = OE_TOKENS[0]
        elif i == 3:
            ids[2], ids[9], ids[20] = OE_TOKENS
        cells.append(ids)
    return cells


def _perturb(perturb_type, ids):
    idx = [ids.index(t) for t in OE_TOKENS if t in ids] or [-100]
    ex = {"input_ids": list(ids), "length": len(ids), "tokens_to_perturb": list(OE_TOKENS),
          "perturb_index": idx}
    ex = isp.overexpress_tokens(ex) if perturb_type == "overexpress" else isp.delete_indices(ex)
    return ex["input_ids"], idx


def _score(perturb_type, forward_batch_size):
    cells = _cells(perturb_type)
    original = Dataset.from_list([{"input_ids": c, "length": len(c)} for c in cells])
    rows, indices = [], []
    for c in cells:
        ids, idx = _perturb(perturb_type, c)
        rows.append({"input_ids": ids, "length": len(ids)})
        indices.append(idx)
    cos_sims = isp.quant_cos_sims(
        _StubModel(), perturb_type, Dataset.from_list(rows), None, None, forward_batch_size,
        -1, original, list(OE_TOKENS), indices, True, None, {}, 0, MAX_LEN, 1,
    )
    out = defaultdict(list)
    isp.add_group_cos_sims(out, cos_sims, cells, indices, OE_TOKENS, "cell_and_gene")
    return out


def _reference(perturb_type):
    model = _StubModel()
    cos = torch.nn.CosineSimilarity(dim=0)
    out = defaultdict(list)
    key = tuple(OE_TOKENS)
    for c in _cells(perturb_type):
        ids, _ = _perturb(perturb_type, c)
        orig_emb = model(torch.tensor([c])).hidden_states[-1][0]
        pert_emb = model(torch.tensor([ids])).hidden_states[-1][0]
        shared = [g for g in c if g in ids and g not in OE_TOKENS]
        sims = [cos(orig_emb[c.index(g)], pert_emb[ids.index(g)]).item() for g in shared]
        out[(key, "cell_emb")].append(sum(sims) / len(sims))
        for g, s in zip(shared, sims):
            out[(key, g)].append(s)
    return out


@unittest.skipIf(DEPS_MISSING, f"torch/datasets/geneformer unavailable: {DEPS_MISSING}")
class TestGroupCosSimsNoStates(unittest.TestCase):
    def _check(self, perturb_type):
        reference = _reference(perturb_type)
        for fbs in (1, 2, 8):
            with self.subTest(forward_batch_size=fbs):
                got = _score(perturb_type, fbs)
                self.assertEqual(set(got), set(reference))
                for key, values in reference.items():
                    torch.testing.assert_close(
                        torch.tensor(got[key]), torch.tensor(values), msg=str(key)
                    )

    def test_overexpress(self):
        self._check("overexpress")

    def test_delete(self):
        self._check("delete")

    def test_compared_genes(self):
        ids = [5, 900, 6, 7, 901, 8, 9]
        self.assertEqual(isp.group_compared_genes(ids, [1, 4], 3), [5, 6, 7, 8])
        self.assertEqual(isp.group_compared_genes(ids, [-100], 2), [5, 900, 6, 7, 901])
        self.assertEqual(isp.group_compared_genes(ids, [1, 4], 2), [5, 6, 7, 8, 9])


if __name__ == "__main__":
    unittest.main()
