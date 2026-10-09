"""Group goal-state scoring must not mean-pool padding.

Group delete / overexpress scoring removes k positions from the original embedding (and the k
leading OE positions from the perturbed one). Cells shorter than the longest cell of their
forward minibatch are padded, so pooling over the unremoved length L takes up to k padding
hidden states into the mean. With forward_batch_size 1 there is no padding, so a padding-free
score is the same at any batch size. ``legacy_padding_mean`` keeps the earlier pooling.
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
OE_TOKENS = [900, 901, 902]
STATES = {"state_key": "disease", "start_state": "somatic", "goal_state": "pluripotent"}


class _StubOutputs:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _StubModel:
    """Embeddings depend only on the token value; the pad token (0) gets a nonzero vector."""

    def __call__(self, input_ids, attention_mask=None):
        weights = torch.arange(1, HIDDEN + 1, device=input_ids.device, dtype=torch.float32)
        embs = torch.sin((input_ids.float().unsqueeze(-1) + 3.0) * weights / 7.0)
        return _StubOutputs([embs, embs])


def _cells():
    # Mixed lengths; OE genes absent, one present, or all present.
    lengths = [30, 27, 22, 25, 18, 30]
    cells = []
    for i, n in enumerate(lengths):
        ids = list(range(10 + i, 10 + i + n))
        if i == 1:
            ids[5] = OE_TOKENS[0]
        if i == 3:
            ids[2], ids[9], ids[20] = OE_TOKENS
        cells.append(ids)
    return cells


def _perturb_index(ids, tokens):
    idx = [ids.index(t) for t in tokens if t in ids]
    return idx or [-100]


def _datasets(perturb_type):
    cells = _cells()
    if perturb_type == "delete":
        cells = [c[:3] + OE_TOKENS + c[3:] for c in cells]
    original = Dataset.from_list(
        [{"input_ids": c, "length": len(c), "disease": "somatic"} for c in cells]
    )
    pert_rows, indices = [], []
    for c in cells:
        ex = {"input_ids": list(c), "length": len(c), "tokens_to_perturb": list(OE_TOKENS),
              "perturb_index": _perturb_index(c, OE_TOKENS)}
        indices.append(ex["perturb_index"])
        ex = isp.overexpress_tokens(ex) if perturb_type == "overexpress" else isp.delete_indices(ex)
        pert_rows.append({"input_ids": ex["input_ids"], "length": len(ex["input_ids"])})
    return original, Dataset.from_list(pert_rows), indices


def _run(perturb_type, forward_batch_size, legacy):
    original, perturbed, indices = _datasets(perturb_type)
    torch.manual_seed(0)
    goal = torch.rand(1, HIDDEN, device=isp.ISP_device)
    start = torch.rand(1, HIDDEN, device=isp.ISP_device)
    out = isp.quant_cos_sims(
        _StubModel(), perturb_type, perturbed, None, None, forward_batch_size, -1, original,
        list(OE_TOKENS), indices, True, STATES, {"somatic": start, "pluripotent": goal}, 0,
        MAX_LEN, 1, legacy_padding_mean=legacy,
    )
    return out["pluripotent"]


@unittest.skipIf(DEPS_MISSING, f"torch/datasets/geneformer unavailable: {DEPS_MISSING}")
class TestGroupGoalStatePadding(unittest.TestCase):
    def _check(self, perturb_type):
        reference = _run(perturb_type, 1, legacy=False)
        torch.testing.assert_close(_run(perturb_type, 1, legacy=True), reference)
        torch.testing.assert_close(_run(perturb_type, 8, legacy=False), reference)
        legacy = _run(perturb_type, 8, legacy=True)
        # The longest cells see no padding; the shorter ones do under the earlier pooling.
        self.assertFalse(torch.allclose(legacy, reference))

    def test_overexpress_is_batch_size_invariant(self):
        self._check("overexpress")

    def test_delete_is_batch_size_invariant(self):
        self._check("delete")

    def test_mean_lengths(self):
        orig, pert = torch.tensor([30, 22]), torch.tensor([30, 22])
        o, p = isp.group_goal_state_mean_lengths("overexpress", 3, orig, pert)
        self.assertEqual((o.tolist(), p.tolist()), ([27, 19], [27, 19]))
        o, p = isp.group_goal_state_mean_lengths("delete", 3, orig, pert - 3)
        self.assertEqual((o.tolist(), p.tolist()), ([27, 19], [27, 19]))
        o, p = isp.group_goal_state_mean_lengths("overexpress", 3, orig, pert, legacy_padding_mean=True)
        self.assertEqual((o.tolist(), p.tolist()), ([30, 22], [30, 22]))


if __name__ == "__main__":
    unittest.main()
