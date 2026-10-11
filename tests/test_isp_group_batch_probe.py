"""Auto forward_batch_size probes must match the memory peak of the scoring loop.

Group ISP clones one layer of the perturbed forward and frees the rest before the original
forward, so its probe keeps one layer between two forwards. State-feedback ISP holds both
forwards. Genome-wide ISP holds one.
"""
from __future__ import annotations

import sys
import unittest
import weakref
from pathlib import Path
from unittest import mock

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
MAX_LEN = 16


class _Outputs:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


def _counting_model():
    class _Model(torch.nn.Module):
        """Records how many earlier forward outputs are still alive at each call."""

        def __init__(self):
            super().__init__()
            self.bert = torch.nn.Module()
            self.bert.embeddings = torch.nn.Module()
            self.bert.embeddings.position_embeddings = torch.nn.Embedding(MAX_LEN, HIDDEN)
            self.live = weakref.WeakSet()
            self.live_at_call = []

        def forward(self, input_ids, attention_mask=None):
            self.live_at_call.append(len(self.live))
            embs = torch.zeros(*input_ids.shape, HIDDEN, device=input_ids.device)
            out = _Outputs([embs, embs + 1.0])
            self.live.add(out)
            return out

    return _Model()


def _probe(**kwargs):
    model = _counting_model()
    data = Dataset.from_list([{"input_ids": [5, 6, 7, 8], "length": 4}])

    def fake_resolve(value, *, default, probe, cache_fields, label):
        probe(3)
        return cache_fields

    with mock.patch.object(isp, "resolve_batch_size", side_effect=fake_resolve):
        cache_fields = isp.resolve_forward_batch_size("auto", model, data, 0, "model_dir", **kwargs)
    return model.live_at_call, cache_fields


@unittest.skipIf(DEPS_MISSING, f"torch/datasets/geneformer unavailable: {DEPS_MISSING}")
class TestForwardBatchProbe(unittest.TestCase):
    def test_single_forward(self):
        live, fields = _probe()
        self.assertEqual(live, [0])
        self.assertNotIn("keep_layer", fields)

    def test_dual_forward_holds_both_outputs(self):
        live, fields = _probe(n_forwards=2, task="rank_edit_dual_forward")
        self.assertEqual(live, [0, 1])
        self.assertNotIn("keep_layer", fields)

    def test_group_isp_keeps_one_layer_between_forwards(self):
        live, fields = _probe(n_forwards=2, keep_layer=1, task="isp_group_forward")
        self.assertEqual(live, [0, 0])
        self.assertEqual(fields["keep_layer"], 1)
        self.assertEqual(fields["task"], "isp_group_forward")


if __name__ == "__main__":
    unittest.main()
