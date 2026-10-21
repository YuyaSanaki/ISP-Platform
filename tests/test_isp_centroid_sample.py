"""State centroids: pooled max_ncells (default) or an opt-in per-state cap."""
from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "core", ROOT / "contracts"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

try:
    from datasets import Dataset

    from geneformer import in_silico_perturber as isp

    DEPS_MISSING = None
except Exception as exc:  # pragma: no cover
    DEPS_MISSING = str(exc)

STATES = {"state_key": "disease", "start_state": "dcm", "goal_state": "nf", "alt_states": []}


def _data():
    states = ["dcm"] * 90 + ["nf"] * 10
    return Dataset.from_dict(
        {"disease": states, "length": list(range(100)), "input_ids": [[1]] * 100}
    )


@unittest.skipIf(DEPS_MISSING is not None, f"deps missing: {DEPS_MISSING}")
class TestCentroidSample(unittest.TestCase):
    def test_default_matches_pooled_downsample_and_warns(self):
        data = _data()
        with self.assertLogs(isp.logger, level="WARNING") as logs:
            sample = isp.centroid_sample(data, STATES, 50, None, 1)
        expected = isp.downsample_and_sort(data, 50)
        self.assertEqual(sample["length"], expected["length"])
        self.assertEqual(Counter(sample["disease"]), {"dcm": 50})
        self.assertIn("max_ncells_per_state", "\n".join(logs.output))

    def test_per_state_cap_takes_cells_from_every_state(self):
        sample = isp.centroid_sample(_data(), STATES, 50, 8, 1)
        self.assertEqual(Counter(sample["disease"]), {"dcm": 8, "nf": 8})
        self.assertEqual(sample["length"], sorted(sample["length"], reverse=True))

    def test_perturber_accepts_option(self):
        p = isp.InSilicoPerturber(
            perturb_type="delete",
            genes_to_perturb="all",
            cell_states_to_model=STATES,
            max_ncells_per_state=100,
        )
        self.assertEqual(p.max_ncells_per_state, 100)
        with self.assertRaises(ValueError):
            isp.InSilicoPerturber(
                perturb_type="delete",
                genes_to_perturb="all",
                cell_states_to_model=STATES,
                max_ncells_per_state="all",
            )


if __name__ == "__main__":
    unittest.main()
