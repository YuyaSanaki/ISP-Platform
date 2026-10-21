"""ISP stops with a ValueError naming the cause when no cell can be perturbed."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "core", ROOT / "contracts"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

try:
    from datasets import Dataset

    import geneformer.in_silico_perturber as isp

    DEPS_MISSING = None
except Exception as exc:  # pragma: no cover
    DEPS_MISSING = str(exc)


@unittest.skipIf(DEPS_MISSING is not None, f"deps missing: {DEPS_MISSING}")
class TestNoMatchingCells(unittest.TestCase):
    def _perturber(self, **kwargs):
        gene = next(
            g
            for g in isp.InSilicoPerturber(genes_to_perturb="all", nproc=1).gene_token_dict
            if not g.startswith("<")
        )
        return isp.InSilicoPerturber(
            perturb_type="delete", genes_to_perturb=[gene], nproc=1, **kwargs
        )

    def test_no_cell_contains_genes_to_perturb(self):
        perturber = self._perturber()
        token = perturber.tokens_to_perturb[0]
        data = Dataset.from_dict({"input_ids": [[token + 1, token + 2]], "length": [2]})
        with mock.patch.object(isp, "get_model_input_size", return_value=2048):
            with self.assertRaisesRegex(ValueError, "No cells in dataset contain all genes"):
                perturber.in_silico_perturb(None, data, 0, None, "out/", "prefix")

    def test_filter_data_leaves_no_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "d.dataset")
            Dataset.from_dict({"input_ids": [[1, 2]], "disease": ["AD"]}).save_to_disk(path)
            with self.assertRaisesRegex(ValueError, "No cells remain after filtering"):
                isp.load_and_filter({"disease": ["WT"]}, 1, path)

    def test_invalid_option_is_value_error(self):
        with self.assertRaisesRegex(ValueError, "cell_inds_to_perturb must be positive"):
            self._perturber(cell_inds_to_perturb={"start": -1, "end": 1})


if __name__ == "__main__":
    unittest.main()
