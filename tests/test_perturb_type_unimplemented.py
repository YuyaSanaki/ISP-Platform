"""inhibit / activate are not implemented and must stop before any ISP work."""
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
    from geneformer.in_silico_perturber import (
        InSilicoPerturber,
        check_perturb_type_implemented,
    )

    DEPS_MISSING = None
except Exception as exc:  # pragma: no cover
    DEPS_MISSING = str(exc)


@unittest.skipIf(DEPS_MISSING is not None, f"deps missing: {DEPS_MISSING}")
class TestPerturbTypeUnimplemented(unittest.TestCase):
    def test_check_rejects_inhibit_and_activate(self):
        for perturb_type in ("inhibit", "activate"):
            with self.assertRaisesRegex(ValueError, "not implemented"):
                check_perturb_type_implemented(perturb_type)

    def test_check_accepts_delete_and_overexpress(self):
        for perturb_type in ("delete", "overexpress"):
            check_perturb_type_implemented(perturb_type)

    def test_perturber_rejects_activate_even_with_direct_shift(self):
        with self.assertRaisesRegex(ValueError, "not implemented"):
            InSilicoPerturber(
                perturb_type="activate",
                perturb_rank_direct_shift=1,
                genes_to_perturb=["ENSG00000204531"],
            )


if __name__ == "__main__":
    unittest.main()
