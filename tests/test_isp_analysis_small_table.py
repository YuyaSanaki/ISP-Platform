"""ISP figures must work for a targeted list shorter than the top-25 bar plot."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "core", ROOT / "contracts"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

try:
    import matplotlib

    matplotlib.use("Agg")
    import pandas as pd

    from isp_analysis import run_isp_figure_analysis

    DEPS_MISSING = None
except Exception as exc:  # pragma: no cover
    DEPS_MISSING = str(exc)


@unittest.skipIf(DEPS_MISSING is not None, f"deps missing: {DEPS_MISSING}")
class TestIspAnalysisSmallTable(unittest.TestCase):
    def test_three_gene_table_writes_bar_plot(self):
        df = pd.DataFrame(
            {
                "Gene": [10, 11, 12],
                "Gene_name": ["A", "B", "C"],
                "Ensembl_ID": ["E10", "E11", "E12"],
                "Shift_to_goal_end": [0.02, -0.01, 0.001],
                "Goal_end_vs_random_pval": [0.001, 0.2, 0.8],
                "Goal_end_FDR": [0.003, 0.3, 0.8],
                "N_Detections": [30, 25, 40],
                "Sig": [1, 0, 0],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            parquet = tmp / "stats.parquet"
            df.to_parquet(parquet)
            run_isp_figure_analysis(parquet, tmp / "figures", tmp / "stats")
            self.assertTrue((tmp / "figures" / "top_genes_barplot.png").is_file())


if __name__ == "__main__":
    unittest.main()
