"""Goal-state ISP stats read (start, goal, alt_1, ..., alt_n) shift tuples for any number of alt states."""
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
    import pandas as pd

    from geneformer.in_silico_perturber_stats import isp_stats_to_goal_state

    DEPS_MISSING = None
except Exception as exc:  # pragma: no cover
    DEPS_MISSING = str(exc)


def _states(alt_states):
    return {"state_key": "disease", "start_state": "AD", "goal_state": "WT", "alt_states": alt_states}


def _genome_wide_input(n_alt):
    # gene 10 moves cells toward the goal; genes 11..13 do not
    dict_list = [{}, {}]
    for token, goal in ((10, 0.5), (11, 0.0), (12, 0.01), (13, -0.01)):
        for cell in range(6):
            shifts = (0.1, goal + cell * 0.001) + tuple(0.2 * (k + 1) + token + cell * 0.001 for k in range(n_alt))
            dict_list[cell % 2].setdefault((token, "cell_emb"), []).append(shifts)
    cos_sims_df = pd.DataFrame(
        {"Gene": [10, 11, 12, 13], "Gene_name": ["A", "B", "C", "D"], "Ensembl_ID": ["E10", "E11", "E12", "E13"]}
    )
    return cos_sims_df, dict_list


@unittest.skipIf(DEPS_MISSING is not None, f"deps missing: {DEPS_MISSING}")
class TestGoalStateStatsAltStates(unittest.TestCase):
    def test_genome_wide_without_alt_states(self):
        cos_sims_df, dict_list = _genome_wide_input(0)
        df = isp_stats_to_goal_state(cos_sims_df, dict_list, _states([]), "all")
        self.assertEqual(
            list(df.columns),
            ["Gene", "Gene_name", "Ensembl_ID", "Shift_to_goal_end", "Goal_end_vs_random_pval",
             "Goal_end_FDR", "N_Detections", "Sig"],
        )
        self.assertEqual(df.iloc[0]["Gene"], 10)

    def test_genome_wide_one_alt_state_keeps_unsuffixed_columns(self):
        cos_sims_df, dict_list = _genome_wide_input(1)
        df = isp_stats_to_goal_state(cos_sims_df, dict_list, _states(["MCI"]), "all")
        self.assertEqual(
            list(df.columns),
            ["Gene", "Gene_name", "Ensembl_ID", "Shift_to_goal_end", "Shift_to_alt_end",
             "Goal_end_vs_random_pval", "Alt_end_vs_random_pval", "Goal_end_FDR", "Alt_end_FDR",
             "N_Detections", "Sig"],
        )
        row = df.set_index("Gene").loc[11]
        self.assertAlmostEqual(row["Shift_to_alt_end"], 0.2 + 11 + 0.0025)

    def test_genome_wide_three_alt_states(self):
        cos_sims_df, dict_list = _genome_wide_input(3)
        df = isp_stats_to_goal_state(cos_sims_df, dict_list, _states(["MCI", "PD", "HD"]), "all")
        for state in ("MCI", "PD", "HD"):
            for col in (f"Shift_to_alt_end_{state}", f"Alt_end_{state}_vs_random_pval", f"Alt_end_{state}_FDR"):
                self.assertIn(col, df.columns)
        self.assertNotIn("Shift_to_alt_end", df.columns)
        row = df.set_index("Gene").loc[12]
        self.assertAlmostEqual(row["Shift_to_goal_end"], 0.01 + 0.0025)
        self.assertAlmostEqual(row["Shift_to_alt_end_MCI"], 0.2 + 12 + 0.0025)
        self.assertAlmostEqual(row["Shift_to_alt_end_HD"], 0.6 + 12 + 0.0025)

    def test_group_mode_three_alt_states(self):
        dict_list = [{((10, 11), "cell_emb"): [(0.1, 0.3, 1.0, 2.0, 3.0), (0.1, 0.4, 1.5, 2.5, 3.5)]}]
        cos_sims_df = pd.DataFrame({"Gene": [(10, 11)], "Gene_name": [("A", "B")], "Ensembl_ID": [("E10", "E11")]})
        df = isp_stats_to_goal_state(cos_sims_df, dict_list, _states(["MCI", "PD", "HD"]), [10, 11])
        self.assertEqual(
            list(df.columns),
            ["Shift_to_goal_end", "Shift_to_alt_end_MCI", "Shift_to_alt_end_PD", "Shift_to_alt_end_HD"],
        )
        self.assertEqual(df["Shift_to_goal_end"].tolist(), [0.4, 0.3])
        self.assertEqual(df["Shift_to_alt_end_HD"].tolist(), [3.5, 3.0])

    def test_tuple_length_mismatch_raises(self):
        cos_sims_df, dict_list = _genome_wide_input(1)
        with self.assertRaisesRegex(ValueError, "expected 4"):
            isp_stats_to_goal_state(cos_sims_df, dict_list, _states(["MCI", "PD"]), "all")


if __name__ == "__main__":
    unittest.main()
