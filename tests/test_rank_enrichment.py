"""ISP rank enrichment: pseudobulk ranks, gene-level Delta-spec, top fraction, offline ORA,
the postprocess switch and the normal ISP path."""
from __future__ import annotations

import gzip
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

import gene_set_enrichment as gse  # noqa: E402
import postprocess_switch as pps  # noqa: E402
from state_feedback import closeness as cl  # noqa: E402
from state_feedback import gene_ranks as gr  # noqa: E402


class PseudobulkRankTests(unittest.TestCase):
    def test_absent_gene_counts_as_bottom(self):
        pos, n = gr.position_sums([[10, 11, 12], [11, 10]])
        self.assertEqual(n, 2)
        rank = gr.pseudobulk_rank(pos, n)
        self.assertAlmostEqual(rank[10], (0.0 + 1.0) / 2)
        self.assertAlmostEqual(rank[11], (0.5 + 0.0) / 2)
        self.assertAlmostEqual(rank[12], (1.0 + 1.0) / 2)

    def test_mean_displacement_includes_dropped_and_inserted_genes(self):
        start, n = gr.position_sums([[1, 2, 3]])
        final, _ = gr.position_sums([[9, 1, 2]])
        d = gr.mean_displacement(start, final, n)
        self.assertAlmostEqual(d[1], 0.5)
        self.assertAlmostEqual(d[2], 0.5)
        self.assertAlmostEqual(d[3], 0.0)
        self.assertAlmostEqual(d[9], -1.0)


class DeltaSpecTests(unittest.TestCase):
    def test_placebo_with_the_gene_is_left_out(self):
        start, n = gr.position_sums([[1, 2, 3, 4]])
        conf, _ = gr.position_sums([[9, 3, 1, 2]])
        p1, _ = gr.position_sums([[8, 1, 2, 3]])
        p2, _ = gr.position_sums([[3, 1, 2, 4]])
        out = gr.gene_delta_spec(start, conf, {"p1": p1, "p2": p2}, {"p1": {8}, "p2": {3}}, n)
        self.assertEqual(int(out.loc[3, "n_placebos"]), 1)
        self.assertAlmostEqual(out.loc[3, "delta_placebo_mean"], 1.0 - 2 / 3)
        self.assertAlmostEqual(out.loc[3, "delta_configured"], 1 / 3 - 2 / 3)
        self.assertAlmostEqual(out.loc[3, "delta_spec"], (1 / 3 - 2 / 3) - (1 - 2 / 3))
        self.assertEqual(int(out.loc[1, "n_placebos"]), 2)


class BackgroundAndTopTests(unittest.TestCase):
    def test_background_rules_and_exclusion(self):
        start = pd.DataFrame({"token": [1, 2, 3], "sum_norm_pos": [0, 0, 0], "n_cells": [5, 5, 1]})
        goal = pd.DataFrame({"token": [2, 3, 4], "sum_norm_pos": [0, 0, 0], "n_cells": [5, 5, 5]})
        self.assertEqual(gr.background_tokens(start, goal, min_detection=5), [2])
        self.assertEqual(
            gr.background_tokens(start, goal, min_detection=5, rule="either", exclude=[4]), [1, 2, 3]
        )
        with self.assertRaises(ValueError):
            gr.background_tokens(start, goal, min_detection=5, rule="any")

    def test_top_fraction_ceil_and_tie_break_by_token(self):
        s = pd.Series([0.3, 0.1, 0.1, 0.9, 0.5], index=pd.Index([5, 4, 2, 1, 3], name="token"))
        self.assertEqual(gr.top_fraction(s, 0.1), [2])
        self.assertEqual(gr.top_fraction(s, 0.5), [2, 4, 5])
        with self.assertRaises(ValueError):
            gr.top_fraction(s, 0.0)


class OraTests(unittest.TestCase):
    def test_matches_fisher_exact_and_uppercases_symbols(self):
        from scipy.stats import fisher_exact

        bg = [f"G{i}" for i in range(100)]
        sets = {"T1": {f"G{i}" for i in range(10)}, "T2": {"G50", "G51", "X"}, "T3": {"Y"}}
        query = ["g0", "g1", "g2", "G50", "G99", "NOT_IN_BG"]
        res = gse.ora(query, bg, sets).set_index("Term")
        self.assertEqual(set(res.index), {"T1", "T2"})
        _, p = fisher_exact([[3, 2], [7, 88]], alternative="greater")
        self.assertAlmostEqual(res.loc["T1", "P-value"], p, places=12)
        self.assertEqual(res.loc["T1", "Overlap"], "3/10")
        self.assertEqual(res.loc["T2", "n_term"], 2)
        self.assertEqual(res.loc["T1", "Genes"], "G0;G1;G2")
        self.assertEqual(int(res.loc["T1", "n_query"]), 5)

    def test_bh_adjust(self):
        adj = gse.bh_adjust([0.01, 0.04, 0.03])
        self.assertAlmostEqual(adj[0], 0.03)
        self.assertAlmostEqual(adj[1], 0.04)
        self.assertAlmostEqual(adj[2], 0.04)

    def test_read_gmt_skips_empty_description(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "lib.gmt"
            p.write_text("Term A\t\tabc1\tABC2\nTerm B\tdesc\tX\n")
            self.assertEqual(gse.read_gmt(p), {"Term A": {"ABC1", "ABC2"}, "Term B": {"X"}})


def _write_run(run: Path) -> None:
    src = run / "rank_enrichment"
    src.mkdir(parents=True)
    start_rows = [[1, 2, 3, 4, 5, 6, 7, 8], [2, 1, 3, 5, 4, 6, 8, 7]]
    goal_rows = [[8, 7, 6, 5, 4, 3, 2, 1], [7, 8, 6, 4, 5, 3, 1, 2]]
    for name, rows in (("start", start_rows), ("goal", goal_rows)):
        pos, _ = gr.position_sums(rows)
        pos.to_csv(src / f"{name}_positions.csv.gz", index=False)
    pd.DataFrame({"token": range(1, 12), "symbol": [f"g{i}" for i in range(1, 12)]}).to_csv(
        src / "genes.csv.gz", index=False)
    chain_tokens = {"configured": [9], "random_s0_0": [10], "random_s0_1": [11]}
    finals = {
        "configured": [[9, 8, 7, 1, 2, 3, 4, 5], [9, 7, 8, 2, 1, 3, 5, 4]],
        "random_s0_0": [[10, 1, 2, 3, 4, 5, 6, 7], [10, 2, 1, 3, 5, 4, 6, 8]],
        "random_s0_1": [[11, 2, 1, 3, 4, 5, 6, 7], [11, 1, 2, 3, 4, 6, 5, 8]],
    }
    meta = {
        "n_start_cells": 2, "n_goal_cells": 2, "goal_state": "goal",
        "species_model": "mouse_geneformer", "configured": "configured",
        "configured_label": "g9", "placebos": ["random_s0_0", "random_s0_1"],
    }
    (src / "chains.json").write_text(json.dumps({**meta, "chain_tokens": chain_tokens}))
    seed_dir = run / "seed0"
    seed_dir.mkdir()
    frames = [gr.position_sums(rows)[0].assign(chain=c) for c, rows in finals.items()]
    pd.concat(frames).to_csv(seed_dir / "rank_enrichment_positions.csv.gz", index=False)


def _gene_set_root(root: Path) -> None:
    gs_dir = root / "core" / "geneformer" / "dicts" / "gene_sets"
    gs_dir.mkdir(parents=True)
    (gs_dir / "LibA.gmt").write_text("Late genes\t\tG7\tG8\tG6\nEarly genes\t\tG1\tG2\n")
    (gs_dir / "libraries.json").write_text(json.dumps({
        "libraries": {"LibA": {"label": "Library A"}},
        "default_by_species": {"human": ["LibA"], "mouse": ["LibA"]},
    }))


def _run_with_gene_sets(root: Path, *args, **kwargs):
    import run_isp_rank_enrichment as rre

    old = os.environ.get("GENEFORMER_ROOT")
    os.environ["GENEFORMER_ROOT"] = str(root)
    try:
        return rre.run(*args, **kwargs)
    finally:
        if old is None:
            os.environ.pop("GENEFORMER_ROOT")
        else:
            os.environ["GENEFORMER_ROOT"] = old


class RunTests(unittest.TestCase):
    def test_run_writes_tables_figures_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _gene_set_root(root)
            run = root / "run"
            _write_run(run)
            summary = _run_with_gene_sets(root, run, {"top_fraction": 0.3, "min_detection": 1})
            out = run / "rank_enrichment"
            self.assertEqual(summary["n_background"], 8)
            self.assertEqual(summary["n_top"], {"start": 3, "goal": 3, "isp": 3, "dspec": 3})
            self.assertEqual(summary["species_vocabulary"], "mouse")
            self.assertEqual(summary["perturbed_genes"], ["g9"])
            ranks = pd.read_csv(out / "gene_ranks.csv", index_col="token")
            self.assertNotIn(9, ranks.index)
            top = pd.read_csv(out / "top_genes.csv")
            self.assertEqual(set(top.loc[top["list"] == "goal", "symbol"]), {"g6", "g7", "g8"})
            self.assertTrue(set(top.loc[top["list"] == "isp", "symbol"]) & {"g7", "g8"})
            dspec = top.loc[top["list"] == "dspec"].set_index("token")["score"]
            self.assertTrue(np.allclose(dspec, ranks.loc[dspec.index, "delta_spec"]))
            self.assertLessEqual(dspec.max(), ranks["delta_spec"].drop(dspec.index).min())
            self.assertTrue(set(dspec.index) & {7, 8})
            for name in ("start", "goal", "isp", "dspec"):
                self.assertTrue((out / "enrichment" / name / "LibA.csv").exists())
            self.assertTrue((out / "figures" / "enrich_liba.png").exists())
            self.assertTrue((out / "figures" / "enrich_liba.pdf").exists())
            self.assertTrue((out / "figures" / "closeness.png").exists())
            close = summary["closeness"]
            self.assertGreater(close["gene"]["progress"]["value"], 0)
            self.assertEqual(close["gene"]["progress"]["n_placebos"], 2)
            self.assertIn("by_seed", close["gene"]["progress"])
            self.assertIn("dspec_neglog10p_cosine", close["ora:LibA"])
            self.assertTrue((out / "closeness" / "closeness_summary.csv").exists())
            self.assertEqual(summary["mode"], "state_feedback")

    def test_normal_isp_lists_and_tested_universe(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _gene_set_root(root)
            run = root / "run"
            out = run / "rank_enrichment"
            out.mkdir(parents=True)
            start_rows = [[1, 2, 3, 4, 5, 6, 7, 8], [2, 1, 3, 5, 4, 6, 8, 7]]
            goal_rows = [[8, 7, 6, 5, 4, 3, 2, 1], [7, 8, 6, 4, 5, 3, 1, 2]]
            for name, rows in (("start", start_rows), ("goal", goal_rows)):
                gr.position_sums(rows)[0].to_csv(out / f"{name}_positions.csv.gz", index=False)
            pd.DataFrame({"token": range(1, 9), "symbol": [f"g{i}" for i in range(1, 9)]}).to_csv(
                out / "genes.csv.gz", index=False)
            (out / "n_cells.json").write_text(json.dumps({"start": 2, "goal": 2}))
            stats = run / "ispstats_results"
            stats.mkdir()
            pd.DataFrame({
                "Gene": [6, 7, 8, 1, 2],
                "Gene_name": ["g6", "g7", "g8", "g1", "g2"],
                "Shift_to_goal_end": [0.2, 0.3, 0.1, -0.1, 0.4],
                "Goal_end_FDR": [0.01, 0.01, 0.01, 0.01, 0.5],
                "N_Detections": [10] * 5,
            }).to_csv(stats / "isp_stats.csv", index=False)
            cfg = {"perturbation": {"type": "overexpress", "start_state": "s", "end_state": "goal"}}
            summary = _run_with_gene_sets(root, run, {"top_fraction": 0.3, "min_detection": 1},
                                          species="mouse_geneformer", isp_config=cfg)
            self.assertEqual(summary["mode"], "isp")
            self.assertEqual(summary["n_tested_in_background"], 5)
            self.assertEqual(summary["n_isp_hits_in_background"], 3)
            top = pd.read_csv(out / "top_genes.csv")
            self.assertEqual(top.loc[top["list"] == "isp", "symbol"].tolist(), ["g7", "g6", "g8"])
            isp = pd.read_csv(out / "enrichment" / "isp" / "LibA.csv").set_index("Term")
            self.assertEqual(int(isp.loc["Late genes", "n_background"]), 5)
            self.assertIn("figures/enrich_liba.png", summary["figures"])

    def test_normal_isp_without_per_gene_stats_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / "ispstats_results").mkdir()
            pd.DataFrame({"Cell": [0], "Shift": [0.1]}).to_csv(
                run / "ispstats_results" / "cells.csv", index=False)
            self.assertIsNone(_run_with_gene_sets(run, run, {}, isp_config={}))


class PostprocessSwitchTests(unittest.TestCase):
    def test_one_switch_with_rank_enrichment_sub_toggle(self):
        self.assertFalse(pps.rank_enrichment_enabled({}))
        self.assertFalse(pps.rank_enrichment_enabled({"postprocess": {"enabled": False}}))
        self.assertTrue(pps.rank_enrichment_enabled({"postprocess": {"enabled": True}}))
        self.assertFalse(pps.rank_enrichment_enabled(
            {"postprocess": {"enabled": True, "rank_enrichment": False}}))

    def test_legacy_state_feedback_switch(self):
        cfg = {"state_feedback": {"rank_enrichment": {"enabled": True}}}
        self.assertTrue(pps.rank_enrichment_enabled(cfg))
        cfg["postprocess"] = {"enabled": False}
        self.assertFalse(pps.rank_enrichment_enabled(cfg))

    def test_settings_layering(self):
        s = pps.rank_enrichment_settings({
            "rank_enrichment": {"top_fraction": 0.2, "fdr": None},
            "state_feedback": {"rank_enrichment": {"enabled": True, "top_terms": 5}},
        })
        self.assertEqual(s["top_fraction"], 0.2)
        self.assertEqual(s["fdr"], 0.05)
        self.assertEqual(s["top_terms"], 5)
        self.assertNotIn("enabled", s)

    def test_cli_flags(self):
        import argparse

        parser = argparse.ArgumentParser()
        pps.add_postprocess_args(parser)
        cfg = pps.apply_postprocess_args({"postprocess": {"enabled": False, "n_clusters": 4}},
                                         parser.parse_args(["--enable-postprocess"]))
        self.assertEqual(cfg["postprocess"], {"enabled": True, "n_clusters": 4})
        cfg = pps.apply_postprocess_args(cfg, parser.parse_args(["--no-rank-enrichment"]))
        self.assertFalse(pps.rank_enrichment_enabled(cfg))
        self.assertTrue(cfg["postprocess"]["enabled"])
        untouched = {"postprocess": {"enabled": True}}
        self.assertIs(pps.apply_postprocess_args(untouched, parser.parse_args([])), untouched)


class ClosenessTests(unittest.TestCase):
    def test_vector_metrics_of_half_way(self):
        G = np.array([-0.4, -0.2, 0.1, 0.3, 0.0])
        m = cl.vector_metrics(0.5 * G, G)
        self.assertAlmostEqual(m["progress"], 0.5)
        self.assertAlmostEqual(m["cosine"], 1.0)
        self.assertAlmostEqual(m["distance_closed"], 0.5)
        self.assertAlmostEqual(m["spearman"], 1.0)
        self.assertEqual(m["toward_goal"], 1.0)
        self.assertAlmostEqual(cl.vector_metrics(-G, G)["progress"], -1.0)

    def test_orthogonal_change_lowers_distance_closed_only(self):
        G = np.array([1.0, 0.0])
        m = cl.vector_metrics(np.array([0.0, 0.5]), G)
        self.assertAlmostEqual(m["progress"], 0.0)
        self.assertLess(m["distance_closed"], 0.0)

    def test_auc_goal_gain(self):
        G = np.linspace(-1, 1, 20)
        self.assertEqual(cl.auc_goal_gain(G, G), 1.0)
        self.assertEqual(cl.auc_goal_gain(-G, G), 0.0)

    def test_placebo_contrast_leaves_out_and_zeroes_own_genes(self):
        disp = pd.DataFrame(
            {"q": [-0.3, 0.1, 0.0], "a": [0.1, -0.5, 0.2], "b": [0.3, 0.1, -0.6]},
            index=pd.Index([1, 2, 3], name="token"),
        )
        query, pseudo = cl.query_and_placebo_deltas(disp, "q", {"q": [], "a": [2], "b": [3]})
        self.assertAlmostEqual(query[1], -0.3 - 0.2)
        self.assertAlmostEqual(query[2], 0.1 - 0.1)
        self.assertAlmostEqual(query[3], 0.0 - 0.2)
        self.assertAlmostEqual(pseudo.loc[1, "a"], 0.1 - 0.3)
        self.assertEqual(pseudo.loc[2, "a"], 0.0)
        self.assertEqual(pseudo.loc[3, "b"], 0.0)
        self.assertTrue(np.isnan(cl._contrast(disp, "b", ["a"], {"a": [2]})[2]))

    def test_compare_to_placebos(self):
        e = cl.compare_to_placebos(0.3, [0.0, 0.1, 0.2, 0.4])
        self.assertAlmostEqual(e["p_empirical"], 2 / 5)
        self.assertAlmostEqual(e["placebo_mean"], 0.175)
        self.assertGreater(e["z"], 0)

    def test_set_members_and_means(self):
        members = cl.set_members([1, 2, 3], {1: "a", 2: "B", 3: "c"},
                                 {"T1": {"A", "B"}, "T2": {"C"}}, min_size=2)
        self.assertEqual(list(members), ["T1"])
        self.assertEqual(cl.set_means(np.array([1.0, 3.0, 5.0]), members).tolist(), [2.0])


if __name__ == "__main__":
    unittest.main()
