"""P2 fixes: finetune warmup, ortholog tables, cross-species remap coverage."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "core", ROOT / "contracts", ROOT / "webui"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


def _load_gene_converter():
    geneformer_pkg = types.ModuleType("geneformer")
    backends_pkg = types.ModuleType("geneformer.backends")
    sys.modules["geneformer"] = geneformer_pkg
    sys.modules["geneformer.backends"] = backends_pkg
    spec = importlib.util.spec_from_file_location(
        "geneformer.backends.registry", ROOT / "core" / "geneformer" / "backends" / "registry.py"
    )
    registry = importlib.util.module_from_spec(spec)
    sys.modules["geneformer.backends.registry"] = registry
    spec.loader.exec_module(registry)
    backends_pkg.registry = registry
    for name in (
        "BackendSpec",
        "HumanVariant",
        "ModelId",
        "MouseVariant",
        "Organism",
        "get_backend",
        "native_organism",
        "needs_gene_conversion",
        "parse_species_config",
        "resolve_pretrained_path",
    ):
        setattr(backends_pkg, name, getattr(registry, name))
    gc_spec = importlib.util.spec_from_file_location(
        "geneformer.gene_converter", ROOT / "core" / "geneformer" / "gene_converter.py"
    )
    gc = importlib.util.module_from_spec(gc_spec)
    sys.modules["geneformer.gene_converter"] = gc
    gc_spec.loader.exec_module(gc)
    return gc


def _load_training_schedule():
    spec = importlib.util.spec_from_file_location(
        "geneformer.training_schedule", ROOT / "core" / "geneformer" / "training_schedule.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


training_schedule = _load_training_schedule()
gene_converter = _load_gene_converter()


class TestFinetuneWarmup(unittest.TestCase):
    def test_warmup_ratio_caps_long_default(self):
        warmup = training_schedule.resolve_warmup_steps(
            {"warmup_ratio": 0.05, "warmup_steps": 500},
            steps_per_epoch=7106,
            num_epochs=5,
        )
        self.assertLessEqual(warmup, 3553)
        self.assertGreater(warmup, 0)

    def test_smoke_run_warmup_is_small(self):
        warmup = training_schedule.resolve_warmup_steps(
            {"warmup_ratio": 0.05},
            steps_per_epoch=7106,
            num_epochs=1,
        )
        self.assertLessEqual(warmup, 710)
        self.assertGreater(warmup, 0)

    def test_default_warmup_steps_500_on_long_run(self):
        warmup = training_schedule.resolve_warmup_steps(
            {"warmup_ratio": None, "warmup_steps": 500},
            steps_per_epoch=4728,
            num_epochs=10,
        )
        self.assertEqual(warmup, 500)

    def test_ratio_overrides_steps_on_long_run(self):
        warmup = training_schedule.resolve_warmup_steps(
            {"warmup_ratio": 0.05, "warmup_steps": 500},
            steps_per_epoch=4728,
            num_epochs=10,
        )
        self.assertEqual(warmup, int(4728 * 10 * 0.05))

    def test_fixed_500_caps_on_short_run(self):
        warmup = training_schedule.resolve_warmup_steps(
            {"warmup_steps": 500},
            steps_per_epoch=40,
            num_epochs=1,
        )
        self.assertLessEqual(warmup, 4)

    def test_warmup_never_exceeds_total_minus_one(self):
        warmup = training_schedule.resolve_warmup_steps(
            {"warmup_steps": 9999},
            steps_per_epoch=10,
            num_epochs=1,
        )
        self.assertLessEqual(warmup, 9)

    def test_dataloader_workers_default_zero(self):
        self.assertEqual(training_schedule.resolve_dataloader_num_workers({}), 0)

    def test_dataloader_workers_override(self):
        self.assertEqual(
            training_schedule.resolve_dataloader_num_workers({"dataloader_num_workers": 4}),
            4,
        )


class TestOrthologTable(unittest.TestCase):
    def setUp(self):
        gene_converter._TABLE_CACHE.clear()

    def test_curated_symbol_igfbp2(self):
        result = gene_converter.convert_gene_ids(
            ["Igfbp2"],
            model_organism="mouse",
            model_id="human_geneformer",
        )
        self.assertEqual(result.mapped["Igfbp2"], "ENSG00000115457")

    def test_curated_does_not_remap_unrelated_ensembl_ids(self):
        """Production curated must not override BioMart for Gnai3/Lypla1/Maoa."""
        mouse_human = {
            "model_organism": "mouse",
            "model": "human_geneformer",
            "human_variant": "v2_104m",
            "ortholog_policy": "one2one",
        }
        human_mouse = {
            "model_organism": "human",
            "model": "mouse_geneformer",
            "ortholog_policy": "one2one",
        }
        # BioMart-correct pairs (gene identity checked against dicts).
        self.assertEqual(
            gene_converter.resolve_gene_for_model("ENSMUSG00000000001", mouse_human),
            "ENSG00000065135",  # Gnai3 → GNAI3
        )
        self.assertEqual(
            gene_converter.resolve_gene_for_model("ENSMUSG00000025903", mouse_human),
            "ENSG00000120992",  # Lypla1 → LYPLA1
        )
        self.assertEqual(
            gene_converter.resolve_gene_for_model("ENSMUSG00000025037", mouse_human),
            "ENSG00000189221",  # Maoa → MAOA
        )
        self.assertEqual(
            gene_converter.resolve_gene_for_model("ENSMUSG00000059552", mouse_human),
            "ENSG00000141510",  # Trp53 → TP53
        )
        self.assertEqual(
            gene_converter.resolve_gene_for_model("ENSMUSG00000041147", mouse_human),
            "ENSG00000139618",  # Brca2 → BRCA2
        )
        self.assertEqual(
            gene_converter.resolve_gene_for_model("Tp53", human_mouse),
            "ENSMUSG00000059552",
        )
        self.assertEqual(
            gene_converter.resolve_gene_for_model("Brca2", human_mouse),
            "ENSMUSG00000041147",
        )
        self.assertEqual(
            gene_converter.resolve_gene_for_model("Trp53", mouse_human),
            "ENSG00000141510",
        )

    def test_gapdh_curated_avoids_pseudogene(self):
        """GAPDH must resolve to Gapdh, not Gm10358; mouse Gapdh must not drop."""
        mouse_human = {
            "model_organism": "mouse",
            "model": "human_geneformer",
            "human_variant": "v2_104m",
            "ortholog_policy": "one2one",
        }
        human_mouse = {
            "model_organism": "human",
            "model": "mouse_geneformer",
            "ortholog_policy": "one2one",
        }
        mouse_gapdh = "ENSMUSG00000057666"
        human_gapdh = "ENSG00000111640"
        self.assertEqual(
            gene_converter.resolve_gene_for_model(mouse_gapdh, mouse_human),
            human_gapdh,
        )
        self.assertEqual(
            gene_converter.resolve_gene_for_model("Gapdh", mouse_human),
            human_gapdh,
        )
        self.assertEqual(
            gene_converter.resolve_gene_for_model(human_gapdh, human_mouse),
            mouse_gapdh,
        )
        self.assertEqual(
            gene_converter.resolve_gene_for_model("GAPDH", human_mouse),
            mouse_gapdh,
        )
        self.assertNotEqual(
            gene_converter.resolve_gene_for_model(human_gapdh, human_mouse),
            "ENSMUSG00000110469",  # Gm10358
        )

    def test_main_table_has_sufficient_ensembl_pairs(self):
        table = gene_converter.load_ortholog_table(gene_converter.ConversionPair.MOUSE_TO_HUMAN)
        ensembl_pairs = sum(1 for k in table if k.startswith("ENSMUSG"))
        self.assertGreater(ensembl_pairs, 1000)

    def test_orthology_type_column_populated(self):
        path = (
            ROOT
            / "core"
            / "geneformer"
            / "dicts"
            / "orthologs"
            / "mouse_to_human.tsv"
        )
        text = path.read_text(encoding="utf-8").splitlines()
        self.assertIn("orthology_type", text[0].split("\t"))
        self.assertTrue(any("\tortholog_one2one" in line for line in text[1:200]))

    @unittest.skipUnless(
        importlib.util.find_spec("anndata") and importlib.util.find_spec("scipy"),
        "anndata/scipy not installed",
    )
    def test_remap_retains_most_genes_with_full_table(self):
        import anndata as ad
        import numpy as np

        table = gene_converter.load_ortholog_table(gene_converter.ConversionPair.MOUSE_TO_HUMAN)
        mouse_ids = [k for k in table if k.startswith("ENSMUSG")][:200]
        if len(mouse_ids) < 50:
            self.skipTest("full ortholog table not present")

        X = np.random.rand(4, len(mouse_ids)).astype(np.float32)
        var = ad.AnnData(np.zeros((4, len(mouse_ids)))).var
        var["ensembl_id"] = mouse_ids
        adata = ad.AnnData(X=X, var=var)
        adata.obs["n_counts"] = [1000.0] * 4

        out, result = gene_converter.remap_adata_ensembl_ids(
            adata,
            {
                "model_organism": "mouse",
                "model": "human_geneformer",
                "human_variant": "v2_104m",
            },
        )
        self.assertGreater(out.n_vars, 50)
        self.assertGreater(result.kept_count, 50)
        self.assertTrue(all(str(x).startswith("ENSG") for x in out.var["ensembl_id"]))


class TestOrthologDownloadScript(unittest.TestCase):
    def test_mouse_script_exists_and_query_file_heredoc(self):
        script = ROOT / "scripts" / "download_mouse_human_orthologs.sh"
        self.assertTrue(script.is_file())
        text = script.read_text(encoding="utf-8")
        self.assertIn("--data-urlencode", text)
        self.assertIn("mmusculus_gene_ensembl", text)
        self.assertIn("hsapiens_homolog_orthology_type", text)
        self.assertIn("mouse_to_human_curated.tsv", text)
        self.assertIn("ortholog_policy", text)

    def test_fly_script_exists_and_biomart_query(self):
        script = ROOT / "scripts" / "download_drosophila_orthologs.sh"
        self.assertTrue(script.is_file())
        text = script.read_text(encoding="utf-8")
        self.assertIn("dmelanogaster_gene_ensembl", text)
        self.assertIn("drosophila_to_human.tsv", text)
        self.assertIn("drosophila_to_mouse.tsv", text)
        self.assertIn("fly_symbol_to_fbgn.tsv", text)
        self.assertIn("hsapiens_homolog_orthology_type", text)
        self.assertIn("mmusculus_homolog_orthology_type", text)
        self.assertIn("ortholog_policy", text)


class TestFlyOrthologTables(unittest.TestCase):
    def setUp(self):
        gene_converter._TABLE_CACHE.clear()

    def test_fly_tables_have_sufficient_pairs_when_downloaded(self):
        ortho_dir = ROOT / "core" / "geneformer" / "dicts" / "orthologs"
        human_path = ortho_dir / "drosophila_to_human.tsv"
        mouse_path = ortho_dir / "drosophila_to_mouse.tsv"
        if not human_path.is_file() or not mouse_path.is_file():
            self.skipTest(
                "fly ortholog tables missing; run scripts/download_drosophila_orthologs.sh"
            )
        # Empty / header-only stubs should skip (not fail CI on fresh checkouts).
        with human_path.open(encoding="utf-8") as fh:
            human_lines = sum(1 for line in fh if line.strip())
        with mouse_path.open(encoding="utf-8") as fh:
            mouse_lines = sum(1 for line in fh if line.strip())
        if human_lines < 50 or mouse_lines < 50:
            self.skipTest(
                "fly ortholog tables are empty stubs; run scripts/download_drosophila_orthologs.sh"
            )

        human_table = gene_converter.load_ortholog_table(gene_converter.ConversionPair.DROSOPHILA_TO_HUMAN)
        mouse_table = gene_converter.load_ortholog_table(gene_converter.ConversionPair.DROSOPHILA_TO_MOUSE)
        fly_ids = [k for k in human_table if k.startswith("FBgn")]
        self.assertGreater(len(fly_ids), 500)
        self.assertGreater(len(mouse_table), 500)

    def test_load_fly_symbol_table(self):
        table = gene_converter.load_fly_symbol_table()
        self.assertEqual(table.get("Tp53"), "FBgn0039044")
        self.assertEqual(table.get("p53"), "FBgn0039044")
        self.assertEqual(table.get("Brca2"), "FBgn0050169")
        self.assertNotIn("Igfbp2", table)

    def test_fly_symbols_do_not_map_to_unrelated_genes(self):
        """P0: Tp53 must not be hh/DHH; SF2 FBgn must not be forced to IGFBP2."""
        fly_human = {
            "model_organism": "drosophila",
            "model": "human_geneformer",
            "human_variant": "v2_104m",
            "ortholog_policy": "one2one",
        }
        fly_mouse = {
            "model_organism": "drosophila",
            "model": "mouse_geneformer",
            "ortholog_policy": "one2one",
        }
        symbols = gene_converter.load_fly_symbol_table()
        # Tp53 → true p53 FBgn (not hh). BioMart labels the family as one2many, so
        # one2one drops rather than picking TP53/TP63/TP73 arbitrarily.
        self.assertEqual(symbols.get("Tp53"), "FBgn0039044")
        self.assertIsNone(
            gene_converter.resolve_gene_for_model("Tp53", fly_human, input_symbol_table=symbols)
        )
        self.assertIsNone(
            gene_converter.resolve_gene_for_model("Tp53", fly_mouse, input_symbol_table=symbols)
        )
        self.assertNotEqual(
            gene_converter.resolve_gene_for_model("FBgn0004644", fly_human),
            gene_converter.resolve_gene_for_model("Tp53", fly_human, input_symbol_table=symbols),
        )
        # hh still maps to DHH; Tp53 must not.
        self.assertEqual(
            gene_converter.resolve_gene_for_model("FBgn0004644", fly_human),
            "ENSG00000139549",
        )
        # Former smoke alias FBgn0283477 is SF2 — must not resolve to IGFBP2.
        self.assertIsNone(
            gene_converter.resolve_gene_for_model("FBgn0283477", fly_human, input_symbol_table=symbols)
        )
        self.assertIsNone(
            gene_converter.resolve_gene_for_model("Igfbp2", fly_human, input_symbol_table=symbols)
        )
        # Brca2 uses current FlyBase ID (may lack BioMart pair → None is OK).
        self.assertEqual(symbols.get("Brca2"), "FBgn0050169")


def _purge_import_stubs() -> None:
    """Clear ModuleType stubs so real geneformer/torch/datasets imports work."""
    for key in list(sys.modules):
        if key == "geneformer" or key.startswith("geneformer."):
            del sys.modules[key]
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None and getattr(torch_mod, "__file__", None) is None:
        for key in list(sys.modules):
            if key == "torch" or key.startswith("torch."):
                del sys.modules[key]
        for key in list(sys.modules):
            if key == "datasets" or key.startswith("datasets."):
                del sys.modules[key]


class TestIspUmapIntegration(unittest.TestCase):
    def setUp(self):
        _purge_import_stubs()

    def test_should_run_only_for_targeted_genes(self):
        from run_isp_umap import should_run_isp_umap

        self.assertFalse(should_run_isp_umap([], {"umap": {"enabled": True}}))
        self.assertTrue(should_run_isp_umap(["Igfbp2"], {"umap": {"enabled": True}}))
        self.assertFalse(should_run_isp_umap(["Igfbp2"], {"umap": {"enabled": False}}))

    def test_build_isp_umap_config_from_isp_stage(self):
        from run_isp_umap import build_isp_umap_config

        isp_cfg = {
            "paths": {
                "dataset": "/app/data/1w/tokenized/1w_0.dataset",
                "geneformer_model": "/app/output/finetune/all_run1",
            },
            "species": {"model_organism": "mouse", "model": "human_geneformer"},
            "perturbation": {
                "type": "overexpress",
                "state_key": "disease",
                "start_state": "AD",
                "end_state": "WT",
            },
            "model": {"num_classes": 2},
            "umap": {"max_cells_per_state": 500},
        }
        umap_cfg = build_isp_umap_config(
            isp_cfg,
            ["ENSG00000115457", "ENSG00000139618"],
            gene_label="Igfbp2+Brca2",
        )
        self.assertEqual(
            umap_cfg["perturbation"]["genes_to_perturb"],
            ["ENSG00000115457", "ENSG00000139618"],
        )
        self.assertEqual(umap_cfg["perturbation"]["gene_label"], "Igfbp2+Brca2")
        self.assertEqual(umap_cfg["perturbation"]["type"], "overexpress")
        self.assertEqual(umap_cfg["umap"]["max_cells_per_state"], 500)
        self.assertEqual(umap_cfg["umap"]["sample_key"], "sample_id")

    def test_remap_isp_cfg_paths_from_cluster_absolute(self):
        from run_isp_umap import remap_isp_cfg_paths_to_run_dir, resolve_path_under_pipeline_run

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "pipeline_042051_549377Z"
            model = run_dir / "finetune" / "all_run1"
            dataset = run_dir / "tokenized_dataset" / "Asano_mouse-mouse_0.dataset"
            model.mkdir(parents=True)
            (model / "config.json").write_text("{}", encoding="utf-8")
            dataset.mkdir(parents=True)

            cluster_model = (
                "/work/LABGENPH/yuya_1/ISP-Platform/output/20260918/"
                "pipeline_042051_549377Z/finetune/all_run1"
            )
            cluster_ds = (
                "/work/LABGENPH/yuya_1/ISP-Platform/output/20260918/"
                "pipeline_042051_549377Z/tokenized_dataset/Asano_mouse-mouse_0.dataset"
            )
            remapped = resolve_path_under_pipeline_run(cluster_model, run_dir)
            self.assertEqual(remapped, model.resolve())

            cfg = remap_isp_cfg_paths_to_run_dir(
                {
                    "paths": {
                        "dataset": cluster_ds,
                        "geneformer_model": cluster_model,
                        "output_root": "/work/LABGENPH/yuya_1/ISP-Platform/output/20260918/pipeline_042051_549377Z",
                    }
                },
                run_dir,
            )
            self.assertEqual(Path(cfg["paths"]["geneformer_model"]), model.resolve())
            self.assertEqual(Path(cfg["paths"]["dataset"]), dataset.resolve())
            self.assertEqual(Path(cfg["paths"]["output_root"]), run_dir.resolve())

    def test_stratified_umap_sampling_avoids_prefix_bias(self):
        from run_isp_umap import allocate_stratified_counts, stratified_sample_indices

        # Equal groups: 40 cells from 80+80 must include both samples (prefix-of-N is all A).
        labels = np.array(["A"] * 80 + ["B"] * 80)
        idx = stratified_sample_indices(labels, 40, np.random.default_rng(0))
        self.assertEqual(len(idx), 40)
        self.assertEqual(len(np.unique(idx)), 40)
        self.assertEqual(set(labels[idx]), {"A", "B"})
        self.assertEqual(int(np.sum(labels[idx] == "A")), 20)
        self.assertEqual(int(np.sum(labels[idx] == "B")), 20)

        # Proportional: 90/10 mix kept; tiny sample still gets ≥1 cell.
        prop = np.array(["A"] * 900 + ["B"] * 100)
        prop_idx = stratified_sample_indices(prop, 100, np.random.default_rng(0))
        self.assertEqual(int(np.sum(prop[prop_idx] == "A")), 90)
        self.assertEqual(int(np.sum(prop[prop_idx] == "B")), 10)

        tiny = np.array(["A"] * 1999 + ["B"] * 1)
        tiny_idx = stratified_sample_indices(tiny, 200, np.random.default_rng(1))
        self.assertIn("B", tiny[tiny_idx])

        counts = allocate_stratified_counts(np.array([50, 50]), 100)
        np.testing.assert_array_equal(counts, [50, 50])
        np.testing.assert_array_equal(allocate_stratified_counts(np.array([900, 100]), 100), [90, 10])
        np.testing.assert_array_equal(allocate_stratified_counts(np.array([1999, 1]), 200), [199, 1])

        a = stratified_sample_indices(labels, 40, np.random.default_rng(7))
        b = stratified_sample_indices(labels, 40, np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)

    def test_subsample_state_dataset_stratifies_and_shuffles(self):
        from datasets import Dataset
        from run_isp_umap import subsample_state_dataset

        sample_id = ["s1"] * 60 + ["s2"] * 60
        ds = Dataset.from_dict(
            {
                "sample_id": sample_id,
                "cell_id": [f"c{i}" for i in range(120)],
            }
        )
        out = subsample_state_dataset(ds, max_cells=40, seed=42, sample_key="sample_id")
        self.assertEqual(len(out), 40)
        kept = list(out["sample_id"])
        self.assertEqual(kept.count("s1"), 20)
        self.assertEqual(kept.count("s2"), 20)
        runs = 1 + sum(a != b for a, b in zip(kept, kept[1:]))
        self.assertGreater(runs, 2)

        shuffled = subsample_state_dataset(ds, max_cells=40, seed=42, sample_key=None)
        self.assertEqual(len(shuffled), 40)

    def test_apply_group_delete_and_overexpress(self):
        from run_isp_umap import apply_group_delete, apply_group_overexpress

        example = {"input_ids": [10, 20, 30, 40, 50], "length": 5}
        deleted = apply_group_delete(example, [20, 40])
        self.assertEqual(deleted["input_ids"], [10, 30, 50])
        self.assertEqual(deleted["length"], 3)

        oe = apply_group_overexpress({"input_ids": [10, 20, 30], "length": 3}, [99, 20])
        self.assertEqual(oe["input_ids"][:2], [99, 20])
        self.assertEqual(oe["length"], 3)

    def test_umap_position_method_label(self):
        from run_isp_umap import umap_position_method_label

        self.assertIn("direct", umap_position_method_label({"pca_components": 0, "seed": 42}))
        self.assertIn("PCA(50)", umap_position_method_label({"pca_components": 50, "seed": 0}))

    def test_fit_umap_projection_direct_and_pca(self):
        from run_isp_umap import fit_umap_projection

        rng = np.random.default_rng(0)
        embs = rng.standard_normal((80, 32)).astype(np.float32)

        direct, direct_method = fit_umap_projection(embs, {"pca_components": 0, "seed": 42})
        self.assertEqual(direct.shape, (80, 2))
        self.assertIn("direct", direct_method)

        pca_xy, pca_method = fit_umap_projection(
            embs,
            {"pca_components": 50, "seed": 0, "n_neighbors": 15, "min_dist": 0.1},
        )
        self.assertEqual(pca_xy.shape, (80, 2))
        self.assertIn("PCA", pca_method)

    def test_isp_umap_scatter_uses_endpoint_axes(self):
        import tempfile

        import matplotlib
        import numpy as np
        from umap_plot_style import apply_endpoint_umap_rc, plot_isp_umap_scatter, style_umap_axes

        matplotlib.use("Agg")
        apply_endpoint_umap_rc()
        fig, ax = matplotlib.pyplot.subplots()
        style_umap_axes(ax)
        self.assertFalse(ax.spines["top"].get_visible())
        self.assertFalse(ax.spines["right"].get_visible())
        self.assertEqual(ax.get_xlabel(), "UMAP-1")
        self.assertEqual(ax.get_ylabel(), "UMAP-2")
        self.assertGreater(ax.get_facecolor()[0], 0.99)
        matplotlib.pyplot.close(fig)

        rng = np.random.default_rng(0)
        xy = rng.normal(size=(30, 2))
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "umap.png"
            plot_isp_umap_scatter(
                xy,
                n_end=10,
                n_start=10,
                end_state="goal",
                start_state="start",
                pert_label="start+ISP(X)",
                title="test",
                out_path=out,
                show_arrows=True,
                num_arrows=5,
            )
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 1000)


class TestTokenizerVocabulary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _purge_import_stubs()

    def test_refresh_vocabulary_keys_intersects_median_and_token_dicts(self):
        from geneformer.tokenizer import TranscriptomeTokenizer

        tk = TranscriptomeTokenizer.__new__(TranscriptomeTokenizer)
        tk.gene_median_dict = {"ENSG1": 1.0, "ENSG2": 2.0, "ENSG00000293164": 3.0}
        tk.gene_token_dict = {"ENSG1": 10, "ENSG3": 30}
        tk._refresh_vocabulary_keys()
        self.assertEqual(set(tk.gene_keys), {"ENSG1"})
        self.assertTrue(tk.genelist_dict.get("ENSG1"))
        self.assertNotIn("ENSG2", tk.genelist_dict)
        self.assertNotIn("ENSG00000293164", tk.genelist_dict)


if __name__ == "__main__":
    unittest.main()
