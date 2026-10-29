"""Methods paragraph shown under Web UI Run directory."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webui"))

from method_description import (  # noqa: E402
    RUN_KIND_E2E,
    LengthReadError,
    apply_form_overrides,
    build_method_view,
    facts_from_config,
    format_method_description,
    length_stats_for_run,
    load_run_record,
    platform_version,
    summarize_lengths,
    tokenized_dataset_dirs,
)


def _assert_methods_punctuation(test: unittest.TestCase, text: str) -> None:
    test.assertNotIn(";", text)
    test.assertNotIn("—", text)
    test.assertNotIn(" – ", text)
    test.assertNotIn(" - ", text)
    for index, char in enumerate(text):
        if char == "–":
            test.assertTrue(index > 0 and text[index - 1].isdigit(), text)
            test.assertTrue(index + 1 < len(text) and text[index + 1].isdigit(), text)


class TestMethodDescription(unittest.TestCase):
    def test_platform_template_paragraph(self):
        cfg = yaml.safe_load((ROOT / "core/config/pipeline.yaml").read_text(encoding="utf-8"))
        text = format_method_description(
            facts_from_config(cfg, version="1.4.1", run_kind="e2e")
        )
        self.assertIn("ISP Platform v1.4.1", text)
        self.assertIn("end-to-end pipeline", text)
        self.assertIn("The input species was mouse.", text)
        self.assertIn("Mouse Geneformer base.", text)
        self.assertNotIn("about 33 million", text)
        self.assertIn("ISP mode was Genome-wide delete.", text)
        self.assertNotIn("Each gene token", text)
        self.assertNotIn("Every gene in the model vocabulary", text)
        self.assertIn("Start was Disease and Goal was WT (disease).", text)
        self.assertIn("Fine-tuning used a batch size of 6.", text)
        self.assertNotIn("tokenize, fine-tune", text)
        self.assertNotIn("Token length", text)
        _assert_methods_punctuation(self, text)

    def test_length_stats_and_cross_species(self):
        cfg = {
            "species": {
                "model_organism": "human",
                "model": "mouse_geneformer",
                "mouse_variant": "base",
                "ortholog_policy": "one2one",
            },
            "perturbation": {
                "type": "overexpress",
                "genes_to_perturb": ["POU5F1", "SOX2"],
                "state_key": "State",
                "start_state": "somatic",
                "end_state": "pluripotent",
            },
            "runtime": {"ft_batch_mode": "micro", "micro_batch_size": 2, "micro_gradient_accumulation_steps": 3},
        }
        lengths = summarize_lengths([100, 2048, 2048, 400], max_input_size=2048)
        text = format_method_description(
            facts_from_config(
                cfg,
                version="1.4.1",
                run_kind="e2e",
                git_commit="abcdef0123456789abcdef0123456789abcdef01",
                length=lengths,
            )
        )
        self.assertIn("commit abcdef0", text)
        self.assertIn("The input species was human.", text)
        self.assertIn("mapped to the mouse model vocabulary by one-to-one orthologs", text)
        self.assertIn("ISP mode was overexpress.", text)
        self.assertNotIn("Genome-wide overexpress", text)
        self.assertIn("POU5F1 and SOX2", text)
        self.assertIn("Start was somatic and Goal was pluripotent (State).", text)
        self.assertIn("effective batch size of 6 (2 cells * 3 passes).", text)
        self.assertNotIn("GPU passes", text)
        self.assertNotIn("selected automatically", text)
        self.assertIn("Token length", text)
        self.assertIn("SD ", text)
        self.assertIn("2 cells were at the input-size cap of 2,048.", text)
        self.assertNotIn(";", text)
        _assert_methods_punctuation(self, text)

    def test_median_and_sample_sd(self):
        stats = summarize_lengths([1, 2, 3, 4], max_input_size=2048)
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats.n, 4)
        self.assertEqual(stats.median, 2.5)
        self.assertEqual(stats.mean, 2.5)
        self.assertAlmostEqual(stats.sd or 0, 1.2909944487358056)
        self.assertEqual(stats.n_at_cap, 0)
        text = format_method_description(
            facts_from_config(
                {"species": {"model_organism": "mouse", "model": "mouse_geneformer"}},
                version="1.4.1",
                run_kind="e2e",
                length=stats,
            )
        )
        self.assertIn("median 2.5", text)
        self.assertIn("mean 2.5", text)
        self.assertIn("SD 1.3", text)
        self.assertIn("range 1–4", text)
        self.assertIn("n = 4 cells", text)
        self.assertNotIn("input-size cap", text)

    def test_isp_stage_config_does_not_invent_a_fine_tune_batch(self):
        cfg = {
            "species": {"model_organism": "mouse", "model": "mouse_geneformer"},
            "perturbation": {
                "type": "delete",
                "start_state": "Disease",
                "end_state": "WT",
                "state_key": "disease",
            },
            "runtime": {"forward_batch_size": "auto"},
        }
        text = format_method_description(
            facts_from_config(cfg, version="1.4.1", run_kind="e2e")
        )
        self.assertNotIn("train_batch_size", text)
        self.assertNotIn("Fine-tuning", text)

    def test_ft_batch_calibration(self):
        cfg = yaml.safe_load(
            (ROOT / "core/config/ft_batch_calibrate.yaml").read_text(encoding="utf-8")
        )
        text = format_method_description(
            facts_from_config(
                cfg,
                version=platform_version(ROOT),
                run_kind="ft_batch",
                recommended_batch=7,
            )
        )
        self.assertIn("Fine-tune batch size was calibrated", text)
        self.assertIn("did not tokenize cells", text)
        self.assertIn("The recommended train_batch_size was 7.", text)
        self.assertNotIn("ISP mode", text)
        self.assertNotIn("end-to-end pipeline", text)
        _assert_methods_punctuation(self, text)

    def test_state_feedback_and_umap(self):
        cfg = {
            "data": {"input_dir": "/tmp/study"},
            "species": {
                "model_organism": "human",
                "model": "human_geneformer",
                "human_variant": "v2_316m",
            },
            "perturbation": {
                "type": "delete",
                "start_state": "D0",
                "end_state": "P20P",
                "state_key": "time",
            },
            "runtime": {"ft_batch_mode": "standard", "train_batch_size": 4},
        }
        sf = format_method_description(
            facts_from_config(
                cfg,
                version="1.4.1",
                run_kind="state_feedback",
                sf_mode="Placebo-corrected (paper method)",
                sf_steps=[
                    {"type": "overexpress", "genes": ["Pou5f1", "Sox2", "Klf4", "Myc"]},
                    {"type": "delete", "genes": ["Trp53"]},
                ],
            )
        )
        self.assertIn("State-feedback ISP", sf)
        self.assertIn("placebo correction", sf)
        self.assertIn("Step 1 was overexpress of Pou5f1, Sox2, Klf4 and Myc.", sf)
        self.assertIn("Step 2 was delete of Trp53.", sf)
        self.assertIn("The model was Human Geneformer V2-316M.", sf)
        self.assertNotIn("about 316 million", sf)
        self.assertIn("train_batch_size 4", sf)
        self.assertNotIn("Every gene in the model vocabulary", sf)
        _assert_methods_punctuation(self, sf)

        umap = format_method_description(
            facts_from_config(
                cfg,
                version="1.4.1",
                run_kind="isp_umap",
                umap_genes=["Igfbp2"],
            )
        )
        self.assertIn("per-cell trajectory UMAP", umap)
        self.assertIn("The UMAP perturbed Igfbp2.", umap)
        self.assertNotIn("together", umap)
        self.assertIn("ISP mode was delete.", umap)
        _assert_methods_punctuation(self, umap)

        pending = format_method_description(
            facts_from_config({}, version="1.4.1", run_kind="isp_umap", source_ready=False)
        )
        self.assertIn("is configured", pending)
        self.assertNotIn("completed end-to-end", pending)

    def test_dataset_choice_and_run_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            tokenized = run / "tokenized_dataset"
            shard = tokenized / "study_0.dataset"
            merged = tokenized / "study_merged.dataset"
            shard.mkdir(parents=True)
            merged.mkdir()
            self.assertEqual(tokenized_dataset_dirs(run), [merged])
            seen: list[str] = []

            def loader(path: Path) -> list[int]:
                seen.append(path.name)
                return [10, 30, 20, 40]

            stats = length_stats_for_run(run, max_input_size=2048, loader=loader)
            self.assertEqual(seen, ["study_merged.dataset"])
            assert stats is not None
            self.assertEqual(stats.median, 25)
            self.assertEqual(stats.n, 4)

            (run / "pipeline_run_metadata.yaml").write_text(
                "git_commit: ABCDEF0123456789abcdef0123456789abcdef01\n",
                encoding="utf-8",
            )
            (run / "pipeline_config_used.yaml").write_text(
                "species:\n  model_organism: mouse\n  model: mouse_geneformer\n",
                encoding="utf-8",
            )
            cfg, commit = load_run_record(run)
            self.assertEqual(commit, "abcdef0")
            self.assertEqual(cfg["species"]["model"], "mouse_geneformer")

    def test_missing_length_column_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "tokenized_dataset" / "study_0.dataset"
            dataset.mkdir(parents=True)

            def loader(_path: Path) -> list[int]:
                raise LengthReadError("study_0.dataset has no length column.")

            with self.assertRaises(LengthReadError):
                length_stats_for_run(Path(tmp), loader=loader)

    def test_version_file(self):
        self.assertEqual(platform_version(ROOT), "1.4.1")

    def test_form_override_and_view_notes(self):
        cfg = apply_form_overrides(
            {"species": {"model_organism": "mouse", "model": "mouse_geneformer"}},
            {"model_organism": "human", "pert_type": "delete", "start": "A", "goal": "B", "state_key": "State"},
        )
        self.assertEqual(cfg["species"]["model_organism"], "human")
        self.assertEqual(cfg["perturbation"]["start_state"], "A")
        text, notes = build_method_view(
            version="1.4.1",
            run_kind=RUN_KIND_E2E,
            live_cfg=cfg,
        )
        self.assertIn("The input species was human.", text)
        self.assertIn("Start was A and Goal was B", text)
        self.assertIn("current form", " ".join(notes))
        self.assertTrue(any("Token lengths" in note for note in notes))
        _assert_methods_punctuation(self, text)


if __name__ == "__main__":
    unittest.main()
