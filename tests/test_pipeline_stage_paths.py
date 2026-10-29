"""Pipeline dataset shards and fine-tuned model selection (pipeline_lib)."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_p0_species import pipeline_lib  # noqa: E402

HAS_DATASETS = importlib.util.find_spec("datasets") is not None


def _resolved() -> dict:
    return {
        "dataset_path": "/data/dataset",
        "pretrained_model": "/models/model",
        "finetune_run_dir": "/out/finetune",
        "species": {},
        "max_input_size": 2048,
    }


def _schedule():
    path = Path(__file__).resolve().parent.parent / "core" / "geneformer" / "training_schedule.py"
    spec = importlib.util.spec_from_file_location("training_schedule_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_run(finetune_dir: Path, name: str, accuracy: float | None) -> Path:
    run = finetune_dir / name
    run.mkdir(parents=True)
    (run / "label_dict.json").write_text('{"AD": 0, "WT": 1}\n', encoding="utf-8")
    if accuracy is not None:
        (run / "eval_results.json").write_text(
            json.dumps({"test_accuracy": accuracy}), encoding="utf-8"
        )
    return run


class TestTokenizedShards(unittest.TestCase):
    def test_single_shard_is_used_as_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok = Path(tmp)
            (tok / "Study_0.dataset").mkdir()
            self.assertEqual(
                pipeline_lib.resolve_tokenized_dataset(tok, "Study"), tok / "Study_0.dataset"
            )
            self.assertFalse((tok / "Study_merged.dataset").exists())

    def test_shards_sorted_numerically_and_other_studies_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok = Path(tmp)
            for name in ("Study_10.dataset", "Study_2.dataset", "Study_0.dataset", "Study_1.dataset"):
                (tok / name).mkdir()
            for i in range(3, 10):
                (tok / f"Study_{i}.dataset").mkdir()
            (tok / "Study_x_0.dataset").mkdir()
            (tok / "Study_merged.dataset").mkdir()
            names = [p.name for p in pipeline_lib.tokenized_shard_paths(tok, "Study")]
            self.assertEqual(names, [f"Study_{i}.dataset" for i in range(11)])

    def test_gap_in_shard_numbers_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok = Path(tmp)
            (tok / "Study_0.dataset").mkdir()
            (tok / "Study_2.dataset").mkdir()
            with self.assertRaises(ValueError):
                pipeline_lib.tokenized_shard_paths(tok, "Study")

    def test_missing_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                pipeline_lib.resolve_tokenized_dataset(Path(tmp), "Study")

    def test_remove_tokenized_outputs_keeps_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok = Path(tmp)
            for name in ("Study_0.dataset", "Study_3.dataset", "Study_merged.dataset"):
                (tok / name).mkdir()
            (tok / "Study_merged.json").write_text("{}", encoding="utf-8")
            (tok / "conversion_report.json").write_text("{}", encoding="utf-8")
            removed = pipeline_lib.remove_tokenized_outputs(tok, "Study")
            self.assertEqual(len(removed), 4)
            self.assertEqual([p.name for p in tok.iterdir()], ["conversion_report.json"])

    @unittest.skipUnless(HAS_DATASETS, "datasets not installed")
    def test_multiple_shards_are_concatenated_in_order(self):
        from datasets import Dataset, load_from_disk

        with tempfile.TemporaryDirectory() as tmp:
            tok = Path(tmp)
            Dataset.from_dict({"input_ids": [[1, 2], [3]], "disease": ["AD", "WT"]}).save_to_disk(
                str(tok / "Study_0.dataset")
            )
            Dataset.from_dict({"input_ids": [[4, 5, 6]], "disease": ["AD"]}).save_to_disk(
                str(tok / "Study_1.dataset")
            )
            merged = pipeline_lib.resolve_tokenized_dataset(tok, "Study")
            self.assertEqual(merged, tok / "Study_merged.dataset")
            ds = load_from_disk(str(merged))
            self.assertEqual(ds["input_ids"], [[1, 2], [3], [4, 5, 6]])
            self.assertEqual(ds["disease"], ["AD", "WT", "AD"])
            manifest = json.loads((tok / "Study_merged.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["cells_per_shard"], [2, 1])
            self.assertEqual(pipeline_lib.resolve_tokenized_dataset(tok, "Study"), merged)


class TestFinetuneModelDir(unittest.TestCase):
    def test_single_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            ft = Path(tmp)
            run = _make_run(ft, "all_run1", 0.8)
            (ft / "best_all").mkdir()
            (ft / "figures").mkdir()
            self.assertEqual(pipeline_lib.resolve_finetune_model_dir(ft), run)

    def test_best_accuracy_across_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            ft = Path(tmp)
            _make_run(ft, "all_run1", 0.80)
            best = _make_run(ft, "all_run2", 0.91)
            _make_run(ft, "all_run3", 0.91)
            self.assertEqual(pipeline_lib.resolve_finetune_model_dir(ft), best)

    def test_runs_without_accuracy_fall_back_to_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            ft = Path(tmp)
            first = _make_run(ft, "all_run1", None)
            _make_run(ft, "all_run2", None)
            self.assertEqual(pipeline_lib.resolve_finetune_model_dir(ft), first)

    def test_organ_groups_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ft = Path(tmp)
            _make_run(ft, "brain_run1", 0.9)
            _make_run(ft, "liver_run1", 0.9)
            with self.assertRaises(ValueError):
                pipeline_lib.resolve_finetune_model_dir(ft)

    def test_non_all_single_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            ft = Path(tmp)
            run = _make_run(ft, "brain_run1", 0.9)
            self.assertEqual(pipeline_lib.resolve_finetune_model_dir(ft), run)

    def test_standard_mode_keeps_historical_batch_and_accumulation_one(self):
        cfg = pipeline_lib.build_finetune_config(
            {
                "runtime": {
                    "ft_batch_mode": "standard",
                    "train_batch_size": 6,
                    "micro_batch_size": 2,
                    "micro_gradient_accumulation_steps": 3,
                }
            },
            _resolved(),
            {"training": {"batch_size": 4, "gradient_accumulation_steps": 5}},
        )
        self.assertEqual(cfg["training"]["batch_size"], 6)
        self.assertEqual(cfg["training"]["gradient_accumulation_steps"], 1)

    def test_micro_mode_does_not_use_the_stored_standard_batch(self):
        cfg = pipeline_lib.build_finetune_config(
            {
                "runtime": {
                    "ft_batch_mode": "micro",
                    "train_batch_size": 6,
                    "micro_batch_size": 2,
                    "micro_effective_batch_size": 6,
                    "micro_gradient_accumulation_steps": 9,
                }
            },
            _resolved(),
            {"training": {"batch_size": 6, "gradient_accumulation_steps": 1}},
        )
        self.assertEqual(cfg["training"]["batch_size"], 2)
        self.assertEqual(cfg["training"]["gradient_accumulation_steps"], 3)

    def test_micro_mode_without_effective_size_keeps_stored_pass_count(self):
        cfg = pipeline_lib.build_finetune_config(
            {
                "runtime": {
                    "ft_batch_mode": "micro",
                    "train_batch_size": 6,
                    "micro_batch_size": 2,
                    "micro_gradient_accumulation_steps": 3,
                }
            },
            _resolved(),
            {"training": {"batch_size": 6}},
        )
        self.assertEqual(cfg["training"]["batch_size"], 2)
        self.assertEqual(cfg["training"]["gradient_accumulation_steps"], 3)

    def test_effective_batch_must_divide_by_cells_on_the_gpu(self):
        with self.assertRaises(ValueError):
            pipeline_lib.build_finetune_config(
                {
                    "runtime": {
                        "ft_batch_mode": "micro",
                        "micro_batch_size": 4,
                        "micro_effective_batch_size": 6,
                    }
                },
                _resolved(),
                {"training": {}},
            )

    def test_legacy_runtime_without_mode_still_uses_train_batch_size(self):
        cfg = pipeline_lib.build_finetune_config(
            {"runtime": {"train_batch_size": 6}},
            _resolved(),
            {"training": {"batch_size": 4, "gradient_accumulation_steps": 1}},
        )
        self.assertEqual(cfg["training"]["batch_size"], 6)
        self.assertEqual(cfg["training"]["gradient_accumulation_steps"], 1)

    def test_optimizer_updates_match_historical_batch_when_accumulation_is_one(self):
        updates = _schedule().optimizer_updates_per_epoch
        self.assertEqual(updates(7998, 6, 1), 1333)
        self.assertEqual(updates(7998, 2, 1), 3999)
        self.assertEqual(updates(7998, 2, 3), 1333)
        choose = _schedule().choose_ft_batch
        self.assertEqual(choose(2048, 6), ("standard", 6, 1))
        self.assertEqual(choose(4096, 6), ("micro", 2, 3))
        self.assertEqual(choose(4096, 4), ("micro", 2, 2))
        profiled = _schedule().choose_ft_batch_from_lengths
        gib = 1024**3
        long_cells = [4096] * 100
        self.assertEqual(
            profiled(long_cells, 121 * gib, model_cap=4096),
            ("micro", 2, 3, 4096),
        )
        self.assertEqual(
            profiled([2048] * 100, 80 * gib, model_cap=2048),
            ("standard", 6, 1, 2048),
        )
        self.assertEqual(
            profiled([1500] * 100, 80 * gib, model_cap=4096),
            ("standard", 6, 1, 1500),
        )
        self.assertEqual(
            profiled(long_cells, 40 * gib, model_cap=4096),
            ("micro", 1, 6, 4096),
        )
        self.assertEqual(profiled([], 121 * gib, model_cap=4096), ("micro", 2, 3, 4096))
        self.assertEqual(profiled([], 80 * gib, model_cap=2048), ("standard", 6, 1, 2048))

    def test_cell_nnz_counts_columns_of_a_10x_matrix(self):
        import data_input_layout

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matrix.mtx"
            path.write_text(
                "%%MatrixMarket matrix coordinate integer general\n"
                "2 3 4\n"
                "1 1 5\n"
                "2 1 1\n"
                "1 2 1\n"
                "2 3 1\n",
                encoding="utf-8",
            )
            self.assertEqual(data_input_layout.cell_nnz_counts(path), [2, 1, 1])

    def test_auto_choice_defers_the_batch_until_the_run(self):
        cfg = pipeline_lib.build_finetune_config(
            {
                "runtime": {
                    "ft_batch_choice": "auto",
                    "ft_batch_mode": "micro",
                    "train_batch_size": 5,
                    "micro_batch_size": 2,
                    "micro_effective_batch_size": 6,
                }
            },
            _resolved(),
            {"training": {"batch_size": 6, "gradient_accumulation_steps": 1}},
        )
        self.assertEqual(cfg["training"]["batch_size_policy"], "auto")
        self.assertEqual(cfg["training"]["effective_batch_size"], 6)
        self.assertEqual(cfg["training"]["batch_size"], 6)
        self.assertEqual(cfg["training"]["gradient_accumulation_steps"], 1)

    def test_auto_decision_is_applied_to_the_run_configs(self):
        decision = _schedule().auto_ft_batch_decision(
            [4096] * 20,
            114 * 1024**3,
            effective_batch=6,
            model_cap=4096,
            memory_source="unified",
        )
        self.assertEqual(decision["mode"], "micro")
        self.assertEqual((decision["cells"], decision["passes"]), (2, 3))
        finetune = {"training": {"batch_size": 6, "gradient_accumulation_steps": 1}}
        pipeline = {"runtime": {"train_batch_size": 6, "ft_batch_choice": "auto"}}
        _schedule().apply_auto_ft_batch_to_configs(finetune, pipeline, decision)
        self.assertEqual(finetune["training"]["batch_size"], 2)
        self.assertEqual(finetune["training"]["gradient_accumulation_steps"], 3)
        self.assertTrue(pipeline["runtime"]["ft_batch_decided"])
        self.assertEqual(pipeline["runtime"]["micro_batch_size"], 2)
        self.assertEqual(pipeline["runtime"]["ft_n_cells"], 20)
        self.assertEqual(pipeline["runtime"]["train_batch_size"], 6)

    def test_missing_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            ft = Path(tmp)
            (ft / "all_run1" / "checkpoint-10").mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                pipeline_lib.resolve_finetune_model_dir(ft)


if __name__ == "__main__":
    unittest.main()
