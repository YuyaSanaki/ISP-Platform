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

    def test_missing_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            ft = Path(tmp)
            (ft / "all_run1" / "checkpoint-10").mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                pipeline_lib.resolve_finetune_model_dir(ft)


if __name__ == "__main__":
    unittest.main()
