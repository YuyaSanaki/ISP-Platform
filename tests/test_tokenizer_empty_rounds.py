"""tokenize_data terminates when a round of files yields no cells."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "core", ROOT / "contracts", ROOT / "webui"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


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


class TestTokenizeDataEmptyRounds(unittest.TestCase):
    def setUp(self) -> None:
        _purge_import_stubs()
        from geneformer.tokenizer import TranscriptomeTokenizer

        self.tk = TranscriptomeTokenizer.__new__(TranscriptomeTokenizer)
        self.tk.start_reading_file_num = 0
        self.tk.last_dataset_flag = False
        self.saved: list[str] = []

        def create_dataset(cells, metadata, use_generator=False):
            ds = mock.Mock()
            ds.save_to_disk.side_effect = self.saved.append
            return ds

        self.tk.create_dataset = create_dataset

    def _rounds(self, rounds):
        """rounds: list of (n_cells, is_last). Fails instead of looping forever."""
        calls = iter(rounds)

        def tokenize_files(data_set_num, data_directory, file_format="loom"):
            try:
                n_cells, is_last = next(calls)
            except StopIteration:
                raise AssertionError("tokenize_files called again after the last round")
            if is_last:
                self.tk.last_dataset_flag = True
            else:
                self.tk.start_reading_file_num += 1
            return [[1, 2, 3]] * n_cells, None

        self.tk.tokenize_files = tokenize_files

    def test_empty_last_round_after_saved_round_stops(self):
        self._rounds([(5, False), (0, True)])
        self.tk.tokenize_data("in", "out", "study")
        self.assertEqual(self.saved, ["out/study_0.dataset"])

    def test_empty_middle_round_is_skipped(self):
        self._rounds([(0, False), (4, True)])
        self.tk.tokenize_data("in", "out", "study")
        self.assertEqual(self.saved, ["out/study_0.dataset"])

    def test_no_cells_at_all_raises(self):
        self._rounds([(0, True)])
        with self.assertRaisesRegex(RuntimeError, "No cells were tokenized"):
            self.tk.tokenize_data("in", "out", "study")
        self.assertEqual(self.saved, [])

    def test_rounds_with_cells_unchanged(self):
        self._rounds([(5, False), (3, False), (2, True)])
        self.tk.tokenize_data("in", "out", "study")
        self.assertEqual(
            self.saved,
            ["out/study_0.dataset", "out/study_1.dataset", "out/study_2.dataset"],
        )


if __name__ == "__main__":
    unittest.main()
