"""Parallel ISP is opt-in. Chunk math, markers and stats order do not need a GPU."""
from __future__ import annotations

import pickle
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "core", ROOT / "contracts"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from isp_parallel import (  # noqa: E402
    ChunkMergeError,
    apply_backend_input_size,
    centroid_signature,
    chunk_config_sha256,
    chunk_shares_output_folder,
    clear_chunk_outputs,
    parse_chunk_spec,
    reject_parallel_mode,
    stride_cell_indices,
    validate_cell_shard,
    validate_parallel_run,
    write_chunk_marker,
    yaml_chunk_pair,
)

try:
    import pandas as pd

    from geneformer.in_silico_perturber_stats import (
        _stable_downsample,
        isp_stats_to_goal_state,
        read_dictionaries,
    )

    STATS_MISSING = None
except Exception as exc:  # pragma: no cover
    STATS_MISSING = str(exc)


def _cfg(**overrides):
    cfg = {
        "paths": {"dataset": "/data/cells.dataset", "geneformer_model": "/models/gf"},
        "species": {"model_organism": "human"},
        "perturbation": {
            "type": "delete",
            "organ_data": "exp",
            "state_key": "state",
            "start_state": "Disease",
            "end_state": "Ctrl",
            "alt_states": [],
            "genes_to_perturb": [],
        },
        "model": {"type": "CellClassifier", "num_classes": 2, "max_input_size": 4096},
        "isp": {"max_ncells": 2000, "combos": 0, "anchor_gene": None},
        "stats": {"mode": "goal_state_shift"},
    }
    cfg.update(overrides)
    return cfg


class TestChunkSpec(unittest.TestCase):
    def test_shared_folder_is_subdir_or_flat_output(self):
        self.assertFalse(chunk_shares_output_folder(None, True))
        self.assertTrue(chunk_shares_output_folder("run1", True))
        self.assertTrue(chunk_shares_output_folder(None, False))

    def test_default_is_single_process(self):
        self.assertIsNone(parse_chunk_spec())
        self.assertIsNone(parse_chunk_spec(env_id="", env_count=""))
        self.assertIsNone(yaml_chunk_pair({"isp": {}})[0])

    def test_cli_beats_env_and_yaml(self):
        self.assertEqual(parse_chunk_spec(1, 4, 0, 8, 2, 8), (1, 4))
        self.assertEqual(parse_chunk_spec(None, None, 3, 8, 0, 4), (3, 8))
        self.assertEqual(parse_chunk_spec(None, None, None, None, 2, 5), (2, 5))

    def test_partial_spec_and_bounds_fail(self):
        with self.assertRaises(ValueError):
            parse_chunk_spec(0, None)
        with self.assertRaises(ValueError):
            parse_chunk_spec(None, None, None, None, 0, None)
        with self.assertRaises(ValueError):
            parse_chunk_spec(0, 1)
        with self.assertRaises(ValueError):
            parse_chunk_spec(4, 4)
        self.assertIsNone(validate_cell_shard(None))
        with self.assertRaises(ValueError):
            validate_cell_shard((0, 1))

    def test_group_mode_is_rejected(self):
        reject_parallel_mode(genes_to_perturb=[], combos=0, anchor_gene=None)
        with self.assertRaisesRegex(ValueError, "genome-wide"):
            reject_parallel_mode(genes_to_perturb=["TP53"], combos=0, anchor_gene=None)
        with self.assertRaisesRegex(ValueError, "combos"):
            reject_parallel_mode(genes_to_perturb=[], combos=1, anchor_gene=None)
        with self.assertRaisesRegex(ValueError, "anchor"):
            reject_parallel_mode(genes_to_perturb=[], combos=0, anchor_gene="ENSG00000141510")


class TestStride(unittest.TestCase):
    def test_covers_every_cell_once_and_balances(self):
        n_cells, count = 10, 4
        parts = [stride_cell_indices(n_cells, i, count) for i in range(count)]
        flat = [i for part in parts for i in part]
        self.assertEqual(sorted(flat), list(range(n_cells)))
        self.assertEqual(len(flat), len(set(flat)))
        lengths = [len(part) for part in parts]
        self.assertLessEqual(max(lengths) - min(lengths), 1)
        # Length-descending cells are not handed to chunk 0 as a prefix.
        self.assertNotEqual(parts[0], list(range(len(parts[0]))))
        self.assertEqual(parts[0], [0, 4, 8])

    def test_too_few_cells_fails(self):
        with self.assertRaisesRegex(ValueError, "at least 4"):
            stride_cell_indices(3, 0, 4)


class TestSignatures(unittest.TestCase):
    def test_config_hash_ignores_chunk_id(self):
        left = _cfg()
        left["isp"] = {**left["isp"], "chunk": {"id": 0, "count": 4}}
        right = _cfg()
        right["isp"] = {**right["isp"], "chunk": {"id": 3, "count": 4}}
        self.assertEqual(chunk_config_sha256(left, "auto"), chunk_config_sha256(right, "auto"))
        right["perturbation"] = {**right["perturbation"], "start_state": "Other"}
        self.assertNotEqual(chunk_config_sha256(left, "auto"), chunk_config_sha256(right, "auto"))
        self.assertNotEqual(chunk_config_sha256(left, "auto"), chunk_config_sha256(left, 64))

    def test_omitted_input_size_is_part_of_the_hash_once_the_backend_fills_it(self):
        cfg = _cfg()
        del cfg["model"]["max_input_size"]
        bare = chunk_config_sha256(cfg, 4)
        previous = apply_backend_input_size(cfg, 2048)
        self.assertIsNone(previous)
        self.assertEqual(cfg["model"]["max_input_size"], 2048)
        filled = chunk_config_sha256(cfg, 4)
        self.assertNotEqual(filled, bare)
        other = _cfg()
        del other["model"]["max_input_size"]
        apply_backend_input_size(other, 2048)
        self.assertEqual(chunk_config_sha256(other, 4), filled)

    def test_centroid_hash_ignores_sub_millesimal_noise(self):
        base = {"Disease": [1.0, -0.5, 0.25], "Ctrl": [0.1, 0.2, 0.3]}
        noisy = {"Ctrl": [0.1000004, 0.2, 0.3], "Disease": [1.0000004, -0.5, 0.25]}
        shifted = {"Disease": [1.02, -0.5, 0.25], "Ctrl": [0.1, 0.2, 0.3]}
        self.assertEqual(centroid_signature(base)[0], centroid_signature(noisy)[0])
        self.assertNotEqual(centroid_signature(base)[0], centroid_signature(shifted)[0])
        self.assertEqual(centroid_signature(None)[0], centroid_signature({})[0])


def _marker(chunk_id, chunk_count, **overrides):
    marker = {
        "version": 1,
        "chunk_id": chunk_id,
        "chunk_count": chunk_count,
        "n_cells": 2 if chunk_id < 2 else 1,
        "n_cells_before_shard": 5,
        "config_sha256": "abc",
        "centroid_sha256": "def",
        "centroid_l2": {"Ctrl": 1.0},
        "output_prefix": "output_in-silico_SEdelete_ORexp_STDisease_ENCtrl",
        "perturb_type": "delete",
        "pickle_files": [f"in_silico_delete_prefix_chunk{chunk_id}of{chunk_count}_dict_1Kbatch-1_raw.pickle"],
    }
    marker.update(overrides)
    return marker


class TestMergeValidation(unittest.TestCase):
    def test_accepts_a_complete_run_and_rejects_the_failure_modes(self):
        count = 3
        with self._tmpdir() as raw:
            root = Path(raw)
            isp = root / "isp_results"
            markers = root / "isp_chunks"
            isp.mkdir()
            written = []
            for i in range(count):
                marker = _marker(i, count, n_cells=2 if i < 2 else 1)
                write_chunk_marker(markers, marker)
                name = marker["pickle_files"][0]
                (isp / name).write_bytes(b"pickle")
                written.append(name)
            self.assertEqual(len(validate_parallel_run(isp, markers, count)), count)

            extra = isp / "in_silico_delete_old_dict_1Kbatch-1_raw.pickle"
            extra.write_bytes(b"old")
            with self.assertRaisesRegex(ChunkMergeError, "Extra"):
                validate_parallel_run(isp, markers, count)
            extra.unlink()

            (isp / written[0]).unlink()
            with self.assertRaisesRegex(ChunkMergeError, "missing"):
                validate_parallel_run(isp, markers, count)
            (isp / written[0]).write_bytes(b"pickle")

            drifted = _marker(1, count, n_cells=2, config_sha256="other")
            write_chunk_marker(markers, drifted)
            with self.assertRaisesRegex(ChunkMergeError, "config hashes"):
                validate_parallel_run(isp, markers, count)

            (markers / "chunk_2_of_3.json").unlink()
            with self.assertRaisesRegex(ChunkMergeError, "Missing chunk"):
                validate_parallel_run(isp, markers, count)

    def test_clear_removes_only_this_chunk(self):
        with self._tmpdir() as raw:
            root = Path(raw)
            isp = root / "isp_results"
            markers = root / "isp_chunks"
            isp.mkdir()
            keep = isp / "in_silico_delete_prefix_chunk1of2_dict_1Kbatch-1_raw.pickle"
            drop = isp / "in_silico_delete_prefix_chunk0of2_dict_1Kbatch-1_raw.pickle"
            keep.write_bytes(b"k")
            drop.write_bytes(b"d")
            write_chunk_marker(markers, _marker(0, 2, n_cells=1, n_cells_before_shard=2))
            write_chunk_marker(markers, _marker(1, 2, n_cells=1, n_cells_before_shard=2))
            clear_chunk_outputs(
                isp, markers,
                perturb_type="delete",
                output_prefix="prefix",
                chunk_id=0,
                chunk_count=2,
            )
            self.assertTrue(keep.is_file())
            self.assertFalse(drop.is_file())
            self.assertFalse((markers / "chunk_0_of_2.json").is_file())
            self.assertTrue((markers / "chunk_1_of_2.json").is_file())

    def _tmpdir(self):
        import tempfile
        return tempfile.TemporaryDirectory()


@unittest.skipIf(STATS_MISSING, f"stats deps unavailable: {STATS_MISSING}")
class TestStatsOrder(unittest.TestCase):
    def test_null_sample_and_table_do_not_depend_on_dict_order(self):
        values = [((i % 7) - 3) * 0.013 for i in range(100_010)]
        tuples = [(0.0, v) for v in values]
        mid = len(tuples) // 2
        forward = [
            {(1, "cell_emb"): tuples[:mid], (2, "cell_emb"): tuples[:mid]},
            {(1, "cell_emb"): tuples[mid:], (2, "cell_emb"): tuples[mid:]},
        ]
        reverse = list(reversed(forward))
        single = [{(1, "cell_emb"): tuples, (2, "cell_emb"): tuples}]
        frame = pd.DataFrame(
            {"Gene": [2, 1], "Gene_name": ["B", "A"], "Ensembl_ID": ["E2", "E1"]}
        )
        states = {"state_key": "state", "start_state": "Disease", "goal_state": "Ctrl", "alt_states": []}

        def table(dict_list):
            out = isp_stats_to_goal_state(frame, dict_list, states, "all")
            return out.sort_values("Gene").reset_index(drop=True)

        left = table(forward)
        right = table(reverse)
        alone = table(single)
        self.assertTrue(left["Gene"].tolist() == right["Gene"].tolist() == alone["Gene"].tolist())
        self.assertTrue((left["N_Detections"] == right["N_Detections"]).all())
        self.assertTrue((left["Shift_to_goal_end"] - right["Shift_to_goal_end"]).abs().max() <= 1e-6)
        self.assertTrue((left["Goal_end_vs_random_pval"] == right["Goal_end_vs_random_pval"]).all())
        self.assertTrue((left["Goal_end_FDR"] == alone["Goal_end_FDR"]).all())
        self.assertTrue((left["Sig"] == alone["Sig"]).all())

        pooled = [t[1] for t in tuples]
        self.assertEqual(_stable_downsample(list(reversed(pooled))), _stable_downsample(pooled))

    def test_read_dictionaries_sorts_filenames(self):
        with self._tmpdir() as root:
            folder = Path(root)
            for name, token in (("b_raw.pickle", 2), ("a_raw.pickle", 1)):
                with open(folder / name, "wb") as handle:
                    pickle.dump({(token, "cell_emb"): [(0.0, 0.1)]}, handle)
            loaded = read_dictionaries(str(folder), "cell", None)
            genes = [next(iter(item))[0] for item in loaded]
            self.assertEqual(genes, [1, 2])

    def _tmpdir(self):
        import tempfile
        return tempfile.TemporaryDirectory()


if __name__ == "__main__":
    unittest.main()
