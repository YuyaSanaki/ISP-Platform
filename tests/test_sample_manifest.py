"""Metadata manifest: QIIME2-style TSV overrides folder-name labels."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "contracts"))


MANIFEST = """\
sample-id\tdisease\tspecies\texperiment\tbatch
#q2:types\tcategorical\tcategorical\tcategorical\tcategorical
20211122_ME10x01_S01_Mat\tMatrigel\thuman\tSomatoid\tS01
20211122_ME10x01_S02_NoMat\tNoMatrigel\tHuman\tSomatoid\tS02
"""


def _touch_10x(sample: Path) -> None:
    sample.mkdir(parents=True)
    for name in ("matrix.mtx.gz", "barcodes.tsv.gz", "features.tsv.gz"):
        (sample / name).write_bytes(b"stub")


class TestLoadSampleManifest(unittest.TestCase):
    def test_required_and_extra_columns(self):
        from sample_manifest import load_sample_manifest

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.tsv"
            path.write_text(MANIFEST, encoding="utf-8")
            manifest = load_sample_manifest(path)
            self.assertEqual(
                manifest.columns,
                ("sample_id", "disease", "species", "experiment", "batch"),
            )
            row = manifest.by_folder["20211122_ME10x01_S01_Mat"]
            self.assertEqual(row["disease"], "Matrigel")
            self.assertEqual(row["species"], "human")
            self.assertEqual(row["experiment"], "Somatoid")
            self.assertEqual(row["batch"], "S01")
            self.assertEqual(manifest.by_folder["20211122_ME10x01_S02_NoMat"]["species"], "human")

    def test_state_alias_and_filepath(self):
        from sample_manifest import load_sample_manifest

        text = (
            "sample-id,filepath,state,species,experiment,donor\n"
            "S01,20211122_ME10x01_S01_Mat,Matrigel,human,Somatoid,D1\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.tsv"
            path.write_text(text, encoding="utf-8")
            manifest = load_sample_manifest(path)
            row = manifest.by_folder["20211122_ME10x01_S01_Mat"]
            self.assertEqual(row["sample_id"], "S01")
            self.assertEqual(row["disease"], "Matrigel")
            self.assertEqual(row["donor"], "D1")
            self.assertNotIn("filepath", row)

    def test_experiment_is_optional(self):
        from sample_manifest import load_sample_manifest, samples_to_metadata_tsv, validate_sample_metadata

        text = (
            "sample-id\tdisease\tspecies\n"
            "20211122_ME10x01_S01_Mat\tMatrigel\thuman\n"
            "20211122_ME10x01_S02_NoMat\tNoMatrigel\thuman\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.tsv"
            path.write_text(text, encoding="utf-8")
            manifest = load_sample_manifest(path)
            self.assertNotIn("experiment", manifest.columns)
            self.assertEqual(manifest.by_folder["20211122_ME10x01_S01_Mat"]["disease"], "Matrigel")
        samples = [
            {"sample_id": "A", "disease": "Matrigel", "species": "human"},
            {"sample_id": "B", "disease": "NoMatrigel", "species": "Human"},
        ]
        self.assertIsNone(validate_sample_metadata(samples))
        tsv = samples_to_metadata_tsv(["disease", "species"], samples)
        self.assertIn("sample-id\tdisease\tspecies\n", tsv)
        self.assertIn("B\tNoMatrigel\thuman\n", tsv)
        self.assertIsNotNone(validate_sample_metadata([
            {"sample_id": "A", "disease": "", "species": "human"},
        ]))

    def test_rejects_bad_species_and_duplicates(self):
        from sample_manifest import SampleManifestError, load_sample_manifest

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.tsv"
            path.write_text(
                "sample-id\tdisease\tspecies\texperiment\n"
                "A\tMatrigel\tfly\tSomatoid\n",
                encoding="utf-8",
            )
            with self.assertRaises(SampleManifestError):
                load_sample_manifest(path)
            path.write_text(
                "sample-id\tdisease\tspecies\texperiment\n"
                "A\tMatrigel\thuman\tSomatoid\n"
                "A\tNoMatrigel\thuman\tSomatoid\n",
                encoding="utf-8",
            )
            with self.assertRaises(SampleManifestError):
                load_sample_manifest(path)

    def test_two_filenames_is_an_error(self):
        from sample_manifest import SampleManifestError, find_sample_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metadata.tsv").write_text("sample-id\n", encoding="utf-8")
            (root / "manifest.tsv").write_text("sample-id\n", encoding="utf-8")
            with self.assertRaises(SampleManifestError):
                find_sample_manifest(root)


class TestManifestStates(unittest.TestCase):
    def test_disease_comes_from_manifest_not_folder_name(self):
        from data_input_layout import (
            column_values_from_study,
            label_column_candidates_from_study,
            unique_states_from_samples,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch_10x(root / "20211122_ME10x01_S01_Mat")
            _touch_10x(root / "20211122_ME10x01_S02_NoMat")
            (root / "metadata.tsv").write_text(MANIFEST, encoding="utf-8")
            self.assertEqual(unique_states_from_samples(root), ["Matrigel", "NoMatrigel"])
            self.assertEqual(column_values_from_study(root, "experiment"), ["Somatoid"])
            self.assertEqual(column_values_from_study(root, "batch"), ["S01", "S02"])
            labels = label_column_candidates_from_study(root)
            self.assertIn("experiment", labels)
            self.assertIn("batch", labels)

    def test_folder_names_still_work_without_manifest(self):
        from data_input_layout import unique_states_from_samples

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch_10x(root / "1w-AD-Rep1")
            _touch_10x(root / "1w-WT-Rep1")
            self.assertEqual(unique_states_from_samples(root), ["AD", "WT"])


class TestManifestLoomConversion(unittest.TestCase):
    def _adata(self):
        import anndata as ad
        import scipy.sparse as sp

        adata = ad.AnnData(sp.csr_matrix(np.ones((2, 1), dtype=np.float32)))
        adata.var_names = ["ENSG00000000001"]
        return adata

    def test_writes_manifest_columns_and_checks_species(self):
        from execute_tokenizer_pipeline import process_single_cell_to_loom

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch_10x(root / "20211122_ME10x01_S01_Mat")
            _touch_10x(root / "20211122_ME10x01_S02_NoMat")
            (root / "metadata.tsv").write_text(MANIFEST, encoding="utf-8")
            cfg = {"custom_attr_name_dict": {"disease": "disease"}}
            written: list = []

            def _capture(adata, path):
                written.append((Path(path).name, {col: str(adata.obs[col].iloc[0]) for col in adata.obs.columns}))

            with mock.patch(
                "execute_tokenizer_pipeline.sc.read_10x_mtx",
                side_effect=lambda *args, **kwargs: self._adata(),
            ), mock.patch(
                "execute_tokenizer_pipeline.write_loom_safe",
                side_effect=_capture,
            ), mock.patch(
                "celltype_annotate_expression.apply_pre_isp_celltype_annotation",
                side_effect=lambda adata, *args, **kwargs: adata,
            ):
                n = process_single_cell_to_loom(
                    str(root),
                    str(root / "loom_files"),
                    {"extract_metadata_from_path": True},
                    cfg,
                    species={"model_organism": "human"},
                )
            self.assertEqual(n, 2)
            self.assertIn("experiment", cfg["custom_attr_name_dict"])
            self.assertIn("batch", cfg["custom_attr_name_dict"])
            by_name = dict(written)
            mat = by_name["20211122_ME10x01_S01_Mat.loom"]
            self.assertEqual(mat["disease"], "Matrigel")
            self.assertEqual(mat["experiment"], "Somatoid")
            self.assertEqual(mat["species"], "human")
            self.assertNotIn("time", mat)

            with mock.patch(
                "execute_tokenizer_pipeline.sc.read_10x_mtx",
                side_effect=lambda *args, **kwargs: self._adata(),
            ), mock.patch(
                "execute_tokenizer_pipeline.write_loom_safe",
            ), mock.patch(
                "celltype_annotate_expression.apply_pre_isp_celltype_annotation",
                side_effect=lambda adata, *args, **kwargs: adata,
            ):
                with self.assertRaises(SystemExit) as ctx:
                    process_single_cell_to_loom(
                        str(root),
                        str(root / "loom_files"),
                        {"extract_metadata_from_path": True},
                        {"custom_attr_name_dict": {}},
                        species={"model_organism": "mouse"},
                    )
            self.assertIn("model_organism", str(ctx.exception))

    def test_missing_sample_row_exits(self):
        from execute_tokenizer_pipeline import process_single_cell_to_loom

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch_10x(root / "20211122_ME10x01_S01_Mat")
            _touch_10x(root / "extra_sample")
            (root / "metadata.tsv").write_text(
                "sample-id\tdisease\tspecies\texperiment\n"
                "20211122_ME10x01_S01_Mat\tMatrigel\thuman\tSomatoid\n",
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit) as ctx:
                process_single_cell_to_loom(
                    str(root),
                    str(root / "loom_files"),
                    {},
                    {"custom_attr_name_dict": {}},
                    species={"model_organism": "human"},
                )
            self.assertIn("extra_sample", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
