"""Pinned Ensembl mouse↔human ortholog tables: checksums, provenance and install script."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORTHO = ROOT / "core" / "geneformer" / "dicts" / "orthologs"
INSTALL = ROOT / "scripts" / "download_mouse_human_orthologs.sh"
DOWNLOAD_BUILD = ROOT / "scripts" / "download_build_assets.sh"
TABLES = ("human_to_mouse.tsv", "mouse_to_human.tsv")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sums() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (ORTHO / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        out[name.strip()] = digest
    return out


class PinnedTableTests(unittest.TestCase):
    def test_tables_match_sha256sums(self) -> None:
        sums = _sums()
        self.assertEqual(set(sums), set(TABLES))
        for name in TABLES:
            self.assertEqual(_sha256(ORTHO / name), sums[name], name)

    def test_manifest_matches_sha256sums_and_rows(self) -> None:
        manifest = json.loads((ORTHO / "ensembl_release.json").read_text())
        self.assertEqual(manifest["release"], 116)
        self.assertEqual(manifest["retrieved_at"], "2026-08-07")
        sums = _sums()
        for name in TABLES:
            entry = manifest["tables"][name]
            self.assertEqual(entry["sha256"], sums[name])
            rows = (ORTHO / name).read_text().splitlines()
            self.assertEqual(rows[0], "source_id\ttarget_id\torthology_type")
            self.assertEqual(len(rows) - 1, entry["rows"])

    def test_human_to_mouse_is_column_swap(self) -> None:
        def pairs(name: str) -> set[tuple[str, str, str]]:
            lines = (ORTHO / name).read_text().splitlines()[1:]
            return {tuple(line.split("\t")) for line in lines}  # type: ignore[misc]

        m2h = pairs("mouse_to_human.tsv")
        h2m = pairs("human_to_mouse.tsv")
        self.assertEqual(h2m, {(t, s, o) for s, t, o in m2h})

    def test_pou5f1_rows_are_one2many(self) -> None:
        rows = {
            tuple(line.split("\t")[:2]): line.split("\t")[2]
            for line in (ORTHO / "human_to_mouse.tsv").read_text().splitlines()[1:]
        }
        self.assertEqual(rows[("ENSG00000204531", "ENSMUSG00000024406")], "ortholog_one2many")
        self.assertEqual(rows[("ENSG00000212993", "ENSMUSG00000024406")], "ortholog_one2many")


class InstallScriptTests(unittest.TestCase):
    def _run(self, script: Path, dest: Path, **extra: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.pop("ORTHOLOG_REFRESH", None)
        env["GENEFORMER_ROOT"] = str(dest)
        env.update(extra)
        return subprocess.run(
            ["bash", str(script)], cwd=str(ROOT), env=env,
            capture_output=True, text=True, check=False,
        )

    def test_installs_and_verifies_pinned_tables_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "assets"
            proc = self._run(INSTALL, dest)
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            self.assertIn("Pinned Ensembl ortholog tables verified", proc.stdout)
            out = dest / "core" / "geneformer" / "dicts" / "orthologs"
            for name in TABLES:
                self.assertEqual(_sha256(out / name), _sha256(ORTHO / name))

    def test_download_models_none_seeds_pinned_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "assets"
            proc = self._run(DOWNLOAD_BUILD, dest, DOWNLOAD_MODELS="none")
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            out = dest / "core" / "geneformer" / "dicts" / "orthologs"
            for name in (*TABLES, "SHA256SUMS", "ensembl_release.json"):
                self.assertTrue((out / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
