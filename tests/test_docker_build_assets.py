"""Tests for Docker build-asset orchestration and entrypoint seed behavior."""
from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_BUILD = ROOT / "scripts" / "download_build_assets.sh"
ENTRYPOINT = ROOT / "scripts" / "container-entrypoint.sh"


class DownloadBuildAssetsTests(unittest.TestCase):
    def test_none_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "assets"
            env = os.environ.copy()
            env["GENEFORMER_ROOT"] = str(dest)
            env["DOWNLOAD_MODELS"] = "none"
            proc = subprocess.run(
                ["bash", str(DOWNLOAD_BUILD)],
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("Skipping downloads", proc.stdout)
            self.assertTrue((dest / "models").is_dir())
            self.assertTrue((dest / "core" / "geneformer" / "dicts" / "mouse").is_dir())
            curated_src = (
                ROOT / "core" / "geneformer" / "dicts" / "orthologs" / "mouse_to_human_curated.tsv"
            )
            curated_dst = (
                dest / "core" / "geneformer" / "dicts" / "orthologs" / "mouse_to_human_curated.tsv"
            )
            if curated_src.is_file():
                self.assertTrue(curated_dst.is_file())

    def test_rejects_unknown_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env["GENEFORMER_ROOT"] = str(Path(tmp) / "assets")
            env["DOWNLOAD_MODELS"] = "not-a-profile"
            proc = subprocess.run(
                ["bash", str(DOWNLOAD_BUILD)],
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("unknown DOWNLOAD_MODELS", proc.stderr)


class EntrypointSeedTests(unittest.TestCase):
    def _run_entrypoint(self, assets: Path, app: Path, *cmd: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmp:
            ep = Path(tmp) / "ep.sh"
            text = ENTRYPOINT.read_text().replace("/app", str(app))
            ep.write_text(text)
            ep.chmod(ep.stat().st_mode | stat.S_IEXEC)
            env = os.environ.copy()
            env["GENEFORMER_IMAGE_ASSETS"] = str(assets)
            env["SKIP_MODEL_SEED"] = "0"
            return subprocess.run(
                [str(ep), *cmd],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

    def test_preserves_host_override_and_seeds_missing_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "opt"
            app = root / "app"
            baked = assets / "models" / "mouse-Geneformer"
            baked.mkdir(parents=True)
            (baked / "config.json").write_text('{"architectures":["BertForMaskedLM"]}\n')
            (baked / "model.safetensors").write_bytes(b"fake-weights")
            dict_src = assets / "core" / "geneformer" / "dicts" / "mouse"
            dict_src.mkdir(parents=True)
            (dict_src / "MLM-re_token_dictionary_v1.pkl").write_bytes(b"tok")

            host_model = app / "models" / "mouse-Geneformer"
            host_model.mkdir(parents=True)
            (host_model / "config.json").write_text('{"host":true}\n')
            (host_model / "model.safetensors").write_bytes(b"host-weights")
            (app / "core" / "geneformer" / "dicts" / "mouse").mkdir(parents=True)

            marker = root / "ran"
            proc = self._run_entrypoint(assets, app, "touch", str(marker))
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            self.assertTrue(marker.is_file())
            self.assertEqual((host_model / "config.json").read_text(), '{"host":true}\n')
            self.assertEqual((host_model / "model.safetensors").read_bytes(), b"host-weights")
            self.assertEqual(
                (app / "core" / "geneformer" / "dicts" / "mouse" / "MLM-re_token_dictionary_v1.pkl").read_bytes(),
                b"tok",
            )

    def test_seeds_missing_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "opt"
            app = root / "app"
            baked = assets / "models" / "mouse-Geneformer"
            baked.mkdir(parents=True)
            (baked / "config.json").write_text("{}\n")
            (baked / "model.safetensors").write_bytes(b"img")
            (app / "models").mkdir(parents=True)

            proc = self._run_entrypoint(assets, app, "true")
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            dst = app / "models" / "mouse-Geneformer" / "model.safetensors"
            self.assertEqual(dst.read_bytes(), b"img")


if __name__ == "__main__":
    unittest.main()
