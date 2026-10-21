"""Run metadata records Python and key package versions."""

from __future__ import annotations

import platform
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

import run_provenance  # noqa: E402


class TestEnvironmentVersions(unittest.TestCase):
    def test_versions_cover_recorded_packages(self):
        env = run_provenance.environment_versions()
        self.assertEqual(env["python"], platform.python_version())
        self.assertEqual(set(env["packages"]), set(run_provenance.RECORDED_PACKAGES))
        self.assertIsNotNone(env["packages"]["numpy"])

    def test_service_metadata_includes_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_provenance.write_service_provenance(Path(tmp), service="tokenize")
            meta = yaml.safe_load((Path(tmp) / "tokenize_run_metadata.yaml").read_text())
        self.assertEqual(meta["environment"]["python"], platform.python_version())

    def test_shared_pins_file_is_used_by_every_setup(self):
        for rel in ("Dockerfile", "scripts/macos/setup.sh", "H100/setup_env.sh"):
            self.assertIn("requirements-pins.txt", (ROOT / rel).read_text(), rel)
        pins = (ROOT / "requirements-pins.txt").read_text()
        self.assertIn("datasets==5.0.1", pins)
        self.assertIn('PYTHON_VERSION:-3.12', (ROOT / "H100/setup_env.sh").read_text())


if __name__ == "__main__":
    unittest.main()
