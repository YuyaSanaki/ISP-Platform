"""Device selection (CUDA / MPS / CPU) and the /app path mapping for the native macOS install."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import compute_device as cd  # noqa: E402
import platform_paths as pp  # noqa: E402


class DeviceTests(unittest.TestCase):
    def _env(self, value: str | None):
        env = {} if value is None else {"ISP_DEVICE": value}
        return mock.patch.dict("os.environ", env, clear=False)

    def test_priority_cuda_then_mps_then_cpu(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("ISP_DEVICE", None)
            with mock.patch.object(cd.torch.cuda, "is_available", return_value=True):
                self.assertEqual(cd.device_type(), "cuda")
            with mock.patch.object(cd.torch.cuda, "is_available", return_value=False), \
                    mock.patch.object(cd, "_mps_available", return_value=True):
                self.assertEqual(cd.device_type(), "mps")
                self.assertTrue(cd.has_accelerator())
            with mock.patch.object(cd.torch.cuda, "is_available", return_value=False), \
                    mock.patch.object(cd, "_mps_available", return_value=False):
                self.assertEqual(cd.device_type(), "cpu")
                self.assertFalse(cd.has_accelerator())
                self.assertEqual(cd.device_name(), "CPU")

    def test_override(self):
        with self._env("cpu"), mock.patch.object(cd.torch.cuda, "is_available", return_value=True):
            self.assertEqual(cd.get_device().type, "cpu")
        with self._env("tpu"):
            with self.assertRaises(ValueError):
                cd.device_type()
        with self._env("mps"), mock.patch.object(cd, "_mps_available", return_value=False):
            with self.assertRaises(RuntimeError):
                cd.device_type()

    def test_is_oom(self):
        self.assertTrue(cd.is_oom(RuntimeError("MPS backend out of memory (MPS allocated: 9 GB)")))
        self.assertFalse(cd.is_oom(RuntimeError("shape mismatch")))
        self.assertFalse(cd.is_oom(ValueError("out of memory")))


class PathMappingTests(unittest.TestCase):
    def setUp(self):
        self.root = "/Users/me/ISP-Platform"
        patches = [
            mock.patch.dict("os.environ", {"GENEFORMER_ROOT": self.root}),
            mock.patch.object(pp, "needs_mapping", return_value=True),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_localize_path(self):
        self.assertEqual(pp.localize_path("/app"), self.root)
        self.assertEqual(pp.localize_path("/app/output/run1"), f"{self.root}/output/run1")
        self.assertEqual(pp.localize_path("/apple/x"), "/apple/x")
        self.assertEqual(pp.localize_path("data/app/x"), "data/app/x")

    def test_localize_nested_config(self):
        cfg = {"a": "/app/model", "b": ["/app/x", 3, "rel"], "c": {"d": "/application"}}
        self.assertEqual(
            pp.localize(cfg),
            {"a": f"{self.root}/model", "b": [f"{self.root}/x", 3, "rel"], "c": {"d": "/application"}},
        )

    def test_localize_text(self):
        text = 'output_root: /app/output  # under /app\nother: "/app/x"\nkeep: /apps/y /data/app/z\n'
        self.assertEqual(
            pp.localize_text(text),
            f'output_root: {self.root}/output  # under {self.root}\nother: "{self.root}/x"\n'
            "keep: /apps/y /data/app/z\n",
        )

    def test_no_mapping_inside_docker(self):
        with mock.patch.object(pp, "needs_mapping", return_value=False):
            self.assertEqual(pp.localize_path("/app/x"), "/app/x")
            self.assertEqual(pp.localize_text("/app/x"), "/app/x")


if __name__ == "__main__":
    unittest.main()
