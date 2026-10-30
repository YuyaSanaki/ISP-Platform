"""Fine-tune resume must not depend on the machine's torch.load default."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import run_finetune as ft  # noqa: E402


class CheckpointLoadTests(unittest.TestCase):
    def test_torch_floor_is_2_6(self):
        with mock.patch.object(ft.torch, "__version__", "2.5.1+cu121"):
            self.assertFalse(ft.torch_can_load_trainer_checkpoints())
        with mock.patch.object(ft.torch, "__version__", "2.6.0+cu124"):
            self.assertTrue(ft.torch_can_load_trainer_checkpoints())
        with mock.patch.object(ft.torch, "__version__", "2.7.0a0+nv"):
            self.assertTrue(ft.torch_can_load_trainer_checkpoints())

    def test_resume_forces_trusted_load_then_restores(self):
        os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        previous = ft.allow_trusted_checkpoint_resume()
        try:
            self.assertIsNone(previous)
            self.assertEqual(os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"], "1")
        finally:
            ft.restore_trusted_checkpoint_resume(previous)
        self.assertNotIn("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", os.environ)

    def test_resume_keeps_an_existing_value(self):
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "0"
        previous = ft.allow_trusted_checkpoint_resume()
        try:
            self.assertEqual(previous, "0")
            self.assertEqual(os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"], "1")
        finally:
            ft.restore_trusted_checkpoint_resume(previous)
        self.assertEqual(os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"], "0")


if __name__ == "__main__":
    unittest.main()
