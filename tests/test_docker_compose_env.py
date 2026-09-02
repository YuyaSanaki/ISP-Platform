"""docker-compose.yml must preserve base env vars when services extend environment."""
from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.yml"

BASE_ENV_KEYS = frozenset(
    {"LD_PRELOAD", "PYTHONPATH", "SKIP_MODEL_SEED", "GENEFORMER_IMAGE_ASSETS"}
)


class DockerComposeEnvTests(unittest.TestCase):
    def test_services_with_environment_override_merge_base_env(self) -> None:
        doc = yaml.safe_load(COMPOSE.read_text())
        for name, svc in doc["services"].items():
            env = svc.get("environment")
            if not isinstance(env, dict):
                continue
            missing = BASE_ENV_KEYS - set(env.keys())
            self.assertFalse(
                missing,
                f"service {name!r} environment missing base keys: {sorted(missing)}",
            )

    def test_isp_honors_skip_model_seed_from_host_env(self) -> None:
        proc = subprocess.run(
            ["docker", "compose", "config"],
            cwd=str(ROOT),
            env={**__import__("os").environ, "SKIP_MODEL_SEED": "1"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        cfg = yaml.safe_load(proc.stdout)
        self.assertEqual(cfg["services"]["isp"]["environment"]["SKIP_MODEL_SEED"], "1")


if __name__ == "__main__":
    unittest.main()
