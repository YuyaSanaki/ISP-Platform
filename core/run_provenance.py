from __future__ import annotations

import os
import shutil
import subprocess
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _service_git_commit_env(service: str) -> str:
    """e.g. tokenize -> TOKENIZE_GIT_COMMIT, isp -> ISP_GIT_COMMIT."""
    return f"{service.upper()}_GIT_COMMIT"


def write_service_provenance(
    run_root: Path,
    config_path: Path | None = None,
    extra_meta: dict[str, Any] | None = None,
    input_paths: dict[str, str] | None = None,
    fast_fingerprint: bool = True,
    *,
    service: str = "tokenize",
    git_commit_env: str | None = None,
) -> dict[str, Any]:
    """
    Standardized provenance recorder for any service.

    For service ``tokenize``: writes ``tokenize_config_used.yaml`` and
    ``tokenize_run_metadata.yaml`` (same naming as the ISP artifacts).

    Args:
        run_root: Directory where metadata will be stored.
        config_path: Optional path to the YAML config used for this run.
        extra_meta: Optional dictionary of service-specific metadata.
        input_paths: Optional dict of {label: path} to fingerprint.
        fast_fingerprint: Whether to use fast metadata-based fingerprinting.
        service: Short name used in artifact filenames (default ``tokenize``).
        git_commit_env: Env var for an explicit commit (default ``{SERVICE}_GIT_COMMIT``).
    """
    run_root.mkdir(parents=True, exist_ok=True)
    meta_name = f"{service}_run_metadata.yaml"
    env_git = git_commit_env or _service_git_commit_env(service)

    if config_path and config_path.is_file():
        config_dest = run_root / f"{service}_config_used{config_path.suffix}"
        shutil.copy2(config_path, config_dest)

    meta: dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_source_path": str(config_path.resolve()) if config_path else None,
    }

    if extra_meta:
        meta.update(extra_meta)

    git_sha = os.environ.get(env_git, "").strip()
    if not git_sha:
        try:
            rev = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parent,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if rev.returncode == 0:
                git_sha = rev.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    if git_sha:
        meta["git_commit"] = git_sha

    if input_paths and os.environ.get("ENABLE_INPUT_FINGERPRINT", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        from run_input_fingerprint import fingerprint_local_path

        print(f"Computing input fingerprints ({'fast' if fast_fingerprint else 'full'} mode)...")
        fps = {}
        for label, path_str in input_paths.items():
            if path_str:
                fp = fingerprint_local_path(Path(path_str), fast=fast_fingerprint)
                fps[label] = fp
                digest = fp.get("content_fingerprint_sha256")
                print(f"  {label}: {digest or 'error'}")
        meta["input_content_fingerprints"] = fps

    with open(run_root / meta_name, "w", encoding="utf-8") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    return meta


def update_service_provenance(
    run_root: Path,
    updates: dict[str, Any],
    *,
    service: str = "tokenize",
) -> None:
    """Merge updates into ``{service}_run_metadata.yaml`` if it exists."""
    path = run_root / f"{service}_run_metadata.yaml"
    if not path.is_file():
        return
    with open(path, encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}
    if not isinstance(meta, dict):
        meta = {}
    meta.update(updates)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)
