"""
Stable content fingerprints for local dataset / model directories (or single files).

Walks all regular files under a path, hashes each file with SHA-256, then hashes the
sorted list of ``relative_path\\0hexdigest`` lines. Same bytes at the same relative
paths yield the same fingerprint even if mtimes differ.

Enable from ISP via environment variable ``ISP_FINGERPRINT_INPUTS`` (see run_isp.py).
Large trees can take noticeable time and I/O.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fingerprint_local_path(root: Path, fast: bool = False) -> dict[str, Any]:
    """
    Return a JSON/YAML-serializable dict with content_fingerprint_sha256, or error info.

    If fast=True, it hashes:
      1. Directory structure (names, sizes, mtimes).
      2. Full content of small key files (config.json, dataset_info.json, etc.).
    Otherwise (fast=False), it hashes the full content of ALL files.
    """
    try:
        resolved = root.expanduser().resolve()
    except OSError as e:
        return {"path": str(root), "error": f"resolve_failed: {e}"}

    if not resolved.exists():
        return {"path": str(resolved), "error": "missing"}

    # Key files that we always want to hash fully if we are in fast mode
    KEY_FILES = {
        "config.json",
        "dataset_info.json",
        "state.json",
        "tokenizer_config.json",
        "generation_config.json",
        "special_tokens_map.json",
    }

    if resolved.is_file():
        digest = _sha256_file(resolved)
        return {
            "path": str(resolved),
            "kind": "file",
            "file_count": 1,
            "content_fingerprint_sha256": digest,
        }

    if not resolved.is_dir():
        return {"path": str(resolved), "error": "not_file_or_dir"}

    entries: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(resolved, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames):
            p = Path(dirpath) / name
            try:
                if not p.is_file():
                    continue
                rel = p.relative_to(resolved).as_posix()
                
                if fast:
                    # For fast mode: hash metadata + content only if it's a key small file
                    stat = p.stat()
                    meta_str = f"{rel}|{stat.st_size}|{stat.st_mtime}"
                    if name.lower() in KEY_FILES:
                        file_hash = _sha256_file(p)
                    else:
                        # Just use a placeholder for large files in fast mode
                        file_hash = "metadata_only"
                    entries.append((meta_str, file_hash))
                else:
                    # Full content hashing
                    entries.append((rel, _sha256_file(p)))
            except OSError as e:
                return {
                    "path": str(resolved),
                    "kind": "directory",
                    "error": f"read_failed: {p}: {e}",
                }

    combiner = hashlib.sha256()
    # Sort entries by the first element of the tuple (relative path or meta_str)
    for entry_key, fh in sorted(entries, key=lambda x: x[0]):
        combiner.update(entry_key.encode("utf-8"))
        combiner.update(b"\0")
        combiner.update(fh.encode("ascii"))
        combiner.update(b"\n")

    return {
        "path": str(resolved),
        "kind": "directory",
        "file_count": len(entries),
        "fingerprint_mode": "fast" if fast else "full",
        "content_fingerprint_sha256": combiner.hexdigest(),
    }


def fingerprint_isp_inputs(dataset: str, model_dir: str, fast: bool = False) -> dict[str, Any]:
    return {
        "dataset": fingerprint_local_path(Path(dataset), fast=fast),
        "geneformer_model": fingerprint_local_path(Path(model_dir), fast=fast),
    }
