"""
Rotating log file mirroring stdout/stderr (for ISP and tokenize entrypoints).

Set {ISP|TOKENIZE}_LOG_MAX_BYTES (default 50 MiB) and _LOG_BACKUP_COUNT (default 5).
Set ISP_DISABLE_RUN_LOG=1 or TOKENIZE_DISABLE_RUN_LOG=1 to skip tee.
"""
from __future__ import annotations

import atexit
import os
import sys
import threading
from pathlib import Path
from typing import Any, TextIO

_DEFAULT_MAX_BYTES = 50 * 1024 * 1024
_DEFAULT_BACKUP_COUNT = 5

_shared: _SharedRotatingTextLog | None = None
_saved_stdout: TextIO | None = None
_saved_stderr: TextIO | None = None


class _SharedRotatingTextLog:
    def __init__(self, log_path: Path, max_bytes: int, backup_count: int) -> None:
        self.path = Path(log_path)
        self.max_bytes = max(0, int(max_bytes))
        self.backup_count = max(0, int(backup_count))
        self._lock = threading.Lock()
        self._fp: TextIO | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._open_append()

    def _open_append(self) -> None:
        if self._fp:
            self._fp.close()
        self._fp = open(self.path, "a", encoding="utf-8", errors="replace")

    def _rotate_unlocked(self) -> None:
        if self._fp:
            self._fp.flush()
            self._fp.close()
            self._fp = None
        base = str(self.path)
        if not os.path.exists(base):
            self._open_append()
            return
        for i in range(self.backup_count - 1, 0, -1):
            src = f"{base}.{i}"
            dst = f"{base}.{i + 1}"
            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(src, dst)
        if self.backup_count > 0:
            dst1 = f"{base}.1"
            if os.path.exists(dst1):
                os.remove(dst1)
            os.rename(base, dst1)
        self._open_append()

    def _maybe_rotate_unlocked(self) -> None:
        if self.max_bytes <= 0 or self._fp is None:
            return
        try:
            self._fp.flush()
            if os.path.getsize(self.path) < self.max_bytes:
                return
        except OSError:
            return
        if self.backup_count <= 0:
            self._fp.close()
            self._fp = None
            try:
                os.remove(self.path)
            except OSError:
                pass
            self._open_append()
            return
        self._rotate_unlocked()

    def write(self, data: str) -> None:
        if not data or self._fp is None:
            return
        with self._lock:
            self._fp.write(data)
            self._fp.flush()
            self._maybe_rotate_unlocked()

    def close(self) -> None:
        with self._lock:
            if self._fp:
                self._fp.close()
                self._fp = None


class _TeeStream:
    __slots__ = ("_orig", "_log")

    def __init__(self, original: TextIO, log: _SharedRotatingTextLog) -> None:
        self._orig = original
        self._log = log

    def write(self, data: str) -> int:
        self._orig.write(data)
        self._orig.flush()
        self._log.write(data)
        return len(data)

    def flush(self) -> None:
        self._orig.flush()

    def isatty(self) -> bool:
        return self._orig.isatty()

    def fileno(self) -> int:
        return self._orig.fileno()

    def writable(self) -> bool:
        return True

    def __getattr__(self, name: str) -> Any:
        """Delegate TextIO attributes (encoding, errors, buffer, newlines, …) to the real stream."""
        return getattr(self._orig, name)


def _parse_size_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def install_rotating_stdio_tee(
    log_path: Path | str,
    *,
    env_prefix: str = "ISP",
    enabled: bool = True,
) -> Path | None:
    """
    Tee stdout and stderr to log_path with size-based rotation.
    env_prefix: read {PREFIX}_LOG_MAX_BYTES, {PREFIX}_LOG_BACKUP_COUNT,
    {PREFIX}_DISABLE_RUN_LOG=1 to skip.
    """
    global _shared, _saved_stdout, _saved_stderr

    if not enabled:
        return None
    if os.environ.get(f"{env_prefix}_DISABLE_RUN_LOG", "").lower() in ("1", "true", "yes"):
        return None
    if _shared is not None:
        return Path(log_path)

    path = Path(log_path)
    max_bytes = _parse_size_env(f"{env_prefix}_LOG_MAX_BYTES", _DEFAULT_MAX_BYTES)
    backup_count = _parse_size_env(f"{env_prefix}_LOG_BACKUP_COUNT", _DEFAULT_BACKUP_COUNT)

    _shared = _SharedRotatingTextLog(path, max_bytes, backup_count)
    _saved_stdout = sys.stdout
    _saved_stderr = sys.stderr
    sys.stdout = _TeeStream(_saved_stdout, _shared)
    sys.stderr = _TeeStream(_saved_stderr, _shared)
    atexit.register(restore_stdio)
    return path


def restore_stdio() -> None:
    global _shared, _saved_stdout, _saved_stderr
    if _saved_stdout is not None:
        sys.stdout = _saved_stdout
    if _saved_stderr is not None:
        sys.stderr = _saved_stderr
    _saved_stdout = None
    _saved_stderr = None
    if _shared is not None:
        _shared.close()
        _shared = None


def format_isp_run_banner(cfg: dict[str, Any], extras: dict[str, Any]) -> str:
    """Human-readable config summary for console + log."""
    paths = cfg.get("paths") or {}
    pert = cfg.get("perturbation") or {}
    mdl = cfg.get("model") or {}
    isp = cfg.get("isp") or {}
    st = cfg.get("stats") or {}
    rt = cfg.get("runtime") or {}
    analysis = cfg.get("analysis") or {}

    genes = pert.get("genes_to_perturb") or []
    genes_note = "all" if len(genes) == 0 else f"{len(genes)} listed"
    alt = pert.get("alt_states") or []
    alt_s = "(none)" if not alt else str(alt[:8]) + ("..." if len(alt) > 8 else "")

    lines = [
        "=" * 72,
        "ISP run configuration",
        "=" * 72,
        f"  config file:        {extras.get('config_path', '')}",
        f"  dataset:            {paths.get('dataset')}",
        f"  geneformer_model:   {paths.get('geneformer_model')}",
        f"  output_root:        {paths.get('output_root')}",
        f"  output date:        {extras.get('date_used', '')}",
        f"  run folder:         {extras.get('run_folder', '')}",
        f"  isp_results:        {extras.get('isp_out', '')}",
        f"  ispstats_results:   {extras.get('stats_out', '')}",
        f"  output_prefix:      {extras.get('output_prefix', '')}",
        "",
        "  perturbation:",
        f"    type:             {pert.get('type', 'delete')}",
        f"    organ_data:       {pert.get('organ_data', '')}",
        f"    state_key:        {pert.get('state_key', '')}",
        f"    start_state:      {pert.get('start_state', '')}",
        f"    end_state (goal): {pert.get('end_state', '')}",
        f"    alt_states:       {alt_s}",
        f"    genes_to_perturb: {genes_note}",
        "",
        "  model:",
        f"    type:             {mdl.get('type', '')}",
        f"    num_classes:      {mdl.get('num_classes', '')}",
        "",
        "  isp:",
        f"    max_ncells:       {isp.get('max_ncells')}",
        f"    emb_layer:        {isp.get('emb_layer')}",
        f"    emb_mode:         {isp.get('emb_mode')}",
        f"    cell_emb_style:   {isp.get('cell_emb_style')}",
        f"    combos:           {isp.get('combos')}",
        f"    anchor_gene:      {isp.get('anchor_gene')}",
        f"    filter_data:      {isp.get('filter_data')}",
        "",
        "  stats:",
        f"    mode:             {st.get('mode', '')}",
        "",
        "  runtime (effective):",
        f"    forward_batch_size: {extras.get('forward_batch_size', '')}",
        f"    nproc:              {extras.get('nproc', '')}",
        "",
        "  analysis:",
        f"    enabled:          {analysis.get('enabled', True)}",
        f"    skip_analysis:    {extras.get('skip_analysis', False)}",
        "=" * 72,
    ]
    return "\n".join(lines) + "\n"


def format_tokenize_run_banner(
    config_path: Path,
    data_cfg: dict[str, Any],
    tokenizer_cfg: dict[str, Any],
    single_cell: dict[str, Any],
) -> str:
    cust = tokenizer_cfg.get("custom_attr_name_dict") or {}
    cust_keys = ", ".join(sorted(cust.keys())) if cust else "(none)"
    lines = [
        "=" * 72,
        "Tokenize pipeline configuration",
        "=" * 72,
        f"  config file:        {config_path}",
        f"  input_type:         {data_cfg.get('input_type')}",
        f"  input_dir:          {data_cfg.get('input_dir')}",
        f"  loom_temp_dir:      {data_cfg.get('loom_temp_dir')}",
        f"  output_dir:         {data_cfg.get('output_dir')}",
        f"  output_prefix:      {data_cfg.get('output_prefix')}",
        "",
        "  tokenizer:",
        f"    nproc:            {tokenizer_cfg.get('nproc', 1)}",
        f"    custom_attrs:     {cust_keys}",
        "",
        "  single_cell_settings:",
        f"    extract_metadata_from_path: {single_cell.get('extract_metadata_from_path')}",
        "=" * 72,
    ]
    return "\n".join(lines) + "\n"