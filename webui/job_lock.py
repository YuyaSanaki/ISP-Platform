"""One Web UI job at a time across all browser sessions, and stopping it.

The lock is a JSON file in the Web UI workspace, not session state, so a second
tab, a reload or a new session sees the running job and cannot start another.
A lock whose process has exited (or was written by another container) is stale
and is cleared on the next read.
"""
from __future__ import annotations

import fcntl
import json
import os
import signal
import socket
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

LOCK_FILE = "active_job.json"
_MUTEX_FILE = "active_job.lock"


def _psutil_start_ticks(pid: int) -> int | None:
    """Start time of ``pid`` in 1/100 s since the epoch (no ``/proc``, e.g. macOS)."""
    import psutil

    try:
        proc = psutil.Process(pid)
        if proc.status() == psutil.STATUS_ZOMBIE:
            return None
        return int(round(proc.create_time() * 100))
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return None
    except psutil.AccessDenied:
        return None


def _proc_start_ticks(pid: int) -> int | None:
    """Start time of ``pid`` in clock ticks since boot; None if gone or a zombie."""
    if not Path("/proc").is_dir():
        return _psutil_start_ticks(pid)
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # Field 2 (comm) may contain spaces and parentheses; the fields after the
    # last ')' are fixed: state is field 3, starttime is field 22.
    fields = stat.rsplit(")", 1)[1].split()
    if fields[0] == "Z":
        return None
    return int(fields[19])


def pid_alive(pid: int, start_ticks: int | None = None) -> bool:
    """True if ``pid`` runs and, when ``start_ticks`` is given, is the same process.

    Does not reap: the Popen that owns the child must collect its exit code.
    """
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    ticks = _proc_start_ticks(pid)
    if ticks is None:
        return False
    return start_ticks is None or ticks == start_ticks


@contextmanager
def _mutex(workspace: Path) -> Iterator[None]:
    workspace.mkdir(parents=True, exist_ok=True)
    with open(workspace / _MUTEX_FILE, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _read_raw(workspace: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((workspace / LOCK_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(workspace: Path, info: dict[str, Any]) -> None:
    tmp = workspace / (LOCK_FILE + ".tmp")
    tmp.write_text(json.dumps(info, indent=2), encoding="utf-8")
    os.replace(tmp, workspace / LOCK_FILE)


def _is_live(info: dict[str, Any]) -> bool:
    if info.get("hostname") != socket.gethostname():
        return False
    return pid_alive(info.get("pid", -1), info.get("start_ticks"))


def _read_live(workspace: Path) -> dict[str, Any] | None:
    info = _read_raw(workspace)
    if info is None:
        return None
    if _is_live(info):
        return info
    (workspace / LOCK_FILE).unlink(missing_ok=True)
    return None


def read_active_job(workspace: Path) -> dict[str, Any] | None:
    """The running job's lock record, or None (stale locks are removed)."""
    workspace = Path(workspace)
    with _mutex(workspace):
        return _read_live(workspace)


def acquire_job_lock(
    workspace: Path, **meta: Any
) -> tuple[bool, dict[str, Any] | None]:
    """Take the lock before starting a job.

    Returns ``(True, record)`` on success, ``(False, holder)`` if a job runs.
    Until :func:`mark_job_running` the record names this (Web UI) process, so a
    crash between acquire and spawn leaves a lock that clears when it exits.
    """
    workspace = Path(workspace)
    with _mutex(workspace):
        holder = _read_live(workspace)
        if holder is not None:
            return False, holder
        pid = os.getpid()
        info = {
            **meta,
            "state": "starting",
            "pid": pid,
            "start_ticks": _proc_start_ticks(pid),
            "hostname": socket.gethostname(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write(workspace, info)
        return True, info


def mark_job_running(workspace: Path, pid: int) -> None:
    """Point the lock at the spawned job (its own session / process group)."""
    workspace = Path(workspace)
    with _mutex(workspace):
        info = _read_raw(workspace) or {}
        info.update(
            state="running",
            pid=pid,
            start_ticks=_proc_start_ticks(pid),
            hostname=socket.gethostname(),
        )
        _write(workspace, info)


def release_job_lock(workspace: Path, pid: int) -> None:
    """Remove the lock if it still belongs to ``pid``."""
    workspace = Path(workspace)
    with _mutex(workspace):
        info = _read_raw(workspace)
        if info is not None and info.get("pid") == pid:
            (workspace / LOCK_FILE).unlink(missing_ok=True)


def stop_job(info: dict[str, Any], grace_s: float = 15.0) -> bool:
    """SIGTERM the job's process group, then SIGKILL after ``grace_s`` seconds.

    Jobs are started with ``start_new_session=True``, so the group id equals the
    job pid and includes its children (tokenize / fine-tune / accelerate workers).
    Returns False if the lock does not describe a live spawned job.
    """
    pid = info.get("pid")
    if info.get("state") != "running" or not pid_alive(pid, info.get("start_ticks")):
        return False
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return False
    if pgid != pid:
        return False
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not pid_alive(pid, info.get("start_ticks")):
            break
        time.sleep(0.25)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        # macOS raises EPERM when only zombies are left in the group.
        pass
    return True
