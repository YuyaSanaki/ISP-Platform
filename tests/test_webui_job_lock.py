"""Web UI job lock: one job across sessions, stale locks, stopping a job."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "webui") not in sys.path:
    sys.path.insert(0, str(ROOT / "webui"))

import job_lock  # noqa: E402


def _spawn(seconds: float = 60) -> subprocess.Popen:
    # Parent and child in one new session, like the Web UI's pipeline jobs.
    script = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', 'import time; time.sleep({seconds})']); "
        f"time.sleep({seconds})"
    )
    return subprocess.Popen([sys.executable, "-c", script], start_new_session=True)


class JobLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self._tmp.name)
        self.procs: list[subprocess.Popen] = []

    def tearDown(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                os.killpg(p.pid, 9)
                p.wait()
        self._tmp.cleanup()

    def _start(self, **meta) -> subprocess.Popen:
        ok, _ = job_lock.acquire_job_lock(self.ws, run_type="Pipeline (E2E)", **meta)
        self.assertTrue(ok)
        p = _spawn()
        self.procs.append(p)
        job_lock.mark_job_running(self.ws, p.pid)
        return p

    def test_second_acquire_is_refused_while_job_runs(self) -> None:
        p = self._start()
        ok, holder = job_lock.acquire_job_lock(self.ws, run_type="ISP UMAP")
        self.assertFalse(ok)
        self.assertEqual(holder["pid"], p.pid)
        self.assertEqual(holder["run_type"], "Pipeline (E2E)")
        self.assertEqual(job_lock.read_active_job(self.ws)["state"], "running")

    def test_lock_of_exited_job_is_stale(self) -> None:
        p = self._start()
        os.killpg(p.pid, 9)
        p.wait()
        self.assertIsNone(job_lock.read_active_job(self.ws))
        self.assertFalse((self.ws / job_lock.LOCK_FILE).exists())
        ok, _ = job_lock.acquire_job_lock(self.ws, run_type="ISP UMAP")
        self.assertTrue(ok)

    def test_zombie_job_is_not_alive(self) -> None:
        p = self._start()
        os.killpg(p.pid, 9)
        time.sleep(0.3)  # not reaped yet: zombie
        self.assertIsNone(job_lock.read_active_job(self.ws))
        p.wait()
        self.assertEqual(p.returncode, -9)

    def test_lock_from_other_host_or_reused_pid_is_stale(self) -> None:
        p = self._start()
        info = json.loads((self.ws / job_lock.LOCK_FILE).read_text())
        (self.ws / job_lock.LOCK_FILE).write_text(json.dumps({**info, "start_ticks": 1}))
        self.assertIsNone(job_lock.read_active_job(self.ws))
        job_lock.mark_job_running(self.ws, p.pid)
        info = json.loads((self.ws / job_lock.LOCK_FILE).read_text())
        (self.ws / job_lock.LOCK_FILE).write_text(json.dumps({**info, "hostname": "other"}))
        self.assertIsNone(job_lock.read_active_job(self.ws))

    def test_release_only_by_owner(self) -> None:
        p = self._start()
        job_lock.release_job_lock(self.ws, p.pid + 100000)
        self.assertIsNotNone(job_lock.read_active_job(self.ws))
        job_lock.release_job_lock(self.ws, p.pid)
        self.assertIsNone(job_lock.read_active_job(self.ws))

    def test_stop_kills_process_group(self) -> None:
        p = self._start()
        time.sleep(0.3)
        info = job_lock.read_active_job(self.ws)
        self.assertTrue(job_lock.stop_job(info, grace_s=5))
        p.wait(timeout=10)
        self.assertEqual(p.returncode, -15)
        with self.assertRaises(ProcessLookupError):
            os.killpg(p.pid, 0)

    def test_stop_refuses_lock_that_is_still_starting(self) -> None:
        ok, info = job_lock.acquire_job_lock(self.ws, run_type="Pipeline (E2E)")
        self.assertTrue(ok)
        self.assertEqual(info["pid"], os.getpid())
        self.assertFalse(job_lock.stop_job(info))
        job_lock.release_job_lock(self.ws, os.getpid())
        self.assertIsNone(job_lock.read_active_job(self.ws))


if __name__ == "__main__":
    unittest.main()
