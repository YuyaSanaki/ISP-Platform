"""Pick a batch size by measuring the real model on the current GPU.

Two ceilings matter: GPU memory, and the batch beyond which throughput stops
improving (on unified-memory boxes such as GB10 the second one usually binds
first, so sizing from free memory alone picks a needlessly large batch). The
probe below measures both, then keeps the smallest batch whose throughput is
within `min_gain` of the best it saw.

On NVIDIA GB10 / Grace-Blackwell unified memory, ``nvidia-smi`` reports memory
as "Not Supported" / N/A and ``torch.cuda.mem_get_info()`` often under-reports
free bytes (~half the pool) even at 0% util with no GPU processes. Busy checks
and memory budgets must not treat that soft under-report as a busy GPU.

Results are cached per (GPU, model, sequence length, dtype) because probing
costs about a minute; set GENEFORMER_BATCH_SIZE_CACHE=off to always re-probe.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATES: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024)
# Fine-tune probe ladder — starts at the historical platform default (6), not 2.
DEFAULT_TRAIN_CANDIDATES: tuple[int, ...] = (6, 8, 12, 16, 24, 32, 48, 64)
# Long-sequence models (human Geneformer V2, max_input_size=4096): attention
# activations dominate; the historical ladder starting at 6 overshoots GB10/H100.
LONG_SEQ_TRAIN_CANDIDATES: tuple[int, ...] = (2, 3, 4, 6, 8, 12, 16, 24)
DEFAULT_MEMORY_FRACTION = 0.8
# Training probe omits some Trainer/accelerate overhead; keep extra headroom
# especially when seq_len > 2048 (Mouse→Human / human V2 fine-tunes).
DEFAULT_TRAIN_MEMORY_FRACTION = 0.85
LONG_SEQ_TRAIN_MEMORY_FRACTION = 0.55
LONG_SEQ_INPUT_SIZE = 2048
DEFAULT_MIN_GAIN = 0.05
# Soft note only when torch free accounting is trusted.
DEFAULT_BUSY_FREE_FRACTION = 0.6
# require_idle: treat as busy when compute util is at/above this (%).
DEFAULT_BUSY_UTIL_PCT = 15.0
# Critical free memory (only when torch free accounting is trusted).
DEFAULT_CRITICAL_FREE_FRACTION = 0.10
DEFAULT_CRITICAL_FREE_BYTES = 4 * (1024**3)


class BusyGPUError(RuntimeError):
    """Raised when batch calibration refuses to run on a contended GPU."""


@dataclass(frozen=True)
class GpuMemorySnapshot:
    """GPU memory / util snapshot used by calibrate gating and budgeting."""

    free: int
    total: int
    allocated: int
    util_pct: float | None
    # False on GB10-class devices where nvidia-smi memory is N/A / Not Supported.
    free_accounting_reliable: bool

    @property
    def usable(self) -> int:
        return max(int(self.total - self.allocated), 1)


def is_auto(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "auto"


def train_candidates_for_seq_len(seq_len: int) -> tuple[int, ...]:
    """Candidate ladder for FT calibrate / training.batch_size=auto.

    Short models (mouse Geneformer, 2048) keep the historical 6+ ladder.
    Long models (human V2, 4096 — typical Mouse→Human / Human→Human V2) start
    at 2 so calibration cannot recommend a batch that only fits the optimistic
    forward+backward probe.
    """
    if int(seq_len) > LONG_SEQ_INPUT_SIZE:
        return LONG_SEQ_TRAIN_CANDIDATES
    return DEFAULT_TRAIN_CANDIDATES


def train_memory_fraction_for_seq_len(seq_len: int) -> float:
    """Memory budget fraction for FT training probes."""
    if int(seq_len) > LONG_SEQ_INPUT_SIZE:
        return LONG_SEQ_TRAIN_MEMORY_FRACTION
    return DEFAULT_TRAIN_MEMORY_FRACTION


def coerce_batch_size(value: Any, *, default: int) -> int | str:
    """Pass "auto" through untouched; otherwise parse an int."""
    if is_auto(value):
        return "auto"
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("Invalid batch size %r; using %d.", value, default)
        return default


def gpu_compute_utilization_pct() -> float | None:
    """Return GPU compute utilization 0–100, or None if unavailable."""
    if not torch.cuda.is_available():
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            return float(line.split(",")[0].strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    util_fn = getattr(torch.cuda, "utilization", None)
    if callable(util_fn):
        try:
            return float(util_fn(0))
        except Exception:  # noqa: BLE001 - optional probe
            return None
    return None


def nvidia_smi_memory_supported() -> bool | None:
    """True if nvidia-smi reports numeric memory; False if N/A; None if unknown."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.strip().splitlines()[0] if out.strip() else ""
    if not line:
        return None
    # GB10: "[N/A], [N/A]" or "Not Supported"
    lowered = line.lower()
    if "n/a" in lowered or "not supported" in lowered:
        return False
    parts = [p.strip() for p in line.split(",")]
    try:
        float(parts[0])
        return True
    except (ValueError, IndexError):
        return False


def read_gpu_memory_snapshot() -> GpuMemorySnapshot:
    """Read free/total/util and whether torch free bytes can be trusted."""
    if not torch.cuda.is_available():
        return GpuMemorySnapshot(0, 0, 0, None, False)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    allocated = int(torch.cuda.memory_allocated())
    util = gpu_compute_utilization_pct()
    smi_ok = nvidia_smi_memory_supported()
    # When nvidia-smi cannot report memory, torch free/total ratios are unreliable
    # on unified-memory GPUs (often ~50% "free" while idle with no processes).
    reliable = True if smi_ok is True else False if smi_ok is False else True
    return GpuMemorySnapshot(
        free=int(free),
        total=int(total),
        allocated=allocated,
        util_pct=util,
        free_accounting_reliable=reliable,
    )


def calibration_memory_budget(
    snapshot: GpuMemorySnapshot,
    *,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
) -> int:
    """Bytes available for the probe.

    When free accounting is unreliable (GB10), budget from device total minus
    our own allocation — not from the under-reported torch free counter.
    """
    if snapshot.free_accounting_reliable:
        return snapshot.allocated + int(memory_fraction * snapshot.free)
    return snapshot.allocated + int(memory_fraction * snapshot.usable)


def gpu_memory_contended(
    snapshot: GpuMemorySnapshot | None = None,
    *,
    free_fraction: float = DEFAULT_BUSY_FREE_FRACTION,
) -> bool:
    """Soft signal: trusted free memory is below ``free_fraction`` of usable.

    Always False when free accounting is unreliable (do not warn on GB10 idle).
    """
    snap = snapshot or read_gpu_memory_snapshot()
    if not snap.free_accounting_reliable or snap.total <= 0:
        return False
    return snap.free < free_fraction * snap.usable


def gpu_is_busy_for_calibration(
    snapshot: GpuMemorySnapshot | None = None,
    *,
    util_threshold: float = DEFAULT_BUSY_UTIL_PCT,
    critical_free_fraction: float = DEFAULT_CRITICAL_FREE_FRACTION,
    critical_free_bytes: int = DEFAULT_CRITICAL_FREE_BYTES,
) -> tuple[bool, str]:
    """Decide whether FT batch calibrate should refuse to run.

    Primary signal: compute utilization.
    Memory fallback: only when torch free accounting is trusted AND free is
    critically low. Never abort solely because GB10 under-reports free bytes.
    """
    if not torch.cuda.is_available():
        return False, ""

    snap = snapshot or read_gpu_memory_snapshot()
    if snap.util_pct is not None and snap.util_pct >= util_threshold:
        return (
            True,
            f"GPU compute utilization is {snap.util_pct:.0f}% "
            f"(threshold {util_threshold:.0f}%). "
            "Stop the other job, then re-run FT batch size calibrate.",
        )

    if snap.free_accounting_reliable and (
        snap.free < critical_free_bytes
        or snap.free < critical_free_fraction * snap.usable
    ):
        return (
            True,
            f"GPU free memory is critically low ({snap.free / 2**30:.1f} GiB free of "
            f"{snap.total / 2**30:.1f} GiB). Free memory (stop other GPU jobs), then re-run.",
        )

    return False, ""


def _is_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _cache_path() -> Path | None:
    raw = os.environ.get("GENEFORMER_BATCH_SIZE_CACHE", "").strip()
    if raw.lower() in ("off", "0", "false", "no"):
        return None
    if raw:
        return Path(raw)
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "geneformer-platform" / "batch_size.json"


def _cache_key(fields: Mapping[str, Any]) -> str:
    payload = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _read_cache(path: Path | None, key: str) -> int | None:
    if path is None or not path.is_file():
        return None
    try:
        entry = json.loads(path.read_text(encoding="utf-8")).get(key)
    except (OSError, ValueError) as e:
        logger.warning("Ignoring unreadable batch-size cache %s: %s", path, e)
        return None
    if isinstance(entry, Mapping) and isinstance(entry.get("batch_size"), int):
        return int(entry["batch_size"])
    return None


def _write_cache(path: Path | None, key: str, batch_size: int, fields: Mapping[str, Any]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {}
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                data = {}
        data[key] = {"batch_size": batch_size, "fields": dict(fields)}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as e:
        logger.warning("Could not write batch-size cache %s: %s", path, e)


def _measure(
    probe: Callable[[int], None], candidate: int, repeats: int
) -> tuple[float, int] | None:
    """Return (samples per second, peak bytes) for one batch, or None on CUDA OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        probe(candidate)  # warm up kernels / allocator before timing
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            probe(candidate)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    except Exception as e:  # noqa: BLE001 - OOM is the expected stop condition
        if not _is_oom(e):
            raise
        torch.cuda.empty_cache()
        return None
    return (candidate * repeats) / max(elapsed, 1e-9), torch.cuda.max_memory_allocated()


def _largest_batch_that_fits(
    probe: Callable[[int], None],
    *,
    below: int,
    budget: int,
    repeats: int,
) -> int:
    """Halve below the smallest candidate until a batch fits; the caller needs one."""
    candidate = below // 2
    while candidate >= 1:
        measured = _measure(probe, candidate, repeats)
        if measured is not None and measured[1] <= budget:
            print(
                f"  batch {candidate}: fits ({measured[1] / 2**30:.1f} GiB), using it.",
                flush=True,
            )
            return candidate
        print(f"  batch {candidate}: still does not fit.", flush=True)
        candidate //= 2
    raise RuntimeError(
        "Batch-size calibration failed: even batch size 1 does not fit in the "
        f"{budget / 2**30:.1f} GiB budget. Free GPU memory (other jobs may be running), "
        "or lower the sequence length / model size."
    )


def calibrate_batch_size(
    probe: Callable[[int], None],
    *,
    candidates: Sequence[int] = DEFAULT_CANDIDATES,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
    min_gain: float = DEFAULT_MIN_GAIN,
    repeats: int = 2,
    label: str = "batch size",
    require_idle: bool = False,
) -> int:
    """Run `probe(batch)` over increasing batches; return the best usable one.

    When ``require_idle`` is True (FT batch calibrate UI/CLI), refuse to run if
    another job is actively using the GPU (compute util). Soft torch free
    under-reporting on GB10 does not abort calibration.
    """
    snap = read_gpu_memory_snapshot()
    budget = calibration_memory_budget(snap, memory_fraction=memory_fraction)
    chosen: int | None = None
    best_rate = 0.0

    accounting = "trusted" if snap.free_accounting_reliable else "unreliable(GB10-like)"
    print(
        f"Calibrating {label} (budget {budget / 2**30:.1f} GiB, "
        f"torch_free {snap.free / 2**30:.1f}/{snap.total / 2**30:.1f} GiB, "
        f"accounting={accounting}"
        + (f", util {snap.util_pct:.0f}%" if snap.util_pct is not None else "")
        + ")...",
        flush=True,
    )
    if require_idle:
        busy, reason = gpu_is_busy_for_calibration(snap)
        if busy:
            raise BusyGPUError(reason)

    if gpu_memory_contended(snap):
        print(
            "  Note: trusted free GPU memory looks limited "
            f"({snap.free / 2**30:.1f} GiB free of {snap.total / 2**30:.1f} GiB). "
            "If throughput plateaus early, re-run with the GPU fully idle.",
            flush=True,
        )

    cand_list = list(candidates)
    for idx, candidate in enumerate(cand_list):
        measured = _measure(probe, candidate, repeats)
        if measured is None:
            print(f"  batch {candidate}: out of memory, stopping.", flush=True)
            break

        rate, peak = measured
        print(
            f"  batch {candidate}: {rate:,.0f} samples/s, peak {peak / 2**30:.1f} GiB",
            flush=True,
        )

        if peak > budget:
            print(f"  batch {candidate}: over memory budget, stopping.", flush=True)
            break
        if best_rate and (rate - best_rate) / best_rate < min_gain:
            print(f"  batch {candidate}: no meaningful speedup, keeping {chosen}.", flush=True)
            break

        chosen, best_rate = int(candidate), rate
        # Extrapolate to the *next* candidate (not always 2x — train ladder is denser).
        if idx + 1 < len(cand_list) and candidate > 0:
            nxt = cand_list[idx + 1]
            extrapolated = peak * (nxt / candidate)
            if extrapolated > budget:
                print(
                    f"  next batch {nxt} would likely exceed budget "
                    f"(~{extrapolated / 2**30:.1f} GiB), keeping {chosen}.",
                    flush=True,
                )
                break

    if chosen is None:
        # The smallest candidate already failed; never return a batch known not to fit.
        chosen = _largest_batch_that_fits(
            probe, below=int(cand_list[0]), budget=budget, repeats=repeats
        )

    torch.cuda.empty_cache()
    print(f"Selected {label}: {chosen}", flush=True)
    return chosen


def resolve_batch_size(
    value: Any,
    *,
    default: int,
    probe: Callable[[int], None],
    cache_fields: Mapping[str, Any],
    candidates: Sequence[int] = DEFAULT_CANDIDATES,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
    min_gain: float = DEFAULT_MIN_GAIN,
    label: str = "batch size",
    require_idle: bool = False,
    skip_cache: bool = False,
) -> int:
    """Return `value` as an int, calibrating on this GPU when it is "auto"."""
    if not is_auto(value):
        return int(value)
    if not torch.cuda.is_available():
        logger.warning("auto %s requires CUDA; using %d.", label, default)
        return default

    fields = {
        **dict(cache_fields),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "memory_fraction": memory_fraction,
        "candidates": list(candidates),
    }
    path = None if skip_cache else _cache_path()
    key = _cache_key(fields)
    if not skip_cache:
        cached = _read_cache(path, key)
        if cached is not None:
            print(f"Using cached {label}: {cached}  (cache: {path})", flush=True)
            return cached

    contended_before = gpu_is_busy_for_calibration()[0]
    chosen = calibrate_batch_size(
        probe,
        candidates=candidates,
        memory_fraction=memory_fraction,
        min_gain=min_gain,
        label=label,
        require_idle=require_idle,
    )
    # Never persist a size measured while another job was actively computing.
    if contended_before or gpu_is_busy_for_calibration()[0]:
        print(
            f"Not caching {label}={chosen}: GPU was busy during calibration.",
            flush=True,
        )
    else:
        _write_cache(path, key, chosen, fields)
    return chosen
