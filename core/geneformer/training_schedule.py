"""Training schedule helpers for fine-tuning (warmup, DataLoader workers)."""
from __future__ import annotations

from typing import Any, Mapping


def resolve_warmup_steps(
    training_cfg: Mapping[str, Any],
    *,
    steps_per_epoch: int,
    num_epochs: int,
) -> int:
    """
    Resolve LR warmup steps without dominating short smoke runs.

    ``warmup_ratio`` (if set) overrides ``warmup_steps``. The result is capped
    at 10% of total optimizer steps and at ``total_steps - 1``.
    """
    total_steps = max(1, steps_per_epoch * num_epochs)
    if training_cfg.get("warmup_ratio") is not None:
        warmup = int(total_steps * float(training_cfg["warmup_ratio"]))
    else:
        warmup = int(training_cfg.get("warmup_steps", 100))
    cap = max(1, total_steps // 10)
    warmup = min(warmup, cap, max(0, total_steps - 1))
    return max(0, warmup)


def resolve_dataloader_num_workers(runtime_cfg: Mapping[str, Any]) -> int:
    """DataLoader workers default to 0 (stable in Docker); dataset map uses ``nproc``."""
    if runtime_cfg.get("dataloader_num_workers") is not None:
        return int(runtime_cfg["dataloader_num_workers"])
    return 0
