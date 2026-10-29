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

    Platform default (v1.1.0) is ``warmup_steps: 500`` with ``warmup_ratio``
    unset — use ratio 0.05 when the fine-tune goal is classification.
    """
    total_steps = max(1, steps_per_epoch * num_epochs)
    if training_cfg.get("warmup_ratio") is not None:
        warmup = int(total_steps * float(training_cfg["warmup_ratio"]))
    else:
        warmup = int(training_cfg.get("warmup_steps", 500))
    cap = max(1, total_steps // 10)
    warmup = min(warmup, cap, max(0, total_steps - 1))
    return max(0, warmup)


# 6 cells at 2,048 tokens is the batch that fits on an 80 GiB GPU.
# Longer ranks cost more with the square of length. Usable memory scales that budget.
_FT_REFERENCE_BYTES = 80 * 1024**3
_FT_BASE_CELLS = 6
_FT_BASE_SEQ = 2048


def ft_pass_fits(cells: int, seq_len: int, usable_bytes: int) -> bool:
    """True when one GPU pass of ``cells`` at ``seq_len`` fits in ``usable_bytes``."""
    if int(usable_bytes) <= 0 or int(cells) < 1 or int(seq_len) < 1:
        return False
    cost = int(cells) * int(seq_len) * int(seq_len)
    budget = _FT_BASE_CELLS * _FT_BASE_SEQ * _FT_BASE_SEQ * (int(usable_bytes) / _FT_REFERENCE_BYTES)
    return cost <= budget


def choose_ft_batch_from_lengths(
    lengths: list[int] | tuple[int, ...],
    usable_bytes: int,
    *,
    effective_batch: int = 6,
    model_cap: int = 4096,
) -> tuple[str, int, int, int]:
    """Pick standard or micro-batch from measured rank lengths and free memory.

    Returns ``(mode, cells_on_gpu, accumulation_steps, seq_len)``.
    ``seq_len`` is the rank length of the batch that was checked (the
    ``cells``-th longest cell, capped at ``model_cap``). Fine-tune groups
    similar lengths, so that cell is the long batch.

    Effective batch stays ``effective_batch``. Cells per pass is the largest
    divisor of that effective size whose long batch still fits. When the full
    effective size fits in one pass, mode is ``standard``.
    """
    cap = max(1, int(model_cap))
    measured = [min(max(int(value), 0), cap) for value in lengths]
    measured = [value for value in measured if value > 0]
    if int(usable_bytes) <= 0:
        mode, cells, passes = choose_ft_batch(cap if not measured else max(measured), effective_batch)
        seq = max(measured) if measured else cap
        return mode, cells, passes, seq
    descending = sorted(measured, reverse=True) if measured else [cap]
    effective = max(1, int(effective_batch))
    divisors = [divisor for divisor in range(effective, 0, -1) if effective % divisor == 0]
    for cells in divisors:
        seq = descending[min(cells, len(descending)) - 1]
        if ft_pass_fits(cells, seq, usable_bytes):
            if cells >= effective:
                return "standard", effective, 1, seq
            return "micro", cells, effective // cells, seq
    return "micro", 1, effective, descending[0]


def auto_ft_batch_decision(
    lengths: list[int] | tuple[int, ...],
    usable_bytes: int,
    *,
    effective_batch: int = 6,
    model_cap: int = 4096,
    memory_source: str = "",
) -> dict[str, Any]:
    """Decide one fine-tune run from that run's rank lengths and free memory."""
    mode, cells, passes, seq_len = choose_ft_batch_from_lengths(
        lengths,
        usable_bytes,
        effective_batch=effective_batch,
        model_cap=model_cap,
    )
    cap = max(1, int(model_cap))
    capped = [min(max(int(value), 0), cap) for value in lengths]
    capped = [value for value in capped if value > 0]
    return {
        "mode": mode,
        "cells": int(cells),
        "passes": int(passes),
        "seq_len": int(seq_len),
        "effective": max(1, int(effective_batch)),
        "n": len(capped),
        "maximum": max(capped) if capped else int(seq_len),
        "memory_bytes": max(0, int(usable_bytes)),
        "memory_gib": round(max(0, int(usable_bytes)) / (1024**3), 1),
        "memory_source": str(memory_source or ""),
    }


def apply_auto_ft_batch_to_configs(
    finetune_cfg: dict[str, Any],
    pipeline_cfg: dict[str, Any] | None,
    decision: Mapping[str, Any],
) -> None:
    """Write a run-time Auto decision onto the fine-tune config and, if given, the pipeline record."""
    training = finetune_cfg.setdefault("training", {})
    effective = int(decision["effective"])
    if str(decision["mode"]) == "standard":
        training["batch_size"] = effective
        training["gradient_accumulation_steps"] = 1
    else:
        training["batch_size"] = int(decision["cells"])
        training["gradient_accumulation_steps"] = int(decision["passes"])
    training["batch_size_policy"] = "auto"
    training["effective_batch_size"] = effective
    if pipeline_cfg is None:
        return
    runtime = pipeline_cfg.setdefault("runtime", {})
    runtime["ft_batch_choice"] = "auto"
    runtime["ft_batch_decided"] = True
    runtime["ft_batch_mode"] = decision["mode"]
    runtime["micro_effective_batch_size"] = effective
    runtime["micro_gradient_accumulation_steps"] = int(decision["passes"])
    runtime["ft_input_length"] = int(decision["seq_len"])
    runtime["ft_n_cells"] = int(decision["n"])
    runtime["ft_length_max"] = int(decision["maximum"])
    runtime["ft_memory_gib"] = decision["memory_gib"]
    if str(decision["mode"]) == "micro":
        runtime["micro_batch_size"] = int(decision["cells"])
    else:
        runtime.pop("micro_batch_size", None)


def choose_ft_batch(seq_len: int, effective_batch: int = 6) -> tuple[str, int, int]:
    """Choose standard or micro-batch so one update still uses ``effective_batch`` cells.

    Returns ``(mode, cells_on_gpu, accumulation_steps)``. ``mode`` is
    ``standard`` when one GPU pass of ``effective_batch`` cells fits the
    sequence length (the historical batch, accumulation 1). Otherwise
    ``micro``: the largest GPU pass that divides ``effective_batch`` and stays
    within the safe per-pass size for that length.

    A length above 2048 is treated as too large for a pass of 6 cells. The
    safe pass size is then 2, so effective batch 6 becomes 2 cells × 3 passes.
    """
    from geneformer.species_context import default_finetune_batch_size

    fits = int(default_finetune_batch_size(int(seq_len)))
    effective = max(1, int(effective_batch))
    if effective <= fits:
        return "standard", effective, 1
    cells = 1
    for divisor in range(1, fits + 1):
        if effective % divisor == 0:
            cells = divisor
    return "micro", cells, effective // cells


def optimizer_updates_per_epoch(
    n_train: int, micro_batch: int, accumulation_steps: int = 1
) -> int:
    """Optimizer updates in one epoch.

    ``accumulation_steps`` 1 matches the historical count ``n_train // micro_batch``
    (floor division, at least 1). Larger values group that many GPU passes into
    one weight update.
    """
    micro = max(1, int(micro_batch))
    accum = max(1, int(accumulation_steps))
    micro_steps = max(1, int(n_train) // micro)
    return max(1, micro_steps // accum)


def resolve_dataloader_num_workers(runtime_cfg: Mapping[str, Any]) -> int:
    """DataLoader workers default to 0 (stable in Docker); dataset map uses ``nproc``."""
    if runtime_cfg.get("dataloader_num_workers") is not None:
        return int(runtime_cfg["dataloader_num_workers"])
    return 0
