"""Compute device selection: NVIDIA CUDA, Apple silicon MPS, or CPU.

``ISP_DEVICE=cuda|mps|cpu`` forces a device; otherwise CUDA, then MPS, then CPU.
On MPS, PyTorch ops without a Metal kernel need ``PYTORCH_ENABLE_MPS_FALLBACK=1``
(set by ``scripts/macos/env.sh``) to run on the CPU instead of raising.
"""
from __future__ import annotations

import os

import torch


def _mps_available() -> bool:
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    return bool(backend is not None and backend.is_available())


def device_type() -> str:
    forced = os.environ.get("ISP_DEVICE", "").strip().lower()
    if forced:
        if forced not in {"cuda", "mps", "cpu"}:
            raise ValueError("ISP_DEVICE must be cuda, mps or cpu")
        if forced == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("ISP_DEVICE=cuda but CUDA is not available")
        if forced == "mps" and not _mps_available():
            raise RuntimeError("ISP_DEVICE=mps but MPS is not available")
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


def get_device() -> torch.device:
    return torch.device(device_type())


def has_accelerator() -> bool:
    """A GPU (CUDA or MPS) holds the model, so host RAM is shared or scarce."""
    return device_type() in {"cuda", "mps"}


def device_name() -> str:
    kind = device_type()
    if kind == "cuda":
        return torch.cuda.get_device_name(0)
    if kind == "mps":
        return "Apple silicon GPU (MPS)"
    return "CPU"


def empty_cache() -> None:
    kind = device_type()
    if kind == "cuda":
        torch.cuda.empty_cache()
    elif kind == "mps":
        torch.mps.empty_cache()


def synchronize() -> None:
    kind = device_type()
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        torch.mps.synchronize()


def is_oom(exc: BaseException) -> bool:
    """CUDA ``OutOfMemoryError`` or the RuntimeError MPS raises when Metal memory runs out."""
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
