#!/usr/bin/env python3
"""Deprecated alias of ``run_ordered_rank_edit_isp.py`` (Sequential ISP was renamed).

Kept so existing commands and ``import run_sequential_isp`` keep working; the
CLI flags and config (including the legacy ``sequential:`` block) are unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_ordered_rank_edit_isp as _impl

globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})

resolve_sequential_forward_batch_size = _impl.resolve_dual_forward_batch_size


if __name__ == "__main__":
    print(
        "run_sequential_isp.py is deprecated; use core/run_ordered_rank_edit_isp.py",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(_impl.main())
