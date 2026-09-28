"""Deprecated alias of ``ordered_rank_edit`` (Sequential ISP was renamed Ordered rank-edit ISP).

Kept so external scripts that ``import sequential_oe`` keep working; new code
should import ``ordered_rank_edit``.
"""
from __future__ import annotations

import ordered_rank_edit as _impl

globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})

parse_sequential_steps = _impl.parse_steps
apply_sequential_overexpress = _impl.apply_ordered_overexpress
apply_sequential_perturb = _impl.apply_ordered_rank_edits
perturb_dataset_sequential = _impl.perturb_dataset_ordered_overexpress
