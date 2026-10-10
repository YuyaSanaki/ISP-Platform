"""Observed rank-displacement teacher signal for the Delta-rank decoder.

The decoder is trained against *observed* rank displacement between a control
and a post-perturbation state:

    delta_r_obs[gene] = rank_norm_obs[gene] - rank_norm_ctrl[gene]

Positions are pseudobulk (mean position across cells of a state), not raw
single-cell ranks, so dropout noise does not dominate. Each state is normalized
by its own mean encoding length, which makes the two states comparable in the
same ``[0, 1]`` rank-normalized units the decoder operates in.

On the OSKM reprogramming surrogate, ``ctrl`` = somatic cells and ``obs`` =
pluripotent cells. This is a coarse endpoint teacher, not a paired
perturbation measurement; see ``docs/state_feedback_decode_methods.md`` for the
priority order of better data sources.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence


def pseudobulk_positions(
    input_ids_rows: Iterable[Sequence[int]],
) -> tuple[dict[int, float], dict[int, int], float]:
    """``(mean_position, detection_count, mean_encoding_length)`` per token."""
    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    total_len = 0
    n_cells = 0
    for ids in input_ids_rows:
        n_cells += 1
        total_len += len(ids)
        for pos, tok in enumerate(ids):
            t = int(tok)
            sums[t] += float(pos)
            counts[t] += 1
    mean_pos = {t: sums[t] / counts[t] for t in sums}
    mean_len = (total_len / n_cells) if n_cells else 0.0
    return mean_pos, dict(counts), mean_len


def observed_delta_rank(
    ctrl_rows: Iterable[Sequence[int]],
    obs_rows: Iterable[Sequence[int]],
    *,
    min_detection_count: int = 5,
) -> dict[int, float]:
    """Normalized observed rank displacement per token, control -> observed.

    Tokens detected in fewer than ``min_detection_count`` cells of either state
    are dropped: their mean position is too noisy to serve as a target.
    """
    pos_ctrl, cnt_ctrl, len_ctrl = pseudobulk_positions(ctrl_rows)
    pos_obs, cnt_obs, len_obs = pseudobulk_positions(obs_rows)
    denom_ctrl = max(1.0, len_ctrl - 1.0)
    denom_obs = max(1.0, len_obs - 1.0)
    out: dict[int, float] = {}
    for tok, p_ctrl in pos_ctrl.items():
        p_obs = pos_obs.get(tok)
        if p_obs is None:
            continue
        if cnt_ctrl.get(tok, 0) < min_detection_count:
            continue
        if cnt_obs.get(tok, 0) < min_detection_count:
            continue
        out[tok] = float(p_obs / denom_obs - p_ctrl / denom_ctrl)
    return out


def split_tokens(
    tokens: Iterable[int],
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[set[int], set[int]]:
    """Deterministic token-level train/val split.

    Held-out **genes** (not just cells) are what make the reported
    Spearman(pred, obs) an out-of-sample number: the teacher is a per-token
    quantity, so a cell-only split would leak every target.
    """
    train: set[int] = set()
    val: set[int] = set()
    frac = min(max(float(val_fraction), 0.0), 1.0)
    modulus = 1_000_003
    for tok in tokens:
        h = (int(tok) * 2_654_435_761 + int(seed) * 97_531) % modulus
        if h / modulus < frac:
            val.add(int(tok))
        else:
            train.add(int(tok))
    return train, val


def priority_from_delta_rank(
    input_ids: Sequence[int],
    delta_rank: Mapping[int, float],
    *,
    max_shift: float,
) -> list[float]:
    """Teacher-derived priority for one cell (used by diagnostics, not training)."""
    from state_feedback.rerank import base_rank_norm, bounded_priority

    base = base_rank_norm(len(input_ids))
    shift = [float(delta_rank.get(int(t), 0.0)) for t in input_ids]
    return bounded_priority(base, shift, max_shift=max_shift)
