"""Rerank strategies that turn one perturbed encoding into the next one.

Every strategy has the same contract:

    (model, ctrl_ds, pert_ds) -> new token lists (one per cell) + diagnostics

The gene set of each cell is preserved; only the within-cell order changes.
Strategies differ only in how they score genes:

| Strategy | Score source |
|----------|--------------|
| ``norm`` | change in hidden-state norm (null baseline) |
| ``delta_mlm`` | change in pretrained MLM self-logit plus inertia |
| ``linear_deltarank`` | trained residual Delta-rank decoder (primary) |
| ``oracle`` | observed endpoint ranks (ceiling, not a decoder) |
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import torch

from state_feedback import gene_states as gs
from state_feedback.decoder import DeltaRankDecoder, TrainingSet
from state_feedback.metrics import displacement_summary, topk_jaccard
from state_feedback.oracle_rerank import rerank_fixed_gene_set, spearman_rank_correlation
from state_feedback.rerank import (
    apply_priority_order,
    base_rank_norm,
    delta_mlm_priority,
    norm_priority,
)

RerankResult = tuple[list[list[int]], dict[str, float]]


def replace_input_ids(dataset, new_ids: Sequence[Sequence[int]], *, num_proc: int = 1):
    """Dataset with ``input_ids`` / ``length`` / ``attention_mask`` replaced per row."""
    if len(new_ids) != len(dataset):
        raise ValueError(f"new_ids ({len(new_ids)}) must match dataset rows ({len(dataset)})")
    rows = [list(map(int, r)) for r in new_ids]

    def _set(example, idx):
        ids = rows[int(idx)]
        return {
            "input_ids": ids,
            "length": len(ids),
            "attention_mask": [1] * len(ids),
        }

    try:
        dataset.reset_format()
    except Exception:
        pass
    return dataset.map(_set, with_indices=True, num_proc=max(1, int(num_proc)))


def _diagnostics(before: Sequence[Sequence[int]], after: Sequence[Sequence[int]]) -> dict[str, float]:
    """Mean per-cell displacement / overlap / self-correlation of a rerank."""
    rho: list[float] = []
    mean_disp: list[float] = []
    frac_moved: list[float] = []
    jac: list[float] = []
    for b, a in zip(before, after):
        rho.append(spearman_rank_correlation(list(b), list(a)))
        summary = displacement_summary(b, a)
        if summary["mean_abs_displacement"] == summary["mean_abs_displacement"]:
            mean_disp.append(summary["mean_abs_displacement"])
            frac_moved.append(summary["frac_moved"])
        j = topk_jaccard(b, a, 100)
        if j == j:
            jac.append(j)

    def _avg(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    return {
        "rerank_spearman_before_after": _avg(rho),
        "rerank_mean_abs_displacement": _avg(mean_disp),
        "rerank_frac_moved": _avg(frac_moved),
        "rerank_top100_jaccard": _avg(jac),
    }


def rerank_oracle(
    pert_ds,
    oracle_priority: Mapping[int, float],
    *,
    hysteresis: float = 0.0,
    model_input_size: int = 4096,
) -> RerankResult:
    """Ceiling: order each cell's gene set by observed endpoint ranks."""
    before = gs.raw_input_ids(pert_ds, 0, len(pert_ds), model_input_size)
    after = [rerank_fixed_gene_set(ids, oracle_priority) for ids in before]
    if hysteresis > 0.0:
        after = [
            apply_priority_order(
                ids,
                [float(oracle_priority.get(int(t), len(ids) + 1)) for t in ids],
                hysteresis=hysteresis,
            )
            for ids in before
        ]
    return after, _diagnostics(before, after)


def rerank_norm(
    model,
    ctrl_ds,
    pert_ds,
    *,
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    alpha: float,
    max_shift: float,
    hysteresis: float = 0.0,
) -> RerankResult:
    """Null baseline: promote genes whose hidden-state norm grew most."""
    before: list[list[int]] = []
    after: list[list[int]] = []
    for pair in gs.iter_gene_state_pairs(
        model,
        ctrl_ds,
        pert_ds,
        layer_to_quant=layer_to_quant,
        pad_token_id=pad_token_id,
        model_input_size=model_input_size,
        forward_batch_size=forward_batch_size,
    ):
        for row, ids in enumerate(pair.pert_ids):
            h_pert, h_ctrl, valid = pair.aligned_states(row)
            delta_norm = (h_pert.norm(dim=-1) - h_ctrl.norm(dim=-1)) * valid
            priority = norm_priority(
                delta_norm.cpu().tolist(), alpha=alpha, max_shift=max_shift
            )
            before.append(list(ids))
            after.append(apply_priority_order(ids, priority, hysteresis=hysteresis))
    return after, _diagnostics(before, after)


def rerank_delta_mlm(
    mlm_model,
    ctrl_ds,
    pert_ds,
    *,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    alpha: float,
    max_shift: float,
    hysteresis: float = 0.0,
) -> RerankResult:
    """Parameter-free comparator: change in pretrained MLM self-logit."""
    before: list[list[int]] = []
    after: list[list[int]] = []
    for _idx, ctrl_ids, pert_ids, ctrl_logits, pert_logits in gs.iter_self_logit_pairs(
        mlm_model,
        ctrl_ds,
        pert_ds,
        pad_token_id=pad_token_id,
        model_input_size=model_input_size,
        forward_batch_size=forward_batch_size,
    ):
        for row, ids in enumerate(pert_ids):
            ctrl_pos = {int(t): i for i, t in enumerate(ctrl_ids[row])}
            c_log = ctrl_logits[row]
            p_log = pert_logits[row]
            deltas: list[float] = []
            for j, tok in enumerate(ids):
                i = ctrl_pos.get(int(tok))
                if i is None or i >= c_log.numel() or j >= p_log.numel():
                    deltas.append(0.0)
                else:
                    deltas.append(float(p_log[j]) - float(c_log[i]))
            priority = delta_mlm_priority(deltas, alpha=alpha, max_shift=max_shift)
            before.append(list(ids))
            after.append(apply_priority_order(ids, priority, hysteresis=hysteresis))
    return after, _diagnostics(before, after)


def rerank_linear_deltarank(
    model,
    decoder: DeltaRankDecoder,
    ctrl_ds,
    pert_ds,
    *,
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    hysteresis: float = 0.0,
) -> RerankResult:
    """Primary method: trained residual Delta-rank decoder on ``h_pert - h_ctrl``."""
    dev = next(decoder.parameters()).device
    before: list[list[int]] = []
    after: list[list[int]] = []
    for pair in gs.iter_gene_state_pairs(
        model,
        ctrl_ds,
        pert_ds,
        layer_to_quant=layer_to_quant,
        pad_token_id=pad_token_id,
        model_input_size=model_input_size,
        forward_batch_size=forward_batch_size,
    ):
        for row, ids in enumerate(pair.pert_ids):
            delta_h, _valid = pair.delta_h(row)
            base = torch.tensor(base_rank_norm(len(ids)), dtype=torch.float32, device=dev)
            with torch.no_grad():
                priority = decoder(delta_h.to(dev), base).detach().cpu().tolist()
            before.append(list(ids))
            after.append(apply_priority_order(ids, priority, hysteresis=hysteresis))
    return after, _diagnostics(before, after)


def collect_training_samples(
    model,
    ctrl_ds,
    pert_ds,
    target_delta_rank: Mapping[int, float],
    *,
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    keep_tokens: set[int] | None = None,
    max_genes_per_cell: int = 256,
    max_cells: int | None = None,
    seed: int = 0,
) -> TrainingSet:
    """Build ``(delta_h, base_rank, observed delta rank)`` samples for training.

    Genes are subsampled per cell so the flattened tensor stays small; only genes
    with a teacher target (and in ``keep_tokens`` when given) are used.
    """
    import random

    rng = random.Random(int(seed))
    rows = list(range(len(pert_ds)))
    if max_cells is not None and len(rows) > int(max_cells):
        rows = rows[: int(max_cells)]

    chunks_dh: list[torch.Tensor] = []
    chunks_base: list[torch.Tensor] = []
    chunks_target: list[torch.Tensor] = []
    tokens: list[int] = []

    for pair in gs.iter_gene_state_pairs(
        model,
        ctrl_ds,
        pert_ds,
        layer_to_quant=layer_to_quant,
        pad_token_id=pad_token_id,
        model_input_size=model_input_size,
        forward_batch_size=forward_batch_size,
        rows=rows,
    ):
        for row, ids in enumerate(pair.pert_ids):
            usable = [
                j
                for j, tok in enumerate(ids)
                if int(tok) in target_delta_rank
                and (keep_tokens is None or int(tok) in keep_tokens)
            ]
            if not usable:
                continue
            if len(usable) > int(max_genes_per_cell):
                usable = rng.sample(usable, int(max_genes_per_cell))
            delta_h, _valid = pair.delta_h(row)
            base = base_rank_norm(len(ids))
            sel = torch.tensor(usable, dtype=torch.long, device=delta_h.device)
            chunks_dh.append(delta_h.index_select(0, sel).detach().cpu())
            chunks_base.append(torch.tensor([base[j] for j in usable], dtype=torch.float32))
            chunks_target.append(
                torch.tensor(
                    [float(target_delta_rank[int(ids[j])]) for j in usable],
                    dtype=torch.float32,
                )
            )
            tokens.extend(int(ids[j]) for j in usable)

    if not chunks_dh:
        d = gs.hidden_size(model)
        return TrainingSet(
            delta_h=torch.zeros((0, d), dtype=torch.float32),
            base_rank=torch.zeros((0,), dtype=torch.float32),
            target=torch.zeros((0,), dtype=torch.float32),
            tokens=[],
        )
    return TrainingSet(
        delta_h=torch.cat(chunks_dh, dim=0),
        base_rank=torch.cat(chunks_base, dim=0),
        target=torch.cat(chunks_target, dim=0),
        tokens=tokens,
    )
