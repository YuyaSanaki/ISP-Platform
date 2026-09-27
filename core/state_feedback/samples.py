"""Unified feature collection for direction-fidelity evaluation.

Every reranking method must be scored on **exactly the same** (cell, gene) samples,
the same teacher, the same split and the same units, otherwise a difference in
Spearman could come from a difference in which genes were scored. This module
collects all scorer inputs for one shared sample set:

| Field | Consumed by |
|-------|-------------|
| ``delta_h`` | linear Δrank decoder (and MLP, if Phase 3 ever happens) |
| ``delta_norm`` | hidden-state norm baseline |
| ``delta_self_logit`` | ΔMLM baseline |
| ``base_rank`` | every method (rank inertia) |
| ``target`` | teacher: observed Δrank |

Gene selection is deterministic per cell, computed *before* any forward pass, so
the fine-tuned-model pass and the pretrained-MLM pass select identical genes.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import torch

from state_feedback import gene_states as gs


def select_gene_indices(
    input_ids: Sequence[int],
    scorable: Mapping[int, float] | set[int],
    *,
    max_genes: int,
    seed: int,
    keep_tokens: set[int] | None = None,
) -> list[int]:
    """Positions within one cell to evaluate, chosen deterministically.

    Seeded per cell so the selection is reproducible and identical across passes.
    """
    usable = [
        j
        for j, tok in enumerate(input_ids)
        if int(tok) in scorable and (keep_tokens is None or int(tok) in keep_tokens)
    ]
    if max_genes > 0 and len(usable) > int(max_genes):
        usable = random.Random(int(seed)).sample(usable, int(max_genes))
        usable.sort()
    return usable


@dataclass
class EvalSamples:
    """Flat ``(cell, gene)`` samples with every scorer's input attached."""

    cell_index: list[int] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    base_rank: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    target: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    delta_h: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 0))
    delta_norm: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    delta_self_logit: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.tokens)

    def token_subset(self, keep: set[int]) -> "EvalSamples":
        idx = [i for i, t in enumerate(self.tokens) if int(t) in keep]
        sel = torch.tensor(idx, dtype=torch.long)
        return EvalSamples(
            cell_index=[self.cell_index[i] for i in idx],
            tokens=[self.tokens[i] for i in idx],
            base_rank=self.base_rank.index_select(0, sel),
            target=self.target.index_select(0, sel),
            delta_h=self.delta_h.index_select(0, sel),
            delta_norm=self.delta_norm.index_select(0, sel),
            delta_self_logit=(
                None
                if self.delta_self_logit is None
                else self.delta_self_logit.index_select(0, sel)
            ),
        )


def collect_eval_samples(
    model,
    ctrl_ds,
    pert_ds,
    target_delta_rank: Mapping[int, float],
    *,
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    mlm_model=None,
    keep_tokens: set[int] | None = None,
    max_genes_per_cell: int = 256,
    rows: Sequence[int] | None = None,
    seed: int = 0,
) -> EvalSamples:
    """One shared sample set with hidden-state, norm and (optionally) ΔMLM features.

    ``mlm_model`` is optional; when omitted, ``delta_self_logit`` stays ``None`` and
    the ΔMLM method is simply unavailable rather than silently zero.
    """
    n = len(pert_ds)
    order = list(range(n)) if rows is None else [int(r) for r in rows]

    cell_index: list[int] = []
    tokens: list[int] = []
    chunks_base: list[torch.Tensor] = []
    chunks_target: list[torch.Tensor] = []
    chunks_dh: list[torch.Tensor] = []
    chunks_dnorm: list[torch.Tensor] = []
    selection: dict[int, list[int]] = {}

    from state_feedback.rerank import base_rank_norm

    for pair in gs.iter_gene_state_pairs(
        model,
        ctrl_ds,
        pert_ds,
        layer_to_quant=layer_to_quant,
        pad_token_id=pad_token_id,
        model_input_size=model_input_size,
        forward_batch_size=forward_batch_size,
        rows=order,
    ):
        for row, ids in enumerate(pair.pert_ids):
            cell = int(pair.cell_indices[row])
            picked = select_gene_indices(
                ids,
                target_delta_rank,
                max_genes=max_genes_per_cell,
                seed=seed * 1_000_003 + cell,
                keep_tokens=keep_tokens,
            )
            if not picked:
                continue
            selection[cell] = picked

            h_pert, h_ctrl, valid = pair.aligned_states(row)
            delta_h, _ = pair.delta_h(row)
            delta_norm = (h_pert.norm(dim=-1) - h_ctrl.norm(dim=-1)) * valid
            base = base_rank_norm(len(ids))
            sel = torch.tensor(picked, dtype=torch.long, device=delta_h.device)

            cell_index.extend([cell] * len(picked))
            tokens.extend(int(ids[j]) for j in picked)
            chunks_base.append(torch.tensor([base[j] for j in picked], dtype=torch.float32))
            chunks_target.append(
                torch.tensor(
                    [float(target_delta_rank[int(ids[j])]) for j in picked],
                    dtype=torch.float32,
                )
            )
            chunks_dh.append(delta_h.index_select(0, sel).detach().cpu())
            chunks_dnorm.append(delta_norm.index_select(0, sel).detach().cpu().float())

    if not chunks_dh:
        return EvalSamples(delta_h=torch.zeros((0, gs.hidden_size(model))))

    samples = EvalSamples(
        cell_index=cell_index,
        tokens=tokens,
        base_rank=torch.cat(chunks_base),
        target=torch.cat(chunks_target),
        delta_h=torch.cat(chunks_dh, dim=0),
        delta_norm=torch.cat(chunks_dnorm),
    )

    if mlm_model is not None:
        samples.delta_self_logit = _collect_self_logit_deltas(
            mlm_model,
            ctrl_ds,
            pert_ds,
            selection,
            order,
            pad_token_id=pad_token_id,
            model_input_size=model_input_size,
            forward_batch_size=forward_batch_size,
        )
    return samples


def _collect_self_logit_deltas(
    mlm_model,
    ctrl_ds,
    pert_ds,
    selection: Mapping[int, Sequence[int]],
    order: Sequence[int],
    *,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
) -> torch.Tensor:
    """ΔMLM self-logits for the already-chosen positions, in the same flat order."""
    out: list[torch.Tensor] = []
    batch = max(1, int(forward_batch_size))
    rows = [r for r in order if r in selection]
    for i in range(0, len(rows), batch):
        idx = rows[i : i + batch]
        ctrl_raw = [gs.raw_input_ids(ctrl_ds, r, r + 1, model_input_size)[0] for r in idx]
        pert_raw = [gs.raw_input_ids(pert_ds, r, r + 1, model_input_size)[0] for r in idx]
        ctrl_logits = gs.self_logits(mlm_model, ctrl_raw, pad_token_id)
        pert_logits = gs.self_logits(mlm_model, pert_raw, pad_token_id)
        for k, cell in enumerate(idx):
            ctrl_pos = {int(t): p for p, t in enumerate(ctrl_raw[k])}
            c_log, p_log = ctrl_logits[k], pert_logits[k]
            ids = pert_raw[k]
            vals: list[float] = []
            for j in selection[cell]:
                tok = int(ids[j]) if j < len(ids) else None
                p = ctrl_pos.get(tok) if tok is not None else None
                if p is None or p >= c_log.numel() or j >= p_log.numel():
                    vals.append(0.0)
                else:
                    vals.append(float(p_log[j]) - float(c_log[p]))
            out.append(torch.tensor(vals, dtype=torch.float32))
    return torch.cat(out) if out else torch.zeros(0)
