"""Potential-outcome contrast for State-feedback reranking (overexpression chains).

The plain decoder reads ``delta_h = h_pert - h_ctrl`` of every gene. Most of that
change appears for any inserted gene (the encoding shifts down by one slot, the
context changes), so a decoder trained on it moves the encoding toward the goal for
random chains almost as much as for the configured one. Here the decoder reads

    delta_h - m_g,   m_g = mean_p [h(swap_p(encoding)) - h_ctrl]

where ``swap_p`` replaces the chain's genes in the current encoding by the genes of
estimation placebo ``p`` (one gene per slot, drawn from the same detection / position
stratum by ``random_chains``) and drops the placebo genes from their old positions.
``m_g`` is what the gene's state would have been under the same perturbation with
other genes: the generic part. The decoder is trained on the same contrast, computed
on the at-once perturbed training encodings.

Estimation placebos are drawn with their own seed, separate from the random chains
the configured chain is compared with (cross-fitting). A placebo that shares a gene
with a chain is left out of that chain's contrast.

Only overexpression steps are defined; deletion needs its own counterfactual.
"""
from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

import torch

from ordered_rank_edit import PERTURB_OVEREXPRESS, normalize_step_type
from state_feedback import gene_states as gs
from state_feedback import random_chains
from state_feedback.decoder import TrainingSet
from state_feedback.feedback import rerank_diagnostics
from state_feedback.rerank import apply_priority_order, base_rank_norm

RerankResult = tuple[list[list[int]], dict[str, Any]]

# Estimation placebos are the first ``n`` of a draw of at least this many chains, so
# placebo sets of different size (n_est sensitivity) are nested.
ESTIMATION_POOL = 20


def check_overexpress_only(steps: Sequence[Mapping[str, Any]]) -> None:
    other = [s.get("name") for s in steps
             if normalize_step_type(str(s["type"])) != PERTURB_OVEREXPRESS]
    if other:
        raise ValueError(
            "the placebo contrast is defined for overexpression steps only; "
            f"non-overexpression steps: {other}"
        )


def draw_estimation_placebos(
    steps: Sequence[Mapping[str, Any]],
    token_by_step: Sequence[Sequence[int]],
    population: Sequence[int],
    detection: Mapping[int, float],
    rank: Mapping[int, float],
    *,
    n: int,
    seed: int,
    min_stratum: int = 50,
) -> tuple[list[list[list[int]]], dict[str, Any]]:
    """``n`` estimation placebos (token lists per step, shaped like ``token_by_step``)."""
    if int(n) < 1:
        raise ValueError("need at least one estimation placebo")
    drawn, record = random_chains.draw_random_chains(
        steps, token_by_step, population, detection, rank,
        n_chains=max(int(n), ESTIMATION_POOL), seed=int(seed),
        min_stratum=min_stratum, prefix="est",
    )
    return [drawn[f"est{k}"][1] for k in range(int(n))], record


def slot_substitutions(
    chain_token_by_step: Sequence[Sequence[int]],
    placebos: Sequence[Sequence[Sequence[int]]],
) -> list[dict[int, int]]:
    """One ``{chain gene: placebo gene}`` map per usable estimation placebo.

    Slots are (step, position within step). Placebos sharing a gene with the chain
    are left out.
    """
    chain = [int(t) for ts in chain_token_by_step for t in ts]
    out: list[dict[int, int]] = []
    for p in placebos:
        flat = [int(t) for ts in p for t in ts]
        if len(flat) != len(chain):
            raise ValueError(f"placebo has {len(flat)} slots, chain has {len(chain)}")
        if set(flat) & set(chain):
            continue
        out.append(dict(zip(chain, flat)))
    if not out:
        raise ValueError("every estimation placebo shares a gene with the chain")
    return out


def swap_tokens(
    ids: Sequence[int],
    sub: Mapping[int, int],
    max_len: int | None = None,
) -> tuple[list[int], set[int]]:
    """Encoding with the chain genes present replaced in place by their placebo genes.

    Placebo genes are dropped from their old positions. Returns the new encoding and
    the genes that have no counterfactual state (swapped chain genes and their
    placebos).
    """
    present = {int(t) for t in ids}
    active = {int(c): int(p) for c, p in sub.items() if int(c) in present}
    repl = set(active.values())
    new = [active.get(int(t), int(t)) for t in ids if int(t) in active or int(t) not in repl]
    if max_len is not None:
        new = new[: int(max_len)]
    return new, set(active) | repl


def placebo_mediator(
    model,
    pair: gs.GeneStatePair,
    subs: Sequence[Mapping[int, int]],
    *,
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """``(m_g [L_pert, d], n_placebos [L_pert])`` per cell of ``pair``.

    ``m_g`` is zero where no placebo gives a state (chain genes, genes missing from
    the control). Cells with no chain gene in the encoding are not re-encoded:
    their counterfactual is the perturbed state itself, so the contrast is exactly 0.
    """
    dev = pair.h_pert.device
    d = pair.h_pert.size(-1)
    ctrl = [gs.align_ctrl(ids, pair.ctrl_ids[r], pair.h_ctrl[r])
            for r, ids in enumerate(pair.pert_ids)]
    sums = [torch.zeros(len(ids), d, device=dev) for ids in pair.pert_ids]
    counts = [torch.zeros(len(ids), device=dev) for ids in pair.pert_ids]
    for sub in subs:
        swapped, excluded = [], []
        for ids in pair.pert_ids:
            new, excl = swap_tokens(ids, sub, model_input_size)
            swapped.append(new)
            excluded.append(excl)
        rerun = [r for r, ex in enumerate(excluded) if ex]
        h = (gs._hidden_states(model, [swapped[r] for r in rerun], pad_token_id, layer_to_quant)
             if rerun else None)
        h_row = {r: i for i, r in enumerate(rerun)}
        for r, ids in enumerate(pair.pert_ids):
            h_ctrl, valid = ctrl[r]
            if r in h_row:
                pos = {t: j for j, t in enumerate(swapped[r])}
                g = torch.tensor(
                    [-1 if int(t) in excluded[r] else pos.get(int(t), -1) for t in ids],
                    dtype=torch.long, device=dev,
                )
                ok = (g >= 0) & valid
                hp = h[h_row[r]].float().index_select(0, g.clamp(min=0))
            else:
                ok = valid.clone()
                hp = pair.h_pert[r][: len(ids)].float()
            okf = ok.float()
            sums[r] += (hp - h_ctrl) * okf.unsqueeze(-1)
            counts[r] += okf
        del h
    return [s / c.clamp(min=1).unsqueeze(-1) for s, c in zip(sums, counts)], counts


def contrast_delta_h(
    pair: gs.GeneStatePair,
    row: int,
    m_g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(delta_h - m_g, valid)`` for one cell, indexed by perturbed position."""
    delta_h, valid = pair.delta_h(row)
    return (delta_h - m_g) * valid.unsqueeze(-1), valid


def rerank_placebo_contrast(
    model,
    decoder,
    ctrl_ds,
    pert_ds,
    subs: Sequence[Mapping[int, int]],
    *,
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    hysteresis: float = 0.0,
) -> RerankResult:
    """Rerank with the decoder applied to the placebo contrast ``delta_h - m_g``."""
    dev = next(decoder.parameters()).device
    before: list[list[int]] = []
    after: list[list[int]] = []
    n_placebos: list[float] = []
    for pair in gs.iter_gene_state_pairs(
        model, ctrl_ds, pert_ds,
        layer_to_quant=layer_to_quant,
        pad_token_id=pad_token_id,
        model_input_size=model_input_size,
        forward_batch_size=forward_batch_size,
    ):
        m_g, counts = placebo_mediator(
            model, pair, subs,
            layer_to_quant=layer_to_quant,
            pad_token_id=pad_token_id,
            model_input_size=model_input_size,
        )
        for row, ids in enumerate(pair.pert_ids):
            feat, valid = contrast_delta_h(pair, row, m_g[row])
            base = torch.tensor(base_rank_norm(len(ids)), dtype=torch.float32, device=dev)
            with torch.no_grad():
                priority = decoder(feat.to(dev), base).detach().cpu().tolist()
            before.append(list(ids))
            after.append(apply_priority_order(ids, priority, hysteresis=hysteresis))
            if bool(valid.any()):
                n_placebos.append(float(counts[row][valid].mean()))
    diag = rerank_diagnostics(before, after)
    diag["contrast_placebos_mean"] = (
        float(sum(n_placebos) / len(n_placebos)) if n_placebos else 0.0
    )
    return after, diag


def collect_contrast_training_samples(
    model,
    ctrl_ds,
    pert_ds,
    target_delta_rank: Mapping[int, float],
    subs: Sequence[Mapping[int, int]],
    *,
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    max_genes_per_cell: int = 256,
    max_cells: int | None = None,
    seed: int = 0,
) -> TrainingSet:
    """``(delta_h - m_g, base_rank, observed delta rank)`` samples for training.

    Same sampling as ``feedback.collect_training_samples``, restricted to genes that
    are in the control encoding, are not chain genes, and have at least one placebo
    state.
    """
    rng = random.Random(int(seed))
    rows = list(range(len(pert_ds)))
    if max_cells is not None and len(rows) > int(max_cells):
        rows = rows[: int(max_cells)]
    chain = {int(c) for sub in subs for c in sub}

    chunks_dh: list[torch.Tensor] = []
    chunks_base: list[torch.Tensor] = []
    chunks_target: list[torch.Tensor] = []
    tokens: list[int] = []
    for pair in gs.iter_gene_state_pairs(
        model, ctrl_ds, pert_ds,
        layer_to_quant=layer_to_quant,
        pad_token_id=pad_token_id,
        model_input_size=model_input_size,
        forward_batch_size=forward_batch_size,
        rows=rows,
    ):
        m_g, counts = placebo_mediator(
            model, pair, subs,
            layer_to_quant=layer_to_quant,
            pad_token_id=pad_token_id,
            model_input_size=model_input_size,
        )
        for row, ids in enumerate(pair.pert_ids):
            feat, valid = contrast_delta_h(pair, row, m_g[row])
            ok = (valid & (counts[row] > 0)).tolist()
            usable = [
                j for j, tok in enumerate(ids)
                if ok[j] and int(tok) in target_delta_rank and int(tok) not in chain
            ]
            if not usable:
                continue
            if len(usable) > int(max_genes_per_cell):
                usable = rng.sample(usable, int(max_genes_per_cell))
            base = base_rank_norm(len(ids))
            sel = torch.tensor(usable, dtype=torch.long, device=feat.device)
            chunks_dh.append(feat.index_select(0, sel).detach().cpu())
            chunks_base.append(torch.tensor([base[j] for j in usable], dtype=torch.float32))
            chunks_target.append(torch.tensor(
                [float(target_delta_rank[int(ids[j])]) for j in usable], dtype=torch.float32
            ))
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
