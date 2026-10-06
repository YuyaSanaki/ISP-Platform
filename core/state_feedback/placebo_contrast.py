"""Potential-outcome contrast for State-feedback reranking (overexpression and delete steps).

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

Delete steps (docs/state_feedback_deletion_counterfactual.md): the counterfactual undoes
the deletion of the factor and deletes the placebo gene instead (``undo_delete_redo``).
The factor goes back right after the nearest gene that preceded it in the start
encoding and is still present, so before any rerank the result is exactly the start
encoding with the placebo gene deleted. Slots whose factor was not in the cell's start
encoding get no placebo edit. The start encoding is the control (``ctrl_reference:
start``).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Union

import torch

from rank_edit import PERTURB_DELETE, normalize_step_type
from state_feedback import gene_states as gs
from state_feedback import random_chains
from state_feedback.decoder import TrainingSet
from state_feedback.feedback import rerank_diagnostics
from state_feedback.rerank import apply_priority_order, base_rank_norm

RerankResult = tuple[list[list[int]], dict[str, Any]]

# Estimation placebos are the first ``n`` of a draw of at least this many chains, so
# placebo sets of different size (n_est sensitivity) are nested.
ESTIMATION_POOL = 20


@dataclass(frozen=True)
class MixedSubstitution:
    """Chain-to-placebo genes of a chain with delete steps: ``oe`` slots are swapped as
    in ``swap_tokens``; ``ko`` slots restore the deleted factor and delete the placebo."""

    oe: dict[int, int]
    ko: dict[int, int]


# Overexpression-only chains use a plain ``{chain gene: placebo gene}`` map.
Substitution = Union[Mapping[int, int], MixedSubstitution]


def delete_step_indices(steps: Sequence[Mapping[str, Any]]) -> frozenset[int]:
    """0-based indices of the delete steps (raises on unknown step types)."""
    return frozenset(
        i for i, s in enumerate(steps) if normalize_step_type(str(s["type"])) == PERTURB_DELETE
    )


def chain_genes(sub: Substitution) -> set[int]:
    if isinstance(sub, MixedSubstitution):
        return set(sub.oe) | set(sub.ko)
    return {int(c) for c in sub}


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
    delete_steps: frozenset[int] = frozenset(),
) -> list[Substitution]:
    """One chain-to-placebo gene map per usable estimation placebo.

    Slots are (step, position within step). Placebos sharing a gene with the chain
    are left out. Without delete steps each map is a plain dict; with them, a
    ``MixedSubstitution`` split by step type.
    """
    slots = [(i, int(t)) for i, ts in enumerate(chain_token_by_step) for t in ts]
    chain = [t for _, t in slots]
    out: list[Substitution] = []
    for p in placebos:
        flat = [int(t) for ts in p for t in ts]
        if len(flat) != len(chain):
            raise ValueError(f"placebo has {len(flat)} slots, chain has {len(chain)}")
        if set(flat) & set(chain):
            continue
        if not delete_steps:
            out.append(dict(zip(chain, flat)))
            continue
        out.append(MixedSubstitution(
            oe={t: q for (i, t), q in zip(slots, flat) if i not in delete_steps},
            ko={t: q for (i, t), q in zip(slots, flat) if i in delete_steps},
        ))
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


def n_usable_placebos(start_ids: Sequence[int], subs: Sequence[Substitution]) -> int:
    """Placebos whose delete genes are all in the start encoding, for the factors it holds."""
    cell = {int(t) for t in start_ids}
    return sum(
        all(int(q) in cell for t, q in sub.ko.items() if int(t) in cell)
        if isinstance(sub, MixedSubstitution) else True
        for sub in subs
    )


def undo_delete_redo(
    ids: Sequence[int],
    start_ids: Sequence[int],
    sub: MixedSubstitution,
    max_len: int | None = None,
) -> tuple[list[int], set[int]] | None:
    """Counterfactual encoding of a chain with delete steps under one placebo.

    Overexpression slots are swapped as in ``swap_tokens``. For each delete slot whose
    factor was in the start encoding, the factor goes back right after the nearest
    gene that preceded it in ``start_ids``, is in the encoding and is not a placebo
    gene (else after the leading swapped overexpression genes, else first), in
    start-encoding order; then the placebo gene is deleted. Returns ``None`` when a
    placebo gene to delete is missing from the encoding (the placebo gives no
    counterfactual for this cell); otherwise the encoding and the genes with no
    counterfactual state.
    """
    start = [int(t) for t in start_ids]
    pos0 = {t: j for j, t in enumerate(start)}
    ko = {int(t): int(q) for t, q in sub.ko.items() if int(t) in pos0}
    present = {int(t) for t in ids}
    if any(q not in present for q in ko.values()):
        return None
    new, excl = swap_tokens(ids, sub.oe)
    placebo = {int(q) for q in sub.oe.values()} | {int(q) for q in sub.ko.values()}
    oe_placed = excl & {int(q) for q in sub.oe.values()}
    front = 0
    while front < len(new) and new[front] in oe_placed:
        front += 1
    in_new = set(new)
    for t in sorted(ko, key=pos0.__getitem__):
        if t in in_new:
            continue
        at = front
        for a in reversed(start[: pos0[t]]):
            if a in in_new and a not in placebo:
                at = new.index(a) + 1
                break
        new.insert(at, t)
        in_new.add(t)
    drop = set(ko.values())
    new = [t for t in new if t not in drop]
    if max_len is not None:
        new = new[: int(max_len)]
    return new, excl | drop


def placebo_mediator(
    model,
    pair: gs.GeneStatePair,
    subs: Sequence[Substitution],
    *,
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """``(m_g [L_pert, d], n_placebos [L_pert])`` per cell of ``pair``.

    ``m_g`` is zero where no placebo gives a state (chain genes, genes missing from
    the control). Cells with no chain gene in the encoding are not re-encoded:
    their counterfactual is the perturbed state itself, so the contrast is exactly 0.
    With delete steps, ``pair.ctrl_ids`` must be the start encodings.
    """
    dev = pair.h_pert.device
    d = pair.h_pert.size(-1)
    ctrl = [gs.align_ctrl(ids, pair.ctrl_ids[r], pair.h_ctrl[r])
            for r, ids in enumerate(pair.pert_ids)]
    sums = [torch.zeros(len(ids), d, device=dev) for ids in pair.pert_ids]
    counts = [torch.zeros(len(ids), device=dev) for ids in pair.pert_ids]
    for sub in subs:
        swapped, excluded, usable = [], [], []
        for r, ids in enumerate(pair.pert_ids):
            if isinstance(sub, MixedSubstitution):
                got = undo_delete_redo(ids, pair.ctrl_ids[r], sub, model_input_size)
            else:
                got = swap_tokens(ids, sub, model_input_size)
            usable.append(got is not None)
            new, excl = got if got is not None else (list(ids), set())
            swapped.append(new)
            excluded.append(excl)
        rerun = [r for r, ex in enumerate(excluded) if ex]
        h = (gs._hidden_states(model, [swapped[r] for r in rerun], pad_token_id, layer_to_quant)
             if rerun else None)
        h_row = {r: i for i, r in enumerate(rerun)}
        for r, ids in enumerate(pair.pert_ids):
            if not usable[r]:
                continue
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
    subs: Sequence[Substitution],
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
    subs: Sequence[Substitution],
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
    state. With delete steps only treated cells are used (every deleted factor in the
    control encoding): elsewhere ``delta_h`` is 0 and the contrast would be minus the
    placebo deletion effect.
    """
    rng = random.Random(int(seed))
    rows = list(range(len(pert_ds)))
    if max_cells is not None and len(rows) > int(max_cells):
        rows = rows[: int(max_cells)]
    chain = set().union(*(chain_genes(sub) for sub in subs))
    deleted = set().union(*(set(sub.ko) for sub in subs if isinstance(sub, MixedSubstitution)))

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
            if deleted and not deleted <= {int(t) for t in pair.ctrl_ids[row]}:
                continue
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
