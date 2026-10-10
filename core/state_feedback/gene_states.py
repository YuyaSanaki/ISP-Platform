"""Gene-level hidden states and MLM self-logits for State-feedback ISP.

The permutation decoder needs, per cell and per gene token, the perturbation
effect in the fine-tuned representation space:

    delta_h[gene] = h_pert[gene] - h_ctrl[gene]

``h_pert`` comes from the current (perturbed) rank-value encoding, ``h_ctrl``
from a reference encoding of the same cell. The two encodings do not share
positions, so alignment is by **token identity**: a gene present only in the
perturbed list (e.g. the freshly inserted OE factor) gets ``delta_h = 0`` and
therefore keeps its base rank.

Hidden states are never accumulated across the whole dataset; callers consume
one chunk at a time so peak memory stays at ``batch x seq_len x hidden``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch

from geneformer import in_silico_perturber as isp


def model_config(model):
    """Config of a possibly accelerate-wrapped model."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        cfg = getattr(getattr(model, "module", None), "config", None)
    if cfg is None:
        raise RuntimeError("Could not resolve model config for hidden-size lookup")
    return cfg


def hidden_size(model) -> int:
    return int(model_config(model).hidden_size)


def raw_input_ids(dataset, start: int, stop: int, model_input_size: int) -> list[list[int]]:
    """Token lists for rows ``[start, stop)``, truncated to the model input size."""
    try:
        dataset.reset_format()
    except Exception:
        pass
    rows = dataset.select(list(range(start, stop)))["input_ids"]
    return [[int(t) for t in row][:model_input_size] for row in rows]


def batch_tensors(
    raw_ids: Sequence[Sequence[int]],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad token lists into ``(input_ids, attention_mask)`` on the ISP device."""
    max_len = max(len(row) for row in raw_ids)
    ids = torch.full((len(raw_ids), max_len), int(pad_token_id), dtype=torch.long)
    mask = torch.zeros((len(raw_ids), max_len), dtype=torch.long)
    for k, row in enumerate(raw_ids):
        n = len(row)
        ids[k, :n] = torch.tensor(row, dtype=torch.long)
        mask[k, :n] = 1
    return isp._tensor_to_device(ids), isp._tensor_to_device(mask)


@dataclass
class GeneStatePair:
    """One chunk of aligned control / perturbed gene states."""

    cell_indices: list[int]
    ctrl_ids: list[list[int]]
    pert_ids: list[list[int]]
    h_ctrl: torch.Tensor  # [b, L_ctrl, d]
    h_pert: torch.Tensor  # [b, L_pert, d]

    def delta_h(self, row: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(delta_h, valid_mask)`` for one cell, indexed by perturbed position."""
        return align_delta_h(
            self.ctrl_ids[row],
            self.pert_ids[row],
            self.h_ctrl[row],
            self.h_pert[row],
        )

    def aligned_states(self, row: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(h_pert, h_ctrl_aligned, valid_mask)`` for one cell."""
        pert_ids = self.pert_ids[row]
        h_ctrl_aligned, valid = align_ctrl(pert_ids, self.ctrl_ids[row], self.h_ctrl[row])
        return self.h_pert[row][: len(pert_ids)].float(), h_ctrl_aligned, valid


def align_ctrl(
    pert_ids: Sequence[int],
    ctrl_ids: Sequence[int],
    h_ctrl_cell: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Control hidden states reindexed onto the perturbed gene order.

    Returns ``(h_ctrl_aligned [L_pert, d], valid_mask [L_pert])``. Genes missing
    from the control encoding are zero-filled and flagged invalid.
    """
    ctrl_pos = {int(t): i for i, t in enumerate(ctrl_ids)}
    n_pert = len(pert_ids)
    device = h_ctrl_cell.device
    gather = torch.full((n_pert,), -1, dtype=torch.long)
    for j, tok in enumerate(pert_ids):
        pos = ctrl_pos.get(int(tok))
        if pos is not None and pos < h_ctrl_cell.size(0):
            gather[j] = pos
    valid = (gather >= 0).to(device)
    aligned = h_ctrl_cell.index_select(0, gather.clamp(min=0).to(device)).float()
    return aligned * valid.unsqueeze(-1), valid


def align_delta_h(
    ctrl_ids: Sequence[int],
    pert_ids: Sequence[int],
    h_ctrl_cell: torch.Tensor,
    h_pert_cell: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-identity aligned ``h_pert - h_ctrl`` over the perturbed gene list.

    Returns ``(delta_h [L_pert, d], valid_mask [L_pert])``. Genes absent from the
    control encoding get a zero delta so the decoder sees "no perturbation
    evidence" rather than an absolute hidden state.
    """
    n_pert = len(pert_ids)
    h_ctrl_aligned, valid = align_ctrl(pert_ids, ctrl_ids, h_ctrl_cell)
    delta = (h_pert_cell[:n_pert].float() - h_ctrl_aligned) * valid.unsqueeze(-1)
    return delta, valid


def iter_gene_state_pairs(
    model,
    ctrl_ds,
    pert_ds,
    *,
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    rows: Sequence[int] | None = None,
) -> Iterator[GeneStatePair]:
    """Yield aligned control/perturbed gene hidden states in chunks.

    ``ctrl_ds`` and ``pert_ds`` must be row-aligned (same cells, same order).
    """
    n = len(pert_ds)
    if len(ctrl_ds) != n:
        raise ValueError(f"ctrl_ds ({len(ctrl_ds)}) and pert_ds ({n}) must be row-aligned")
    order = list(range(n)) if rows is None else [int(r) for r in rows]
    batch = max(1, int(forward_batch_size))
    for i in range(0, len(order), batch):
        idx = order[i : i + batch]
        lo, hi = min(idx), max(idx) + 1
        contiguous = idx == list(range(lo, hi))
        if contiguous:
            ctrl_raw = raw_input_ids(ctrl_ds, lo, hi, model_input_size)
            pert_raw = raw_input_ids(pert_ds, lo, hi, model_input_size)
        else:
            ctrl_raw = [raw_input_ids(ctrl_ds, r, r + 1, model_input_size)[0] for r in idx]
            pert_raw = [raw_input_ids(pert_ds, r, r + 1, model_input_size)[0] for r in idx]

        h_ctrl = _hidden_states(model, ctrl_raw, pad_token_id, layer_to_quant)
        h_pert = _hidden_states(model, pert_raw, pad_token_id, layer_to_quant)
        yield GeneStatePair(
            cell_indices=list(idx),
            ctrl_ids=ctrl_raw,
            pert_ids=pert_raw,
            h_ctrl=h_ctrl,
            h_pert=h_pert,
        )
        del h_ctrl, h_pert


def _hidden_states(
    model,
    raw_ids: Sequence[Sequence[int]],
    pad_token_id: int,
    layer_to_quant: int,
) -> torch.Tensor:
    input_ids, attention_mask = batch_tensors(raw_ids, pad_token_id)
    with torch.no_grad():
        outputs = isp._forward_with_oom_retry(
            lambda: model(input_ids=input_ids, attention_mask=attention_mask),
        )
    embs = isp.batched_layer_embs(outputs, layer_to_quant).detach().clone()
    del outputs
    return embs


def self_logits(
    mlm_model,
    raw_ids: Sequence[Sequence[int]],
    pad_token_id: int,
) -> list[torch.Tensor]:
    """MLM logit of each token at its own position (one tensor per cell).

    The vocabulary axis is reduced inside ``no_grad`` so the full
    ``[cell, pos, vocab]`` tensor is never held longer than one forward.
    """
    input_ids, attention_mask = batch_tensors(raw_ids, pad_token_id)
    with torch.no_grad():
        outputs = isp._forward_with_oom_retry(
            lambda: mlm_model(input_ids=input_ids, attention_mask=attention_mask),
        )
        logits = outputs.logits
        picked = logits.gather(2, input_ids.unsqueeze(-1)).squeeze(-1).float().detach().clone()
        del outputs, logits
    return [picked[k, : len(raw_ids[k])] for k in range(len(raw_ids))]


def iter_self_logit_pairs(
    mlm_model,
    ctrl_ds,
    pert_ds,
    *,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
) -> Iterator[tuple[list[int], list[list[int]], list[list[int]], list[torch.Tensor], list[torch.Tensor]]]:
    """Yield ``(cell_indices, ctrl_ids, pert_ids, ctrl_self_logits, pert_self_logits)``."""
    n = len(pert_ds)
    if len(ctrl_ds) != n:
        raise ValueError(f"ctrl_ds ({len(ctrl_ds)}) and pert_ds ({n}) must be row-aligned")
    batch = max(1, int(forward_batch_size))
    for i in range(0, n, batch):
        hi = min(i + batch, n)
        ctrl_raw = raw_input_ids(ctrl_ds, i, hi, model_input_size)
        pert_raw = raw_input_ids(pert_ds, i, hi, model_input_size)
        yield (
            list(range(i, hi)),
            ctrl_raw,
            pert_raw,
            self_logits(mlm_model, ctrl_raw, pad_token_id),
            self_logits(mlm_model, pert_raw, pad_token_id),
        )


def load_mlm_model(pretrained_dir: str):
    """Pretrained ``BertForMaskedLM`` for the delta-MLM baseline.

    Only the **difference** of self-logits is used; the absolute logits of a
    pretrained head are not comparable with fine-tuned representations.
    """
    return isp.load_model("Pretrained", 0, str(pretrained_dir))
