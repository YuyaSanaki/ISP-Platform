"""Dataset, GPU and scoring helpers shared by the State-feedback runners.

Start / centroid cell selection, dual-forward batch sizing, typed rank-edit steps
on a dataset row, and the two ``goal_state_shift`` scorers: group aligned
(``quant_cos_sims``) and cell-mean (mixed OE+KD, no gene-rank alignment).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd
import torch

from geneformer import in_silico_perturber as isp
from rank_edit import (
    PERTURB_OVEREXPRESS,
    apply_step,
    distinct_tokens,
    normalize_step_type,
    perturb_index_for_tokens,
)

# Baseline condition: the same steps with no feedback between them.
NO_FEEDBACK = "no_feedback"
# Name the baseline had while Ordered rank-edit ISP was a separate run type.
LEGACY_CONDITION_NAMES = {"ordered_rank_edit": NO_FEEDBACK}


def canonical_condition(name: str) -> str:
    return LEGACY_CONDITION_NAMES.get(str(name), str(name))


def gpu_resident_map_workers(nproc: int) -> int:
    """Keep dataset.map single-process while the model occupies the GPU.

    Multiprocessing duplicates the Arrow table in host RAM. On unified-memory
    boxes (GB10 / Grace-Blackwell) that RAM is the same pool as VRAM, so a
    4-worker map during dual-forward scoring is a common OOM trigger.
    """
    if torch.cuda.is_available():
        return 1
    return max(1, min(int(nproc), 4))


def is_cuda_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def centroid_dataset(dataset, state_key: str, states: Sequence[str], max_ncells: int | None, nproc: int):
    """Keep up to max_ncells per state so centroids do not embed the full 6000-cell matrix."""
    if not max_ncells:
        return dataset
    from datasets import concatenate_datasets

    parts = []
    nproc_f = gpu_resident_map_workers(nproc)
    for st in states:
        sub = dataset.filter(lambda ex, s=st: ex[state_key] == s, num_proc=nproc_f)
        n = min(len(sub), int(max_ncells))
        if n == 0:
            raise ValueError(f"No cells with {state_key}={st!r} for state centroids")
        parts.append(sub.select(range(n)))
        print(f"Centroid cells ({st}): n={n}", flush=True)
    return concatenate_datasets(parts)


def select_start_cells(dataset, state_key: str, start_state: str, max_ncells: int | None):
    """Cells of ``start_state``, longest encodings first, capped at ``max_ncells``."""
    data = dataset.filter(lambda x: x.get(state_key) == start_state)
    data = data.sort("length", reverse=True)
    if max_ncells is not None and len(data) > max_ncells:
        data = data.select(range(max_ncells))
    return data


def resolve_dual_forward_batch_size(
    value,
    model,
    input_data,
    pad_token_id,
    model_directory,
) -> int:
    """Auto-batch sized for group scoring (perturbed + original forwards)."""
    return isp.resolve_forward_batch_size(
        value,
        model,
        input_data,
        pad_token_id,
        model_directory,
        n_forwards=2,
        task="rank_edit_dual_forward",
        label="Rank-edit scoring forward_batch_size",
    )


def resolve_gene_tokens(cfg: Mapping[str, Any], genes: Sequence[str]) -> tuple[list[int], list[str]]:
    from run_isp_umap import resolve_perturbation_tokens

    return resolve_perturbation_tokens(cfg, list(genes))


def apply_typed_step(
    example: dict[str, Any],
    tokens: Sequence[int],
    perturb_type: str,
) -> dict[str, Any]:
    return apply_step(example, tokens, perturb_type)


def _shift_frame(original_ds, shifts) -> pd.DataFrame:
    meta = {
        c: original_ds[c]
        for c in original_ds.column_names
        if c not in {"input_ids", "length", "attention_mask"}
    }
    meta["cell_index"] = list(range(len(original_ds)))
    meta["Shift_to_goal_end"] = shifts
    return pd.DataFrame(meta)


def compute_goal_state_shifts(
    model,
    original_ds,
    perturbed_ds,
    oe_tokens: Sequence[int],
    goal_state: str,
    cell_states_to_model: dict[str, Any],
    state_embs_dict: dict[str, torch.Tensor],
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    nproc: int,
    *,
    perturb_type: str = PERTURB_OVEREXPRESS,
    batch_state: list[int] | None = None,
) -> pd.DataFrame:
    """Per-cell ``goal_state_shift`` via ``quant_cos_sims`` (group aligned).

    ``batch_state`` is a one-element list of the live forward batch size. On CUDA
    OOM it is halved and the scoring call is retried (the dual original+perturbed
    forward can exceed a 1-forward auto cache).
    """
    ptype = normalize_step_type(perturb_type)
    oe_tokens = distinct_tokens(oe_tokens)
    orig_input_ids = [list(x) for x in original_ds["input_ids"]]
    indices_to_perturb = [perturb_index_for_tokens(ids, oe_tokens) for ids in orig_input_ids]
    try:
        original_ds.reset_format()
    except Exception:
        pass
    try:
        perturbed_ds.reset_format()
    except Exception:
        pass

    batch = int(forward_batch_size)
    map_workers = gpu_resident_map_workers(nproc)
    while True:
        try:
            cos_dict = isp.quant_cos_sims(
                model,
                ptype,
                perturbed_ds,
                None,
                None,
                batch,
                layer_to_quant,
                original_ds,
                list(oe_tokens),
                indices_to_perturb,
                True,
                cell_states_to_model,
                state_embs_dict,
                pad_token_id,
                model_input_size,
                map_workers,
            )
            break
        except Exception as exc:
            if not is_cuda_oom(exc) or batch <= 1:
                raise
            new_batch = max(1, batch // 2)
            print(
                f"CUDA OOM during rank-edit scoring (batch={batch}); "
                f"retrying at {new_batch}",
                flush=True,
            )
            empty_cuda_cache()
            batch = new_batch
            if batch_state is not None:
                batch_state[0] = batch

    return _shift_frame(original_ds, cos_dict[goal_state].numpy())


def _forward_cell_means(
    model,
    dataset,
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    nproc: int,
    *,
    strip_leading: int = 0,
) -> torch.Tensor:
    """Mean-pooled cell embeddings [n_cells, hidden] for mixed OE+KD scoring."""
    n = len(dataset)
    try:
        dataset.reset_format()
    except Exception:
        pass
    map_workers = gpu_resident_map_workers(nproc)
    chunks: list[torch.Tensor] = []
    batch = max(1, int(forward_batch_size))
    for i in range(0, n, batch):
        max_range = min(i + batch, n)
        minibatch = dataset.select(list(range(i, max_range)))
        lengths = [int(x) for x in minibatch["length"]]
        max_len = min(max(lengths), model_input_size)

        def _pad(example):
            example["input_ids"] = isp.pad_or_truncate_encoding(
                example["input_ids"], pad_token_id, max_len
            )
            return example

        if any(L != max_len for L in lengths) or max(lengths) > model_input_size:
            minibatch = minibatch.map(_pad, num_proc=map_workers)
        minibatch.set_format(type="torch")
        input_ids = isp._tensor_to_device(isp._force_tensor(minibatch["input_ids"]))
        attention_mask = isp.gen_attention_mask(minibatch, max_len)
        with torch.no_grad():
            outputs = isp._forward_with_oom_retry(
                lambda: model(input_ids=input_ids, attention_mask=attention_mask),
            )
        embs = isp.batched_layer_embs(outputs, layer_to_quant).clone()
        del outputs
        if strip_leading > 0:
            embs = embs[:, strip_leading:, :]
            adj = torch.clamp(
                torch.as_tensor(lengths, dtype=torch.long, device=embs.device) - strip_leading,
                min=1,
            )
        else:
            adj = torch.as_tensor(lengths, dtype=torch.long, device=embs.device)
        means = isp.mean_nonpadding_embs(embs, adj)
        chunks.append(means.detach().to("cpu"))
        del embs, means
    return torch.cat(chunks, dim=0)


def compute_cell_mean_goal_state_shifts(
    model,
    original_ds,
    perturbed_ds,
    goal_state: str,
    state_embs_dict: dict[str, torch.Tensor],
    layer_to_quant: int,
    pad_token_id: int,
    model_input_size: int,
    forward_batch_size: int,
    nproc: int,
    *,
    strip_leading: int = 0,
    batch_state: list[int] | None = None,
) -> pd.DataFrame:
    """Cell-level goal_state_shift without gene-rank alignment (mixed OE+KD)."""
    batch = int(forward_batch_size)
    while True:
        try:
            orig_means = _forward_cell_means(
                model,
                original_ds,
                layer_to_quant,
                pad_token_id,
                model_input_size,
                batch,
                nproc,
            )
            pert_means = _forward_cell_means(
                model,
                perturbed_ds,
                layer_to_quant,
                pad_token_id,
                model_input_size,
                batch,
                nproc,
                strip_leading=strip_leading,
            )
            break
        except Exception as exc:
            if not is_cuda_oom(exc) or batch <= 1:
                raise
            new_batch = max(1, batch // 2)
            print(
                f"CUDA OOM during mixed OE+KD scoring (batch={batch}); "
                f"retrying at {new_batch}",
                flush=True,
            )
            empty_cuda_cache()
            batch = new_batch
            if batch_state is not None:
                batch_state[0] = batch

    goal = state_embs_dict[goal_state]
    if goal.dim() == 1:
        goal = goal.unsqueeze(0)
    goal = goal.detach().float().to("cpu")
    if goal.dim() == 3:
        goal = goal.squeeze(1)
    orig_means = orig_means.float()
    pert_means = pert_means.float()
    if goal.size(-1) != orig_means.size(-1):
        raise RuntimeError(
            f"Centroid hidden size {tuple(goal.shape)} does not match "
            f"cell embeddings {tuple(orig_means.shape)}"
        )
    if goal.size(0) == 1 and orig_means.size(0) != 1:
        goal_b = goal.expand(orig_means.size(0), -1)
    else:
        goal_b = goal
    cos = torch.nn.CosineSimilarity(dim=1)
    origin_v_end = cos(orig_means, goal_b)
    perturb_v_end = cos(pert_means, goal_b)
    return _shift_frame(original_ds, (perturb_v_end - origin_v_end).numpy())
