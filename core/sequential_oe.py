"""Sequential length-preserving OE / KD on rank-value encodings (token space).

Each step applies one (or more) genes on ``input_ids`` — never on embeddings.

- overexpress (OE): move-to-front insert, matching the platform group-OE operator.
  Rank convention: later OE steps call ``insert(0)``, so the most recently added
  factor occupies the highest rank (leftmost token).
- delete (KD): remove those tokens from the encoding (length shrinks).
"""
from __future__ import annotations

import itertools
from typing import Any, Mapping, Sequence

PERTURB_OVEREXPRESS = "overexpress"
PERTURB_DELETE = "delete"
VALID_STEP_TYPES = frozenset({PERTURB_OVEREXPRESS, PERTURB_DELETE})

OSKM_FACTOR_KEYS: tuple[str, ...] = ("O", "S", "K", "M")

OSKM_FACTORS: dict[str, dict[str, str]] = {
    "O": {
        "human": "ENSG00000204531",
        "mouse": "ENSMUSG00000024406",
        "symbol": "POU5F1",
    },
    "S": {
        "human": "ENSG00000181449",
        "mouse": "ENSMUSG00000074637",
        "symbol": "SOX2",
    },
    "K": {
        "human": "ENSG00000136826",
        "mouse": "ENSMUSG00000003032",
        "symbol": "KLF4",
    },
    "M": {
        "human": "ENSG00000136997",
        "mouse": "ENSMUSG00000022346",
        "symbol": "MYC",
    },
}


def _delete_indices(example: dict[str, Any]) -> dict[str, Any]:
    indices = example["perturb_index"]
    for index in sorted(indices, reverse=True):
        del example["input_ids"][index]
    return example


def _overexpress_tokens(example: dict[str, Any]) -> dict[str, Any]:
    """Length-preserving OE (mirrors ``geneformer.in_silico_perturber.overexpress_tokens``)."""
    ids = example["input_ids"]
    if not isinstance(ids, list):
        ids = list(ids)
        example["input_ids"] = ids
    orig_len = len(ids)
    if example["perturb_index"] != [-100]:
        example = _delete_indices(example)
        ids = example["input_ids"]
        if not isinstance(ids, list):
            ids = list(ids)
            example["input_ids"] = ids
    for token in example["tokens_to_perturb"][::-1]:
        ids.insert(0, token)
    overflow = len(ids) - orig_len
    if overflow > 0:
        del ids[-overflow:]
    if "length" in example:
        example["length"] = len(ids)
    return example


def order_label(factor_keys: Sequence[str]) -> str:
    """Human-readable order tag, e.g. ``O-S-K-M``."""
    return "-".join(factor_keys)


COCKTAIL_FACTOR_KEYS: tuple[str, ...] = (
    "NANOG",
    "OCT4",
    "SOX2",
    "ESRRB",
    "LIN28A",
    "DPPA4",
    "TERT",
)

COCKTAIL_FACTORS: dict[str, dict[str, str]] = {
    "NANOG": {"human": "ENSG00000111704", "symbol": "NANOG"},
    "OCT4": {"human": "ENSG00000204531", "symbol": "POU5F1"},
    "SOX2": {"human": "ENSG00000181449", "symbol": "SOX2"},
    "ESRRB": {"human": "ENSG00000119715", "symbol": "ESRRB"},
    "LIN28A": {"human": "ENSG00000131914", "symbol": "LIN28A"},
    "DPPA4": {"human": "ENSG00000121570", "symbol": "DPPA4"},
    "TERT": {"human": "ENSG00000164362", "symbol": "TERT"},
}


def all_oskm_orders(factor_keys: Sequence[str] | None = None) -> list[tuple[str, ...]]:
    """All permutations of the four Yamanaka factor keys (24 by default)."""
    keys = tuple(factor_keys or OSKM_FACTOR_KEYS)
    return list(itertools.permutations(keys))


def normalize_step_type(perturb_type: str | None) -> str:
    """Map UI / YAML aliases onto ``overexpress`` or ``delete``."""
    raw = str(perturb_type or PERTURB_OVEREXPRESS).strip().lower()
    if raw in {"overexpress", "oe", "overexpression"}:
        return PERTURB_OVEREXPRESS
    if raw in {"delete", "kd", "knockdown", "knock-down", "knock_down"}:
        return PERTURB_DELETE
    raise ValueError(
        f"Unsupported sequential step type {perturb_type!r}; "
        f"use overexpress (OE) or delete (KD)."
    )


def parse_sequential_steps(seq_cfg: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Normalize ``sequential.steps`` from YAML into typed gene lists."""
    raw_steps = (seq_cfg or {}).get("steps") or []
    if not isinstance(raw_steps, list) or not raw_steps:
        return []
    steps: list[dict[str, Any]] = []
    for i, item in enumerate(raw_steps, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"sequential.steps[{i-1}] must be a mapping with type and genes")
        ptype = normalize_step_type(item.get("type") or item.get("perturbation_type"))
        genes = item.get("genes") if item.get("genes") is not None else item.get("genes_to_perturb")
        if genes is None:
            genes = []
        if isinstance(genes, str):
            genes = [g.strip() for g in genes.replace(",", "\n").splitlines() if g.strip()]
        else:
            genes = [str(g).strip() for g in genes if str(g).strip()]
        if not genes:
            raise ValueError(f"sequential.steps[{i-1}] ({ptype}) has no genes")
        name = str(item.get("name") or item.get("tag") or "").strip() or f"step{i:02d}_{ptype}"
        steps.append({"name": name, "type": ptype, "genes": genes, "index": i})
    return steps


def apply_single_step_overexpress(example: Mapping[str, Any], tokens: Sequence[int]) -> dict[str, Any]:
    """Length-preserving OE for one step (one or more tokens inserted together)."""
    ex = dict(example)
    ids = list(ex["input_ids"])
    ex["input_ids"] = ids
    token_list = list(tokens)
    ex["tokens_to_perturb"] = token_list
    present = [ids.index(t) for t in token_list if t in ids]
    ex["perturb_index"] = present if present else [-100]
    ex = _overexpress_tokens(ex)
    length = len(ex["input_ids"])
    ex["length"] = length
    ex["attention_mask"] = [1] * length
    return ex


def apply_single_step_delete(example: Mapping[str, Any], tokens: Sequence[int]) -> dict[str, Any]:
    """Knockdown: remove the given tokens from the rank-value encoding."""
    ex = dict(example)
    ids = list(ex["input_ids"])
    token_set = set(tokens)
    indices = [i for i, tok in enumerate(ids) if tok in token_set]
    for index in sorted(indices, reverse=True):
        del ids[index]
    ex["input_ids"] = ids
    length = len(ids)
    ex["length"] = length
    ex["attention_mask"] = [1] * length
    return ex


def apply_step(
    example: Mapping[str, Any],
    tokens: Sequence[int],
    perturb_type: str = PERTURB_OVEREXPRESS,
) -> dict[str, Any]:
    """Apply one sequential OE or KD step on ``input_ids``."""
    ptype = normalize_step_type(perturb_type)
    if ptype == PERTURB_DELETE:
        return apply_single_step_delete(example, tokens)
    return apply_single_step_overexpress(example, tokens)


def apply_sequential_overexpress(
    example: Mapping[str, Any],
    step_token_lists: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Chain length-preserving OE steps on rank-value ``input_ids``."""
    ex = dict(example)
    for tokens in step_token_lists:
        ex = apply_single_step_overexpress(ex, tokens)
    return ex


def apply_sequential_perturb(
    example: Mapping[str, Any],
    steps: Sequence[tuple[str, Sequence[int]]],
) -> dict[str, Any]:
    """Chain typed OE / KD steps on rank-value ``input_ids``."""
    ex = dict(example)
    for perturb_type, tokens in steps:
        ex = apply_step(ex, tokens, perturb_type)
    return ex


def perturb_dataset_sequential(
    dataset,
    step_token_lists: Sequence[Sequence[int]],
    batch_size: int = 64,
):
    """Apply sequential OE to every row in a HuggingFace dataset."""
    steps = [list(s) for s in step_token_lists]

    def _map_batch(batch):
        out_ids, out_len, out_mask = [], [], []
        for i in range(len(batch["input_ids"])):
            row = {k: batch[k][i] for k in batch}
            pert = apply_sequential_overexpress(row, steps)
            out_ids.append(pert["input_ids"])
            out_len.append(pert["length"])
            out_mask.append(pert["attention_mask"])
        return {"input_ids": out_ids, "length": out_len, "attention_mask": out_mask}

    cols = [c for c in dataset.column_names if c not in {"input_ids", "length", "attention_mask"}]
    meta = {c: dataset[c] for c in cols}
    perturbed = dataset.map(
        _map_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names,
    )
    for col, values in meta.items():
        perturbed = perturbed.add_column(col, values)
    return perturbed


def perturb_index_for_tokens(input_ids: Sequence[int], oe_tokens: Sequence[int]) -> list[int] | list:
    """Present-gene indices for group-OE alignment (``[-100]`` if all absent)."""
    present = [input_ids.index(t) for t in oe_tokens if t in input_ids]
    return present if present else [-100]


def front_token_order_after_steps(step_token_lists: Sequence[Sequence[int]]) -> list[int]:
    """Token order at the sequence front after sequential steps (last step leftmost)."""
    front: list[int] = []
    for tokens in step_token_lists:
        for t in reversed(list(tokens)):
            if t in front:
                front.remove(t)
            front.insert(0, t)
    return front


def tokens_for_order(
    order: Sequence[str],
    token_by_factor: Mapping[str, int],
) -> list[list[int]]:
    """Per-step single-token lists for a factor-key permutation."""
    return [[token_by_factor[k]] for k in order]


def simultaneous_token_order(step_token_lists: Sequence[Sequence[int]]) -> list[int]:
    """Front token order if all steps were applied in one group-OE call."""
    return [t for step in step_token_lists for t in step]
