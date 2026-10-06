"""Rank-edit step operators: length-preserving OE / KD on rank-value encodings.

Each step applies one or more genes on ``input_ids``, never on embeddings.
State-feedback ISP (``core/state_feedback/``) applies these steps and, between
steps, reorders the genes from the model output. Its no-feedback baseline applies
the same steps without that reordering.

- overexpress (OE): move-to-front insert, matching the platform group-OE operator.
  Later OE steps call ``insert(0)``, so the most recently added factor occupies the
  highest rank (leftmost token).
- delete (KD): remove those tokens from the encoding (length shrinks).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

PERTURB_OVEREXPRESS = "overexpress"
PERTURB_DELETE = "delete"
VALID_STEP_TYPES = frozenset({PERTURB_OVEREXPRESS, PERTURB_DELETE})

# Step blocks of configs written for the removed Ordered rank-edit runner. State-feedback
# reads their ``steps`` when ``state_feedback.steps`` is absent.
LEGACY_STEP_KEYS: tuple[str, ...] = ("ordered_rank_edit", "sequential")


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


def normalize_step_type(perturb_type: str | None) -> str:
    """Map UI / YAML aliases onto ``overexpress`` or ``delete``."""
    raw = str(perturb_type or PERTURB_OVEREXPRESS).strip().lower()
    if raw in {"overexpress", "oe", "overexpression"}:
        return PERTURB_OVEREXPRESS
    if raw in {"delete", "kd", "knockdown", "knock-down", "knock_down"}:
        return PERTURB_DELETE
    raise ValueError(
        f"Unsupported rank-edit step type {perturb_type!r}; "
        f"use overexpress (OE) or delete (KD)."
    )


def legacy_steps_block(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """First legacy step block (``ordered_rank_edit:`` then ``sequential:``) of a run config."""
    cfg = cfg or {}
    for key in LEGACY_STEP_KEYS:
        block = cfg.get(key)
        if block is not None:
            return dict(block)
    return {}


def parse_steps(block: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Normalize a YAML ``steps`` list (under ``block``) into typed gene lists."""
    raw_steps = (block or {}).get("steps") or []
    if not isinstance(raw_steps, list) or not raw_steps:
        return []
    steps: list[dict[str, Any]] = []
    for i, item in enumerate(raw_steps, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"steps[{i-1}] must be a mapping with type and genes")
        ptype = normalize_step_type(item.get("type") or item.get("perturbation_type"))
        genes = item.get("genes") if item.get("genes") is not None else item.get("genes_to_perturb")
        if genes is None:
            genes = []
        if isinstance(genes, str):
            genes = [g.strip() for g in genes.replace(",", "\n").splitlines() if g.strip()]
        else:
            genes = [str(g).strip() for g in genes if str(g).strip()]
        if not genes:
            raise ValueError(f"steps[{i-1}] ({ptype}) has no genes")
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
    """Apply one rank-edit step (OE or KD) on ``input_ids``."""
    ptype = normalize_step_type(perturb_type)
    if ptype == PERTURB_DELETE:
        return apply_single_step_delete(example, tokens)
    return apply_single_step_overexpress(example, tokens)


def apply_rank_edits(
    example: Mapping[str, Any],
    steps: Sequence[tuple[str, Sequence[int]]],
) -> dict[str, Any]:
    """Apply typed OE / KD steps in order on rank-value ``input_ids``."""
    ex = dict(example)
    for perturb_type, tokens in steps:
        ex = apply_step(ex, tokens, perturb_type)
    return ex


def perturb_index_for_tokens(input_ids: Sequence[int], oe_tokens: Sequence[int]) -> list[int] | list:
    """Present-gene indices for group-OE alignment (``[-100]`` if all absent)."""
    present = [input_ids.index(t) for t in oe_tokens if t in input_ids]
    return present if present else [-100]
