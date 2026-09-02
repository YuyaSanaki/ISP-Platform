"""Species / backend helpers for platform entrypoints (P0)."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

from .backends.registry import BackendSpec, get_backend, parse_species_config
from .gene_converter import (
    convert_perturbation_genes,
    get_backend_for_species,
    load_fly_symbol_table,
    load_ortholog_table,
    remap_adata_ensembl_ids,
    should_convert,
    summarize_conversion,
    conversion_pair,
)


def species_from_config(config: Mapping[str, Any] | None) -> dict[str, str]:
    return parse_species_config((config or {}).get("species"))


def backend_from_config(config: Mapping[str, Any] | None) -> BackendSpec:
    return get_backend_for_species(species_from_config(config))


def load_input_symbol_table(species: Mapping[str, str] | None) -> dict[str, str]:
    """Symbol → primary gene ID for the input model_organism (when dict file exists)."""
    parsed = parse_species_config(species)
    org = parsed["model_organism"]
    if org == "drosophila":
        return load_fly_symbol_table()
    if org == "mouse":
        backend = get_backend(
            {
                "model": "mouse_geneformer",
                "mouse_variant": parsed.get("mouse_variant") or "base",
            }
        )
    elif org == "human":
        backend = get_backend(
            {
                "model": "human_geneformer",
                "human_variant": parsed.get("human_variant") or "v2_104m",
            }
        )
    else:
        return {}
    path = backend.gene_symbol_to_ensembl
    if path is None or not path.is_file():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def default_isp_forward_batch_size(max_input_size: int) -> int:
    """Conservative ISP minibatch default from model sequence length."""
    return 25 if int(max_input_size) > 2048 else 100


def default_finetune_batch_size(max_input_size: int) -> int:
    """Conservative fine-tune batch default from model sequence length."""
    return 2 if int(max_input_size) > 2048 else 6


def log_species_banner(species: Mapping[str, str] | None, *, prefix: str = "") -> str:
    parsed = parse_species_config(species)
    backend = get_backend(parsed)
    pair = conversion_pair(parsed["model_organism"], parsed["model"])
    lines = [
        f"{prefix}species.model_organism: {parsed['model_organism']}",
        f"{prefix}species.model:         {parsed['model']}",
    ]
    if parsed.get("human_variant"):
        lines.append(f"{prefix}species.human_variant:  {parsed['human_variant']}")
    if parsed.get("mouse_variant"):
        lines.append(f"{prefix}species.mouse_variant:  {parsed['mouse_variant']}")
    lines.append(f"{prefix}species.ortholog_policy: {parsed['ortholog_policy']}")
    overlay = (parsed.get("ortholog_curated_overlay") or "").strip()
    if overlay:
        lines.append(f"{prefix}species.ortholog_curated_overlay: {overlay}")
    lines.append(f"{prefix}backend max_input_size:  {backend.max_input_size}")
    lines.append(f"{prefix}pretrained model:       {backend.pretrained_model}")
    if pair:
        lines.append(f"{prefix}gene conversion:       {pair.value} (auto)")
    else:
        lines.append(f"{prefix}gene conversion:       not required")
    msg = "\n".join(lines)
    print(msg)
    return msg


__all__ = [
    "backend_from_config",
    "convert_perturbation_genes",
    "default_finetune_batch_size",
    "default_isp_forward_batch_size",
    "load_input_symbol_table",
    "load_ortholog_table",
    "log_species_banner",
    "remap_adata_ensembl_ids",
    "should_convert",
    "species_from_config",
    "summarize_conversion",
]
