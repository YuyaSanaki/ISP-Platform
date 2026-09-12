"""
Model / organism registry for ISP³ Platform (P0).

Resolves species.model_organism + species.model → checkpoint paths, dict paths,
max_input_size, and whether ortholog gene conversion is required.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

GENEFORMER_PKG_ROOT = Path(__file__).resolve().parent.parent
DICTS_ROOT = GENEFORMER_PKG_ROOT / "dicts"
DEFAULT_MODELS_ROOT = Path(os.environ.get("GENEFORMER_MODELS_ROOT", "/app/models"))


class Organism(str, Enum):
    MOUSE = "mouse"
    HUMAN = "human"
    DROSOPHILA = "drosophila"


class ModelId(str, Enum):
    MOUSE_GENEFORMER = "mouse_geneformer"
    HUMAN_GENEFORMER = "human_geneformer"


class HumanVariant(str, Enum):
    V2_104M = "v2_104m"
    V2_316M = "v2_316m"


class MouseVariant(str, Enum):
    BASE = "base"
    L12_E20 = "12l_e20"


class OrthologPolicy(str, Enum):
    """How to resolve many-to-many orthologs before tokenization / ISP."""

    ONE2ONE = "one2one"  # keep reciprocal 1:1 only; never sum counts
    BEST_OF_N = "best_of_n"  # N→1: keep highest-median source gene (no sum)
    LEGACY_SUM = "legacy_sum"  # last-write-wins + expression sum (old behavior)


@dataclass(frozen=True)
class BackendSpec:
    model_id: ModelId
    native_organism: Organism
    max_input_size: int
    pretrained_dir: Path
    token_dictionary: Path
    gene_median_dictionary: Path
    gene_symbol_to_ensembl: Path | None
    hf_hub_id: str | None = None
    human_variant: HumanVariant | None = None
    mouse_variant: MouseVariant | None = None

    @property
    def pretrained_model(self) -> str:
        return str(self.pretrained_dir)


def _dicts_mouse() -> Path:
    return DICTS_ROOT / "mouse"


def _dicts_human() -> Path:
    return DICTS_ROOT / "human"


def _legacy_dicts() -> Path:
    """Flat dicts/ layout used before P0 reorganization."""
    return DICTS_ROOT


def _first_existing(*candidates: Path) -> Path:
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def _mouse_token_dictionary() -> Path:
    mouse = _dicts_mouse()
    legacy = _legacy_dicts()
    return _first_existing(
        mouse / "MLM-re_token_dictionary_v1.pkl",
        legacy / "MLM-re_token_dictionary_v1.pkl",
    )


def _mouse_gene_median_dictionary() -> Path:
    mouse = _dicts_mouse()
    legacy = _legacy_dicts()
    return _first_existing(
        mouse / "mouse_gene_median_dictionary.pkl",
        legacy / "mouse_gene_median_dictionary.pkl",
    )


def _mouse_gene_symbol_to_ensembl() -> Path:
    mouse = _dicts_mouse()
    legacy = _legacy_dicts()
    return _first_existing(
        mouse / "MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl",
        legacy / "MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl",
    )


def _human_pretrained_dir(variant: HumanVariant) -> Path:
    if variant is HumanVariant.V2_316M:
        return DEFAULT_MODELS_ROOT / "human-Geneformer-V2-316M"
    return DEFAULT_MODELS_ROOT / "human-Geneformer-V2-104M"


def _mouse_pretrained_dir(variant: MouseVariant) -> Path:
    if variant is MouseVariant.L12_E20:
        return DEFAULT_MODELS_ROOT / "mouse-Geneformer-12L-E20"
    return DEFAULT_MODELS_ROOT / "mouse-Geneformer"


def _mouse_backend(variant: MouseVariant) -> BackendSpec:
    return BackendSpec(
        model_id=ModelId.MOUSE_GENEFORMER,
        native_organism=Organism.MOUSE,
        max_input_size=2048,
        pretrained_dir=_mouse_pretrained_dir(variant),
        token_dictionary=_mouse_token_dictionary(),
        gene_median_dictionary=_mouse_gene_median_dictionary(),
        gene_symbol_to_ensembl=_mouse_gene_symbol_to_ensembl(),
        mouse_variant=variant,
    )


def _human_backend(variant: HumanVariant) -> BackendSpec:
    return BackendSpec(
        model_id=ModelId.HUMAN_GENEFORMER,
        native_organism=Organism.HUMAN,
        max_input_size=4096,
        pretrained_dir=_human_pretrained_dir(variant),
        token_dictionary=_dicts_human() / "token_dictionary_gc104M.pkl",
        gene_median_dictionary=_dicts_human() / "gene_median_dictionary_gc104M.pkl",
        gene_symbol_to_ensembl=_dicts_human() / "gene_name_id_dict_gc104M.pkl",
        hf_hub_id="ctheodoris/Geneformer",
        human_variant=variant,
    )


# Keys are (model_id, human_variant|None, mouse_variant|None). Exactly one of the
# variant slots is set for a given model family.
BACKENDS: dict[tuple[ModelId, HumanVariant | None, MouseVariant | None], BackendSpec] = {
    (ModelId.MOUSE_GENEFORMER, None, MouseVariant.BASE): _mouse_backend(MouseVariant.BASE),
    (ModelId.MOUSE_GENEFORMER, None, MouseVariant.L12_E20): _mouse_backend(MouseVariant.L12_E20),
    (ModelId.HUMAN_GENEFORMER, HumanVariant.V2_104M, None): _human_backend(HumanVariant.V2_104M),
    (ModelId.HUMAN_GENEFORMER, HumanVariant.V2_316M, None): _human_backend(HumanVariant.V2_316M),
}


def parse_species_config(config: Mapping[str, Any] | None) -> dict[str, str]:
    """Normalize species block from pipeline or stage YAML."""
    raw = config or {}
    if not isinstance(raw, Mapping):
        raise ValueError("species config must be a mapping")

    model_organism = str(raw.get("model_organism", Organism.MOUSE.value)).strip().lower()
    model = str(raw.get("model", ModelId.MOUSE_GENEFORMER.value)).strip().lower()
    human_variant = str(raw.get("human_variant", HumanVariant.V2_104M.value)).strip().lower()
    mouse_variant = str(raw.get("mouse_variant", MouseVariant.BASE.value)).strip().lower()
    ortholog_policy = str(
        raw.get("ortholog_policy", OrthologPolicy.ONE2ONE.value)
    ).strip().lower()
    overlay_raw = raw.get("ortholog_curated_overlay")
    if overlay_raw is None or str(overlay_raw).strip() in ("", "null", "None"):
        ortholog_curated_overlay = ""
    else:
        ortholog_curated_overlay = str(overlay_raw).strip()

    try:
        Organism(model_organism)
    except ValueError as exc:
        valid = ", ".join(o.value for o in Organism)
        raise ValueError(f"species.model_organism must be one of: {valid}") from exc

    try:
        ModelId(model)
    except ValueError as exc:
        valid = ", ".join(m.value for m in ModelId)
        raise ValueError(f"species.model must be one of: {valid}") from exc

    try:
        OrthologPolicy(ortholog_policy)
    except ValueError as exc:
        valid = ", ".join(p.value for p in OrthologPolicy)
        raise ValueError(f"species.ortholog_policy must be one of: {valid}") from exc

    if model == ModelId.HUMAN_GENEFORMER.value:
        try:
            HumanVariant(human_variant)
        except ValueError as exc:
            valid = ", ".join(v.value for v in HumanVariant)
            raise ValueError(f"species.human_variant must be one of: {valid}") from exc

    if model == ModelId.MOUSE_GENEFORMER.value:
        try:
            MouseVariant(mouse_variant)
        except ValueError as exc:
            valid = ", ".join(v.value for v in MouseVariant)
            raise ValueError(f"species.mouse_variant must be one of: {valid}") from exc

    return {
        "model_organism": model_organism,
        "model": model,
        "human_variant": human_variant if model == ModelId.HUMAN_GENEFORMER.value else "",
        "mouse_variant": mouse_variant if model == ModelId.MOUSE_GENEFORMER.value else "",
        "ortholog_policy": ortholog_policy,
        "ortholog_curated_overlay": ortholog_curated_overlay,
    }


def get_backend(species: Mapping[str, Any] | None) -> BackendSpec:
    parsed = parse_species_config(species)
    model_id = ModelId(parsed["model"])
    human_variant: HumanVariant | None = None
    mouse_variant: MouseVariant | None = None
    if model_id is ModelId.HUMAN_GENEFORMER:
        human_variant = HumanVariant(parsed["human_variant"])
    elif model_id is ModelId.MOUSE_GENEFORMER:
        mouse_variant = MouseVariant(parsed["mouse_variant"])
    key = (model_id, human_variant, mouse_variant)
    if key not in BACKENDS:
        raise KeyError(f"No backend registered for {key!r}")
    return BACKENDS[key]


def native_organism(model_id: ModelId | str) -> Organism:
    mid = ModelId(model_id) if isinstance(model_id, str) else model_id
    for (registered_id, _hv, _mv), spec in BACKENDS.items():
        if registered_id is mid:
            return spec.native_organism
    raise KeyError(f"Unknown model: {mid}")


def needs_gene_conversion(model_organism: Organism | str, model_id: ModelId | str) -> bool:
    org = Organism(model_organism) if isinstance(model_organism, str) else model_organism
    return org is not native_organism(model_id)


def resolve_pretrained_path(
    species: Mapping[str, Any] | None,
    paths_cfg: Mapping[str, Any] | None = None,
) -> str:
    """Explicit paths.pretrained_model wins; otherwise resolve from species.model."""
    paths_cfg = paths_cfg or {}
    explicit = paths_cfg.get("pretrained_model")
    if explicit:
        return str(explicit)
    return get_backend(species).pretrained_model
