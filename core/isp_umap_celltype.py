"""Marker-gene cell-type prediction for ISP UMAP start-state cells.

Scores each cell by marker tokens present in its Geneformer ``input_ids``
(rank-value encoding). Panels are selected by the **model-native** organism
(mouse vs human) so cross-species tokenized datasets (already remapped into
model vocabulary) still resolve markers. Optional rank weighting prefers
markers that appear early in the rank list (higher expression).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# Fine-grained panels (Igfbp2 analysis final set; Igfbp2 excluded from choroid).
# Mouse symbols (title case) — used when the Geneformer backend is mouse-native.
MOUSE_MARKERS: dict[str, list[str]] = {
    "Vascular_SMC": ["Acta2", "Myh11", "Tagln", "Cnn1", "Myl9", "Tpm2"],
    "Pericyte": ["Pdgfrb", "Rgs5", "Abcc9"],
    "Fibroblast": ["Col1a1", "Dcn", "Lum", "Pdgfra"],
    "Microglia": ["Cx3cr1", "P2ry12", "Tmem119", "Hexb", "C1qa", "Ctss", "Aif1"],
    "Endothelial": ["Cldn5", "Pecam1", "Flt1", "Kdr", "Ly6c1"],
    "Astrocyte": ["Gfap", "Aqp4", "Aldh1l1", "Slc1a3"],
    "OPC": ["Cspg4", "Olig1", "Sox10"],
    "Oligodendrocyte": ["Mbp", "Plp1", "Mog", "Mobp"],
    "Neuron": ["Rbfox3", "Snap25", "Syt1", "Slc17a7", "Gad1"],
    "Choroid_plexus": ["Ttr", "Folr1", "Kcnj13", "Aqp1"],
}

# Human HGNC orthologs of the mouse brain panel (uppercase).
HUMAN_MARKERS: dict[str, list[str]] = {
    "Vascular_SMC": ["ACTA2", "MYH11", "TAGLN", "CNN1", "MYL9", "TPM2"],
    "Pericyte": ["PDGFRB", "RGS5", "ABCC9"],
    "Fibroblast": ["COL1A1", "DCN", "LUM", "PDGFRA"],
    "Microglia": ["CX3CR1", "P2RY12", "TMEM119", "HEXB", "C1QA", "CTSS", "AIF1"],
    "Endothelial": ["CLDN5", "PECAM1", "FLT1", "KDR", "CDH5"],
    "Astrocyte": ["GFAP", "AQP4", "ALDH1L1", "SLC1A3"],
    "OPC": ["CSPG4", "OLIG1", "SOX10"],
    "Oligodendrocyte": ["MBP", "PLP1", "MOG", "MOBP"],
    "Neuron": ["RBFOX3", "SNAP25", "SYT1", "SLC17A7", "GAD1"],
    "Choroid_plexus": ["TTR", "FOLR1", "KCNJ13", "AQP1"],
}

# Back-compat alias.
DEFAULT_MARKERS = MOUSE_MARKERS

MARKERS_BY_ORGANISM: dict[str, dict[str, list[str]]] = {
    "mouse": MOUSE_MARKERS,
    "human": HUMAN_MARKERS,
}

# Coarse vascular vs immune programs (for L2-by-group plots).
COARSE_MICROGLIA_MARKERS_MOUSE = [
    "Cx3cr1",
    "P2ry12",
    "Hexb",
    "C1qa",
    "C1qb",
    "Ctss",
    "Aif1",
    "Apoe",
    "Cd83",
    "Tyrobp",
]
COARSE_SMC_MARKERS_MOUSE = ["Acta2", "Myh11", "Tagln", "Cnn1", "Myl9", "Tpm2", "Lmod1"]

COARSE_MICROGLIA_MARKERS_HUMAN = [
    "CX3CR1",
    "P2RY12",
    "HEXB",
    "C1QA",
    "C1QB",
    "CTSS",
    "AIF1",
    "APOE",
    "CD83",
    "TYROBP",
]
COARSE_SMC_MARKERS_HUMAN = ["ACTA2", "MYH11", "TAGLN", "CNN1", "MYL9", "TPM2", "LMOD1"]

# Legacy aliases used by older callers / docs.
COARSE_MICROGLIA_MARKERS = COARSE_MICROGLIA_MARKERS_MOUSE
COARSE_SMC_MARKERS = COARSE_SMC_MARKERS_MOUSE

# Prefer dataset metadata labels when present (ground truth / external annot).
METADATA_CELLTYPE_COLUMNS = (
    "cell_type",
    "celltype",
    "CellType",
    "cell_type_label",
    "annotation",
)

# Tokens that never contribute to marker scoring.
_PAD_LIKE = frozenset({0})


def select_marker_panel(
    organism: str | None = None,
    markers: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, list[str]]:
    """Return a copy of the marker panel for ``organism`` (or an override)."""
    if markers is not None:
        return {k: list(v) for k, v in markers.items()}
    key = (organism or "mouse").strip().lower()
    if key not in MARKERS_BY_ORGANISM:
        logger.warning(
            "No marker panel for organism=%r; falling back to mouse brain markers",
            organism,
        )
        key = "mouse"
    return {k: list(v) for k, v in MARKERS_BY_ORGANISM[key].items()}


def _coarse_marker_genes(organism: str | None) -> tuple[list[str], list[str]]:
    key = (organism or "mouse").strip().lower()
    if key == "human":
        return list(COARSE_MICROGLIA_MARKERS_HUMAN), list(COARSE_SMC_MARKERS_HUMAN)
    return list(COARSE_MICROGLIA_MARKERS_MOUSE), list(COARSE_SMC_MARKERS_MOUSE)


def _load_dictionaries() -> tuple[dict, dict]:
    """Legacy mouse-only load (kept for callers that omit species)."""
    from geneformer import tokenizer as gf_tokenizer
    from geneformer.in_silico_perturber_stats import GENE_NAME_ID_DICTIONARY_FILE

    with open(gf_tokenizer.TOKEN_DICTIONARY_FILE, "rb") as f:
        token_dict = pickle.load(f)  # ensembl -> token id
    with open(GENE_NAME_ID_DICTIONARY_FILE, "rb") as f:
        name_id = pickle.load(f)  # symbol -> ensembl
    return token_dict, name_id


def load_model_dictionaries(
    species: Mapping[str, Any] | None = None,
) -> tuple[dict, dict, str]:
    """Load token + symbol dicts for the configured Geneformer backend.

    Returns ``(token_dict, name_id, native_organism)``. Falls back to the
    legacy mouse dictionaries when backend paths are missing (unit tests / bare
    checkout without downloaded pickles).
    """
    if not species:
        token_dict, name_id = _load_dictionaries()
        return token_dict, name_id, "mouse"

    try:
        from geneformer.backends.registry import get_backend, parse_species_config

        parsed = parse_species_config(species)
        backend = get_backend(parsed)
        native = backend.native_organism.value
        token_path = Path(backend.token_dictionary)
        sym_path = backend.gene_symbol_to_ensembl
        if not token_path.is_file():
            logger.warning(
                "Backend token dictionary missing (%s); falling back to legacy mouse dicts",
                token_path,
            )
            token_dict, name_id = _load_dictionaries()
            return token_dict, name_id, native
        with open(token_path, "rb") as f:
            token_dict = pickle.load(f)
        name_id: dict = {}
        if sym_path is not None and Path(sym_path).is_file():
            with open(sym_path, "rb") as f:
                name_id = pickle.load(f)
        else:
            logger.warning(
                "Backend gene-symbol dictionary missing (%s); marker symbol lookup may fail",
                sym_path,
            )
        return token_dict, name_id, native
    except Exception as exc:  # noqa: BLE001 — keep UMAP postprocess resilient
        logger.warning("Species-aware dict load failed (%s); using legacy mouse dicts", exc)
        token_dict, name_id = _load_dictionaries()
        return token_dict, name_id, "mouse"


def _symbol_lookup_local(name_id: Mapping, symbol: str) -> str | None:
    """Case-tolerant symbol → Ensembl lookup against a model-native table."""
    if symbol in name_id:
        return name_id[symbol]
    for variant in (symbol.upper(), symbol.lower(), symbol.capitalize()):
        if variant in name_id:
            return name_id[variant]
    # Title-case multi-part (e.g. Slc17a7) vs all-upper human tables.
    if len(symbol) > 1:
        titled = symbol[0].upper() + symbol[1:].lower()
        if titled in name_id:
            return name_id[titled]
    return None


def _genes_to_tokens(
    genes: Sequence[str],
    token_dict: Mapping,
    name_id: Mapping,
    *,
    species: Mapping[str, Any] | None = None,
    panel_organism: str | None = None,
) -> list[int]:
    """Resolve marker symbols to model token IDs.

    Prefer direct model-native symbol lookup. When that fails and ``species`` is
    set, fall back to ``resolve_gene_for_model`` with ``model_organism`` set to
    the marker panel organism (ortholog path for cross-species robustness).
    """
    toks: list[int] = []
    seen: set[int] = set()
    resolver = None
    if species is not None:
        try:
            from geneformer.gene_converter import resolve_gene_for_model

            resolver = resolve_gene_for_model
        except Exception:  # noqa: BLE001
            resolver = None

    for g in genes:
        eid = _symbol_lookup_local(name_id, g)
        if eid is None and resolver is not None:
            # Ortholog / input-organism path: treat markers as panel-organism symbols.
            panel_species = dict(species)
            if panel_organism:
                panel_species["model_organism"] = panel_organism
            try:
                eid = resolver(g, panel_species)
            except Exception:  # noqa: BLE001
                eid = None
        if eid is None:
            continue
        tok = token_dict.get(eid)
        if tok is None:
            # Some tables store int keys; others strip versions inconsistently.
            tok = token_dict.get(str(eid))
        if tok is None:
            continue
        tok_i = int(tok)
        if tok_i in seen or tok_i in _PAD_LIKE:
            continue
        seen.add(tok_i)
        toks.append(tok_i)
    return toks


def _build_marker_tokens(
    markers: Mapping[str, Sequence[str]],
    token_dict: Mapping,
    name_id: Mapping,
    *,
    species: Mapping[str, Any] | None = None,
    panel_organism: str | None = None,
) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for ct, genes in markers.items():
        toks = _genes_to_tokens(
            genes,
            token_dict,
            name_id,
            species=species,
            panel_organism=panel_organism,
        )
        if not toks:
            logger.warning("No tokens resolved for cell-type markers: %s (%s)", ct, list(genes))
            continue
        if len(toks) < len(genes):
            resolved_eids = {
                _symbol_lookup_local(name_id, g) for g in genes
            }
            missing = [
                g
                for g in genes
                if _symbol_lookup_local(name_id, g) not in token_dict
                and (
                    _symbol_lookup_local(name_id, g) is None
                    or token_dict.get(_symbol_lookup_local(name_id, g)) is None
                )
            ]
            # Prefer a short missing list for logs; ortholog fallback may still have filled toks.
            if missing and len(toks) < len(genes):
                logger.info(
                    "Cell-type %s: %d/%d markers resolved (unresolved symbols: %s)",
                    ct,
                    len(toks),
                    len(genes),
                    missing,
                )
            elif resolved_eids:
                logger.info(
                    "Cell-type %s: %d/%d markers resolved to tokens",
                    ct,
                    len(toks),
                    len(genes),
                )
        out[ct] = toks
    return out


def _score_ids(ids: Sequence[int], ct_tokens: Sequence[int]) -> float:
    """Presence fraction (legacy)."""
    if not ct_tokens:
        return 0.0
    s = set(ids)
    return sum(1 for t in ct_tokens if t in s) / len(ct_tokens)


def _score_ids_rank_weighted(
    ids: Sequence[int],
    ct_tokens: Sequence[int],
    *,
    rank_power: float = 0.5,
) -> float:
    """Presence score with higher weight for earlier (higher-expressed) ranks.

    Geneformer rank-value encoding places the highest-expressed genes first in
    ``input_ids``. Weight ``1 / (rank+1)^rank_power`` boosts confident markers
    without ignoring mid-rank hits. Misses contribute 0; denominator is the
    unweighted panel size so scores stay comparable to presence fractions.
    """
    if not ct_tokens:
        return 0.0
    pos = {int(t): i for i, t in enumerate(ids) if int(t) not in _PAD_LIKE}
    total = 0.0
    for t in ct_tokens:
        rank = pos.get(int(t))
        if rank is None:
            continue
        total += 1.0 / ((rank + 1) ** float(rank_power))
    # Normalize so a full top-of-list hit ≈ 1.0 for small panels.
    ideal = sum(1.0 / ((i + 1) ** float(rank_power)) for i in range(len(ct_tokens)))
    if ideal <= 0:
        return 0.0
    # Mix presence (stability) with rank boost: scale by ideal so max ≈ 1.
    return float(min(1.0, total / ideal))


def _assign_pred(
    scores: Mapping[str, float],
    *,
    min_score: float = 0.25,
    min_margin: float = 0.05,
    min_markers_hit: int = 1,
    hit_counts: Mapping[str, int] | None = None,
) -> tuple[str, float]:
    if not scores:
        return "Unknown", 0.0
    best = max(scores, key=scores.get)
    best_score = float(scores[best])
    ranked = sorted(scores.values(), reverse=True)
    pred = best if best_score >= min_score else "Ambiguous"
    if (
        pred != "Ambiguous"
        and hit_counts is not None
        and int(hit_counts.get(best, 0)) < int(min_markers_hit)
        and best_score < 0.5
    ):
        pred = "Ambiguous"
    if pred != "Ambiguous" and len(ranked) >= 2 and (ranked[0] - ranked[1]) < min_margin and ranked[0] < 0.5:
        pred = "Ambiguous"
    return pred, best_score


def _assign_coarse(ids: Sequence[int], mic_tokens: Sequence[int], smc_tokens: Sequence[int]) -> str:
    mic = _score_ids(ids, mic_tokens)
    smc = _score_ids(ids, smc_tokens)
    if mic >= 0.3 and mic >= smc:
        return "Microglia-like"
    if smc >= 0.5:
        return "Vascular_SMC-like"
    if smc >= 0.3:
        return "SMC-intermediate"
    return "Other/Ambiguous"


def _hit_count(ids: Sequence[int], ct_tokens: Sequence[int]) -> int:
    s = set(int(x) for x in ids)
    return sum(1 for t in ct_tokens if int(t) in s)


def predict_cell_types_from_input_ids(
    input_ids_list: Sequence[Sequence[int]],
    markers: Mapping[str, Sequence[str]] | None = None,
    token_dict: Mapping | None = None,
    name_id: Mapping | None = None,
    *,
    species: Mapping[str, Any] | None = None,
    organism: str | None = None,
    use_rank_weights: bool = True,
    rank_power: float = 0.5,
    min_score: float = 0.25,
    min_margin: float = 0.05,
    min_markers_hit: int = 1,
) -> pd.DataFrame:
    """Return one row per cell with score_*, pred_cell_type, pred_score, coarse_type."""
    native = organism
    if token_dict is None or name_id is None:
        token_dict, name_id, native = load_model_dictionaries(species)
    elif native is None and species is not None:
        try:
            from geneformer.backends.registry import get_backend, parse_species_config

            native = get_backend(parse_species_config(species)).native_organism.value
        except Exception:  # noqa: BLE001
            native = "mouse"
    native = native or "mouse"

    markers_resolved = select_marker_panel(native, markers)
    marker_tokens = _build_marker_tokens(
        markers_resolved,
        token_dict,
        name_id,
        species=species,
        panel_organism=native,
    )
    mic_genes, smc_genes = _coarse_marker_genes(native)
    mic_tokens = _genes_to_tokens(
        mic_genes, token_dict, name_id, species=species, panel_organism=native
    )
    smc_tokens = _genes_to_tokens(
        smc_genes, token_dict, name_id, species=species, panel_organism=native
    )

    score_fn = (
        (lambda ids, toks: _score_ids_rank_weighted(ids, toks, rank_power=rank_power))
        if use_rank_weights
        else _score_ids
    )

    rows: list[dict[str, Any]] = []
    for ids in input_ids_list:
        scores = {ct: score_fn(ids, toks) for ct, toks in marker_tokens.items()}
        hits = {ct: _hit_count(ids, toks) for ct, toks in marker_tokens.items()}
        pred, pred_score = _assign_pred(
            scores,
            min_score=min_score,
            min_margin=min_margin,
            min_markers_hit=min_markers_hit,
            hit_counts=hits,
        )
        row = {
            "pred_cell_type": pred,
            "pred_score": pred_score,
            "coarse_type": _assign_coarse(ids, mic_tokens, smc_tokens),
            "celltype_source": "markers",
        }
        for ct, sc in scores.items():
            row[f"score_{ct}"] = sc
            row[f"hits_{ct}"] = hits.get(ct, 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(
        "Cell-type prediction (organism=%s, rank_weighted=%s): %s",
        native,
        use_rank_weights,
        df["pred_cell_type"].value_counts().to_dict() if len(df) else {},
    )
    logger.info(
        "Coarse programs: %s",
        df["coarse_type"].value_counts().to_dict() if len(df) else {},
    )
    return df


def apply_metadata_cell_types(
    df: pd.DataFrame,
    *,
    prefer_metadata: bool = True,
    metadata_columns: Sequence[str] = METADATA_CELLTYPE_COLUMNS,
) -> pd.DataFrame:
    """When a metadata cell-type column exists, prefer it over marker predictions.

    Writes ``celltype_plot`` (always) and, when preferred, overwrites
    ``pred_cell_type`` / ``coarse_type`` for non-empty metadata labels while
    recording ``celltype_source`` as ``metadata`` or ``markers``.
    """
    out = df.copy()
    meta_col = next((c for c in metadata_columns if c in out.columns), None)
    if meta_col is None:
        if "pred_cell_type" in out.columns:
            out["celltype_plot"] = out["pred_cell_type"]
        elif "coarse_type" in out.columns:
            out["celltype_plot"] = out["coarse_type"]
        else:
            out["celltype_plot"] = "Unknown"
        if "celltype_source" not in out.columns:
            out["celltype_source"] = "markers" if "pred_cell_type" in out.columns else "none"
        return out

    meta = out[meta_col].astype(str).str.strip()
    empty = meta.isna() | meta.isin(("", "nan", "None", "NA", "unknown", "Unknown"))
    if "pred_cell_type" not in out.columns:
        out["pred_cell_type"] = "Unknown"
        out["pred_score"] = 0.0
    if "coarse_type" not in out.columns:
        out["coarse_type"] = "Other/Ambiguous"
    if "celltype_source" not in out.columns:
        out["celltype_source"] = "markers"

    if prefer_metadata:
        use_meta = ~empty
        out.loc[use_meta, "pred_cell_type"] = meta[use_meta]
        out.loc[use_meta, "pred_score"] = 1.0
        out.loc[use_meta, "coarse_type"] = meta[use_meta]
        out.loc[use_meta, "celltype_source"] = "metadata"
        n = int(use_meta.sum())
        if n:
            logger.info(
                "Using metadata column %r for %d/%d cells (prefer_metadata=True)",
                meta_col,
                n,
                len(out),
            )

    out["celltype_plot"] = out["pred_cell_type"].astype(str)
    return out


def annotate_dataframe_with_cell_types(
    df: pd.DataFrame,
    input_ids_list: Sequence[Sequence[int]],
    markers: Mapping[str, Sequence[str]] | None = None,
    token_dict: Mapping | None = None,
    name_id: Mapping | None = None,
    *,
    species: Mapping[str, Any] | None = None,
    organism: str | None = None,
    use_rank_weights: bool = True,
    prefer_metadata: bool = True,
    min_score: float = 0.25,
    min_margin: float = 0.05,
    min_markers_hit: int = 1,
) -> pd.DataFrame:
    """Append prediction columns onto an existing per-cell table."""
    if len(df) != len(input_ids_list):
        raise ValueError(
            f"Row count mismatch: dataframe has {len(df)} rows, "
            f"input_ids has {len(input_ids_list)}"
        )
    pred = predict_cell_types_from_input_ids(
        input_ids_list,
        markers=markers,
        token_dict=token_dict,
        name_id=name_id,
        species=species,
        organism=organism,
        use_rank_weights=use_rank_weights,
        min_score=min_score,
        min_margin=min_margin,
        min_markers_hit=min_markers_hit,
    )
    out = df.copy()
    drop_cols = [
        c
        for c in out.columns
        if c
        in {
            "pred_cell_type",
            "pred_score",
            "coarse_type",
            "celltype_plot",
            "celltype_source",
        }
        or c.startswith("score_")
        or c.startswith("hits_")
    ]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    merged = pd.concat([out.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    return apply_metadata_cell_types(merged, prefer_metadata=prefer_metadata)
