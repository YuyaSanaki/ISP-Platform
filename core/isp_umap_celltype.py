"""Marker-gene cell-type prediction for ISP UMAP start-state cells.

Scores each cell by the fraction of canonical marker tokens present in its
Geneformer ``input_ids`` (rank-value encoding). Same approach used in the
Igfbp2 cluster / co-expression analysis (Mouse-Geneformer-WebUI).
"""

from __future__ import annotations

import logging
import pickle
from typing import Any, Mapping, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# Fine-grained panels (Igfbp2 analysis final set; Igfbp2 excluded from choroid).
DEFAULT_MARKERS: dict[str, list[str]] = {
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

# Coarse vascular vs immune programs (for L2-by-group plots).
COARSE_MICROGLIA_MARKERS = [
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
COARSE_SMC_MARKERS = ["Acta2", "Myh11", "Tagln", "Cnn1", "Myl9", "Tpm2", "Lmod1"]


def _load_dictionaries() -> tuple[dict, dict]:
    from geneformer import tokenizer as gf_tokenizer
    from geneformer.in_silico_perturber_stats import GENE_NAME_ID_DICTIONARY_FILE

    with open(gf_tokenizer.TOKEN_DICTIONARY_FILE, "rb") as f:
        token_dict = pickle.load(f)  # ensembl -> token id
    with open(GENE_NAME_ID_DICTIONARY_FILE, "rb") as f:
        name_id = pickle.load(f)  # symbol -> ensembl
    return token_dict, name_id


def _genes_to_tokens(genes: Sequence[str], token_dict: Mapping, name_id: Mapping) -> list[int]:
    toks = []
    for g in genes:
        eid = name_id.get(g)
        if eid is None:
            continue
        tok = token_dict.get(eid)
        if tok is not None:
            toks.append(int(tok))
    return toks


def _build_marker_tokens(
    markers: Mapping[str, Sequence[str]],
    token_dict: Mapping,
    name_id: Mapping,
) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for ct, genes in markers.items():
        toks = _genes_to_tokens(genes, token_dict, name_id)
        if not toks:
            logger.warning("No tokens resolved for cell-type markers: %s (%s)", ct, list(genes))
            continue
        if len(toks) < len(genes):
            missing = [g for g in genes if name_id.get(g) not in token_dict]
            logger.info(
                "Cell-type %s: %d/%d markers resolved (missing: %s)",
                ct,
                len(toks),
                len(genes),
                missing,
            )
        out[ct] = toks
    return out


def _score_ids(ids: Sequence[int], ct_tokens: Sequence[int]) -> float:
    if not ct_tokens:
        return 0.0
    s = set(ids)
    return sum(1 for t in ct_tokens if t in s) / len(ct_tokens)


def _assign_pred(
    scores: Mapping[str, float], min_score: float = 0.25, min_margin: float = 0.05
) -> tuple[str, float]:
    if not scores:
        return "Unknown", 0.0
    best = max(scores, key=scores.get)
    best_score = float(scores[best])
    ranked = sorted(scores.values(), reverse=True)
    pred = best if best_score >= min_score else "Ambiguous"
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


def predict_cell_types_from_input_ids(
    input_ids_list: Sequence[Sequence[int]],
    markers: Mapping[str, Sequence[str]] | None = None,
    token_dict: Mapping | None = None,
    name_id: Mapping | None = None,
) -> pd.DataFrame:
    """Return one row per cell with score_*, pred_cell_type, pred_score, coarse_type."""
    if token_dict is None or name_id is None:
        token_dict, name_id = _load_dictionaries()
    markers = dict(markers or DEFAULT_MARKERS)
    marker_tokens = _build_marker_tokens(markers, token_dict, name_id)
    mic_tokens = _genes_to_tokens(COARSE_MICROGLIA_MARKERS, token_dict, name_id)
    smc_tokens = _genes_to_tokens(COARSE_SMC_MARKERS, token_dict, name_id)

    rows: list[dict[str, Any]] = []
    for ids in input_ids_list:
        scores = {ct: _score_ids(ids, toks) for ct, toks in marker_tokens.items()}
        pred, pred_score = _assign_pred(scores)
        row = {
            "pred_cell_type": pred,
            "pred_score": pred_score,
            "coarse_type": _assign_coarse(ids, mic_tokens, smc_tokens),
        }
        for ct, sc in scores.items():
            row[f"score_{ct}"] = sc
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(
        "Cell-type prediction: %s",
        df["pred_cell_type"].value_counts().to_dict() if len(df) else {},
    )
    logger.info(
        "Coarse programs: %s",
        df["coarse_type"].value_counts().to_dict() if len(df) else {},
    )
    return df


def annotate_dataframe_with_cell_types(
    df: pd.DataFrame,
    input_ids_list: Sequence[Sequence[int]],
    markers: Mapping[str, Sequence[str]] | None = None,
    token_dict: Mapping | None = None,
    name_id: Mapping | None = None,
) -> pd.DataFrame:
    """Append prediction columns onto an existing per-cell table."""
    if len(df) != len(input_ids_list):
        raise ValueError(
            f"Row count mismatch: dataframe has {len(df)} rows, "
            f"input_ids has {len(input_ids_list)}"
        )
    pred = predict_cell_types_from_input_ids(
        input_ids_list, markers=markers, token_dict=token_dict, name_id=name_id
    )
    out = df.copy()
    drop_cols = [
        c
        for c in out.columns
        if c in {"pred_cell_type", "pred_score", "coarse_type", "celltype_plot"}
        or c.startswith("score_")
    ]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return pd.concat([out.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
