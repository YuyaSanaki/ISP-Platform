"""Pre-ISP cell-type annotation from the expression matrix (AnnData).

Scores curated marker panels with ``scanpy.tl.score_genes`` on a normalized copy
of the count matrix, then writes:

- ``cell_type`` — predicted label (or Ambiguous / Doublet_suspected / Unknown)
- ``tissue`` — coarse tissue compartment
- ``celltype_score`` — winning panel score
- ``celltype_margin`` — best minus runner-up score
- ``celltype_core_hits`` — core markers detected in the winning type
- ``celltype_runner_up`` — next label, or ``A|B`` when a doublet is called
- ``celltype_doublet_flag`` — ``1`` when two compartments both clear the gate
- ``celltype_state`` — lineage state such as ``fiber_program=fast|slow|mixed|unresolved``
- ``celltype_confidence`` — ``low`` when fewer than half of that type's core genes are in the model vocabulary
- ``celltype_annotator`` — ``isp_expression_v2`` (platform provenance)

Default panel is ``isp_expression_v2`` (core / support / anti). Flat v1 lists
still score. Token-rank brain marker panels were removed.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ANNOTATOR_ID = "isp_expression_v2"
_PANEL_PATH = (
    Path(__file__).resolve().parent / "geneformer" / "dicts" / "celltype_panels" / "isp_expression_v2.json"
)

# Columns written onto AnnData.obs / loom / HF dataset.
ANNOTATION_OBS_COLUMNS = (
    "cell_type",
    "tissue",
    "celltype_score",
    "celltype_margin",
    "celltype_core_hits",
    "celltype_runner_up",
    "celltype_doublet_flag",
    "celltype_state",
    "celltype_confidence",
    "celltype_annotator",
)


def default_panel_path() -> Path:
    return _PANEL_PATH


def load_panel(panel_path: Path | str | None = None) -> dict[str, Any]:
    path = Path(panel_path) if panel_path else _PANEL_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Cell-type panel not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _gene_symbols(adata: ad.AnnData) -> pd.Index:
    """Prefer symbol-like var names; fall back to a symbol column if present."""
    for col in ("gene_symbols", "gene_symbol", "symbol", "Gene", "gene_name"):
        if col in adata.var.columns:
            return pd.Index(adata.var[col].astype(str))
    return pd.Index(adata.var_names.astype(str))


def _resolve_markers(
    markers: list[str],
    gene_index: pd.Index,
) -> list[str]:
    """Case-tolerant intersection of panel markers with adata genes."""
    lookup = {g: g for g in gene_index}
    for g in list(gene_index):
        lookup.setdefault(g.upper(), g)
        lookup.setdefault(g.lower(), g)
        if len(g) > 1:
            lookup.setdefault(g[0].upper() + g[1:].lower(), g)
    out: list[str] = []
    seen: set[str] = set()
    for m in markers:
        hit = lookup.get(m) or lookup.get(m.upper()) or lookup.get(m.lower())
        if hit is None and len(m) > 1:
            hit = lookup.get(m[0].upper() + m[1:].lower())
        if hit is not None and hit not in seen:
            seen.add(hit)
            out.append(hit)
    return out


def _prepare_scoring_adata(adata: ad.AnnData) -> ad.AnnData:
    """Normalized copy for score_genes (does not modify the raw-count object)."""
    import scanpy as sc

    if adata.n_obs == 0 or adata.n_vars == 0:
        return adata.copy()
    scor = adata.copy()
    symbols = _gene_symbols(adata).astype(str)
    # score_genes matches on var_names; keep symbols unique.
    scor.var_names = symbols
    scor.var_names_make_unique()
    sc.pp.normalize_total(scor, target_sum=1e4)
    sc.pp.log1p(scor)
    return scor


def _simple_marker_score(scor: ad.AnnData, resolved: list[str], col: str) -> None:
    """Mean marker expression minus mean of remaining genes (log-normalized copy)."""
    X = scor[:, resolved].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    marker_mean = np.asarray(X, dtype=float).mean(axis=1)
    rest = [g for g in scor.var_names.astype(str) if g not in set(resolved)]
    if rest:
        R = scor[:, rest].X
        if hasattr(R, "toarray"):
            R = R.toarray()
        rest_mean = np.asarray(R, dtype=float).mean(axis=1)
    else:
        rest_mean = np.zeros(scor.n_obs, dtype=float)
    scor.obs[col] = marker_mean - rest_mean


def _score_gene_list(scor: ad.AnnData, resolved: list[str], col: str) -> None:
    """``score_genes`` with fallbacks for tiny matrices (no control-gene bins)."""
    import scanpy as sc

    ctrl_size = max(1, min(50, int(scor.n_vars) - len(resolved)))
    attempts = (
        dict(use_raw=False, ctrl_size=ctrl_size),
        dict(use_raw=False, ctrl_as_ref=False, ctrl_size=ctrl_size),
    )
    for kwargs in attempts:
        try:
            sc.tl.score_genes(scor, gene_list=resolved, score_name=col, **kwargs)
            return
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.info("score_genes failed for %s (%s); trying fallback", col, exc)
    logger.info("Using mean-difference fallback for %s", col)
    _simple_marker_score(scor, resolved, col)


def _marker_block(spec: Mapping[str, Any], organism: str) -> dict[str, list[str]]:
    """Normalize a v1 gene list or a v2 ``{core, support, anti}`` block."""
    raw = spec.get(organism)
    if raw is None:
        raw = spec.get("mouse")
    if isinstance(raw, list):
        return {"core": [str(g) for g in raw], "support": [], "anti": []}
    if isinstance(raw, dict):
        anti = raw.get("anti") or raw.get("anti_markers") or []
        return {
            "core": [str(g) for g in (raw.get("core") or [])],
            "support": [str(g) for g in (raw.get("support") or [])],
            "anti": [str(g) for g in anti],
        }
    return {"core": [], "support": [], "anti": []}


def _detected_counts(scor: ad.AnnData, genes: list[str]) -> np.ndarray:
    """How many ``genes`` are non-zero in each cell of the scoring matrix."""
    if not genes:
        return np.zeros(scor.n_obs, dtype=int)
    X = scor[:, genes].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return (np.asarray(X, dtype=float) > 0).sum(axis=1).astype(int)


_DICTS = Path(__file__).resolve().parent / "geneformer" / "dicts"
_MODEL_VOCAB = {
    "mouse": (
        _DICTS / "mouse" / "MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl",
        _DICTS / "mouse" / "MLM-re_token_dictionary_v1.pkl",
    ),
    "human": (
        _DICTS / "human" / "gene_name_id_dict_gc104M.pkl",
        _DICTS / "human" / "token_dictionary_gc104M.pkl",
    ),
}
_CONTINUUM_GROUPS = (
    ("mural_program", frozenset({"Pericyte", "Vascular_SMC", "Fibroblast", "Myofibroblast"})),
    (
        "myeloid_program",
        frozenset(
            {
                "Monocyte",
                "Macrophage",
                "C1QC_APOE_macrophage",
                "cDC1",
                "cDC2",
                "pDC",
                "Neutrophil",
                "Microglia",
            }
        ),
    ),
)


def vocab_fraction_is_low(tokenized: int, registered: int) -> bool:
    """True when at least half of the registered markers are outside the vocabulary."""
    return registered > 0 and tokenized * 2 <= registered


@lru_cache(maxsize=4)
def _symbols_in_model_vocab(organism: str) -> frozenset[str] | None:
    """Symbols whose Ensembl id is in the Geneformer token dictionary."""
    paths = _MODEL_VOCAB.get(organism)
    if paths is None or not paths[0].is_file() or not paths[1].is_file():
        return None
    with open(paths[0], "rb") as handle:
        symbols = pickle.load(handle)
    with open(paths[1], "rb") as handle:
        tokens = pickle.load(handle)
    keep: set[str] = set()
    for name, ensembl in symbols.items():
        ens = str(ensembl).split(".")[0]
        if ens in tokens or str(ensembl) in tokens:
            keep.add(str(name))
    return frozenset(keep)


def _symbol_in_vocab(gene: str, vocab: frozenset[str]) -> bool:
    if gene in vocab:
        return True
    candidates = {gene.upper(), gene.lower()}
    if len(gene) > 1:
        candidates.add(gene[0].upper() + gene[1:].lower())
    return any(cand in vocab for cand in candidates)


def _log_core_vocab(
    cell_type: str,
    core_genes: list[str],
    *,
    vocab: frozenset[str] | None,
    resolved_in_input: int,
) -> bool:
    """Log ``tokenized_marker_count/registered_marker_count``. Return True if low-confidence."""
    registered = len(core_genes)
    if vocab is not None:
        tokenized = sum(_symbol_in_vocab(gene, vocab) for gene in core_genes)
        source = "model"
    else:
        tokenized = resolved_in_input
        source = "input"
    low = vocab_fraction_is_low(tokenized, registered)
    message = (
        f"  celltype_vocab: {cell_type} tokenized_marker_count={tokenized}/{registered} ({source})"
    )
    print(message, flush=True)
    if low:
        logger.warning(
            "%s core markers are mostly outside the %s vocabulary (%d/%d); low confidence",
            cell_type,
            source,
            tokenized,
            registered,
        )
    return low


def _lineage_positive_boost(
    scor: ad.AnnData,
    spec: Mapping[str, Any] | None,
    organism: str,
    gene_index: pd.Index,
    vocab: frozenset[str] | None,
    low_confidence_types: set[str],
) -> np.ndarray | None:
    """Positive epithelial-gate score. A low or negative score is not a penalty."""
    if not isinstance(spec, Mapping):
        return None
    block = _marker_block(spec, organism)
    if _log_core_vocab(
        "Epithelial_general",
        block["core"],
        vocab=vocab,
        resolved_in_input=len(_resolve_markers(block["core"], gene_index)),
    ):
        low_confidence_types.add("Epithelial_general")
    core = _resolve_markers(block["core"], gene_index)
    if len(core) < 2:
        return None
    col = "_isp_core_Epithelial_general"
    _score_gene_list(scor, core, col)
    return np.clip(scor.obs[col].to_numpy(dtype=float), 0, None)


_NEURAL_PRIOR_COLS = ("organ_major", "organ", "tissue_prior", "sample_organ", "tissue")
_NEURAL_PRIOR_RE = re.compile(r"brain|neural|cns|cortex|hippocamp|spinal|cerebell", re.I)


def _neural_prior_mask(obs: pd.DataFrame) -> np.ndarray:
    """Per-cell neural/brain prior from sample metadata, when that column exists."""
    mask = np.zeros(len(obs), dtype=bool)
    for col in _NEURAL_PRIOR_COLS:
        if col not in obs.columns:
            continue
        mask |= obs[col].astype(str).str.contains(_NEURAL_PRIOR_RE, na=False).to_numpy()
    return mask


def _assign_fiber_state(
    labels: list[str],
    scor: ad.AnnData,
    states: Mapping[str, Any],
    organism: str,
    parent: str = "Skeletal_myocyte",
) -> list[str]:
    """Attach ``fiber_program`` only after the skeletal lineage label is set."""
    out = [""] * len(labels)
    fiber = states.get("fiber_program")
    if not isinstance(fiber, Mapping):
        return out
    need = int(fiber.get("min_detected", 2))
    hits: dict[str, np.ndarray] = {}
    for prog, spec in fiber.items():
        if prog == "min_detected" or not isinstance(spec, Mapping):
            continue
        genes = _resolve_markers(_marker_block(spec, organism)["core"], scor.var_names)
        hits[prog] = _detected_counts(scor, genes) if genes else np.zeros(len(labels), dtype=int)
    for i, lab in enumerate(labels):
        if lab != parent:
            continue
        on = [name for name, count in hits.items() if int(count[i]) >= need]
        if len(on) >= 2:
            out[i] = "fiber_program=mixed"
        elif len(on) == 1:
            out[i] = f"fiber_program={on[0]}"
        else:
            out[i] = "fiber_program=unresolved"
    return out


def _write_unknown(target: ad.AnnData, annotator_id: str) -> ad.AnnData:
    n = target.n_obs
    target.obs["cell_type"] = "Unknown"
    target.obs["tissue"] = "Other"
    target.obs["celltype_score"] = 0.0
    target.obs["celltype_margin"] = 0.0
    target.obs["celltype_core_hits"] = 0
    target.obs["celltype_runner_up"] = ""
    target.obs["celltype_doublet_flag"] = "0"
    target.obs["celltype_state"] = ""
    target.obs["celltype_confidence"] = "ok"
    target.obs["celltype_annotator"] = annotator_id
    if n == 0:
        return target
    return target


def annotate_adata_cell_types(
    adata: ad.AnnData,
    *,
    organism: str = "mouse",
    panel_path: Path | str | None = None,
    panel: Mapping[str, Any] | None = None,
    enabled: bool = True,
    min_score: float | None = None,
    min_margin: float | None = None,
    inplace: bool = True,
) -> ad.AnnData:
    """Annotate ``adata.obs`` with expression-matrix cell-type labels.

    Operates on a normalized copy for scoring; the returned object keeps the
    original ``X`` when ``inplace=True`` (only obs columns are added).

    v2 blocks (``core`` / ``support`` / ``anti``) require ``min_core_detected``
    core genes with nonzero expression. Two eligible types from different
    ``compartment`` values both clearing ``min_score`` become ``Doublet_suspected``.
    Types with ``assign: false`` are lineage gates and are not labeled.
    A flat v1 gene list is treated as ``core`` with a 1-gene hit floor.
    """
    if not enabled:
        return adata

    loaded = dict(panel) if panel is not None else load_panel(panel_path)
    org = (organism or "mouse").strip().lower()
    if org not in ("mouse", "human"):
        logger.warning("Unsupported organism %r for expression annotation; using mouse", organism)
        org = "mouse"

    annotator_id = str(loaded.get("version") or ANNOTATOR_ID)
    thr = float(loaded.get("min_score", 0.15) if min_score is None else min_score)
    margin_min = float(loaded.get("min_margin", 0.02) if min_margin is None else min_margin)
    default_hits = int(loaded.get("min_core_detected", 2))
    support_weight = float(loaded.get("support_weight", 0.35))
    anti_weight = float(loaded.get("anti_weight", 0.75))
    types: Mapping[str, Any] = loaded.get("types") or {}

    target = adata if inplace else adata.copy()
    neural_prior = _neural_prior_mask(target.obs)
    scor = _prepare_scoring_adata(target)
    gene_index = scor.var_names

    myeloid_hits = None
    myeloid_min = 2
    myeloid_gate = (loaded.get("gates") or {}).get("myeloid") or {}
    if isinstance(myeloid_gate, Mapping):
        myeloid_genes = _resolve_markers(list(myeloid_gate.get(org) or []), gene_index)
        myeloid_min = int(myeloid_gate.get("min_detected", 2))
        if len(myeloid_genes) >= 2:
            myeloid_hits = _detected_counts(scor, myeloid_genes)

    model_vocab = _symbols_in_model_vocab(org)
    low_confidence_types: set[str] = set()
    epithelial_boost = _lineage_positive_boost(
        scor,
        types.get("Epithelial_general") if isinstance(types.get("Epithelial_general"), Mapping) else None,
        org,
        gene_index,
        model_vocab,
        low_confidence_types,
    )

    type_names: list[str] = []
    tissues: list[str] = []
    compartments: list[str] = []
    score_rows: list[np.ndarray] = []
    hit_rows: list[np.ndarray] = []
    core_score_rows: list[np.ndarray] = []
    min_hit_rows: list[int] = []
    requires_flags: list[Mapping[str, Any]] = []
    state_by_parent: dict[str, Mapping[str, Any]] = {}

    for ct, spec in types.items():
        if not isinstance(spec, Mapping):
            continue
        if spec.get("assign", True) is False:
            logger.info("Expression annotator: %s is a lineage gate; not assigned", ct)
            continue
        if isinstance(spec.get("states"), Mapping):
            state_by_parent[ct] = spec["states"]
        block = _marker_block(spec, org)
        structured = isinstance(spec.get(org) or spec.get("mouse"), dict)
        # Per-type min_core_detected overrides the panel default. Flat v1 lists stay at 1.
        if "min_core_detected" in spec:
            min_hits = int(spec["min_core_detected"])
        elif structured:
            min_hits = int(default_hits)
        else:
            min_hits = 1
        if _log_core_vocab(
            ct,
            block["core"],
            vocab=model_vocab,
            resolved_in_input=len(_resolve_markers(block["core"], gene_index)),
        ):
            low_confidence_types.add(ct)
        core = _resolve_markers(block["core"], gene_index)
        support = _resolve_markers(block["support"], gene_index)
        anti = _resolve_markers(block["anti"], gene_index)
        if len(core) < 2:
            logger.info(
                "Expression annotator: %s — fewer than 2 core markers resolved (%s); skipping",
                ct,
                core,
            )
            continue

        core_col = f"_isp_core_{ct}"
        _score_gene_list(scor, core, core_col)
        score = scor.obs[core_col].to_numpy(dtype=float)
        if len(support) >= 2:
            sup_col = f"_isp_sup_{ct}"
            _score_gene_list(scor, support, sup_col)
            score = score + support_weight * np.clip(
                scor.obs[sup_col].to_numpy(dtype=float), 0, None
            )
        if len(anti) >= 2:
            anti_col = f"_isp_anti_{ct}"
            _score_gene_list(scor, anti, anti_col)
            score = score - anti_weight * np.clip(
                scor.obs[anti_col].to_numpy(dtype=float), 0, None
            )
        hits = _detected_counts(scor, core)
        core_only = scor.obs[core_col].to_numpy(dtype=float).copy()
        # Hit-count gate before any lineage boost. A negative epithelial score never subtracts.
        score = np.where(hits >= min_hits, score, -np.inf)
        if epithelial_boost is not None and str(spec.get("compartment") or "") == "epithelial":
            score = np.where(
                np.isfinite(score),
                score + support_weight * epithelial_boost,
                score,
            )
        req = spec.get("requires") if isinstance(spec.get("requires"), Mapping) else {}
        if req.get("myeloid_compartment"):
            # Missing myeloid genes fail the gate. They do not silently pass.
            gate_fail = (
                np.ones(scor.n_obs, dtype=bool)
                if myeloid_hits is None
                else myeloid_hits < myeloid_min
            )
            if req.get("microglia_core_rescue"):
                # P2RY12/TMEM119/SALL1/HEXB/GPR34 core can rescue microglia when PTPRC is low.
                score = np.where(gate_fail & ~(hits >= min_hits), -np.inf, score)
            else:
                score = np.where(gate_fail, -np.inf, score)
        type_names.append(ct)
        tissues.append(str(spec.get("tissue") or "Other"))
        compartments.append(str(spec.get("compartment") or spec.get("tissue") or "Other"))
        score_rows.append(score)
        hit_rows.append(hits)
        core_score_rows.append(core_only)
        min_hit_rows.append(min_hits)
        requires_flags.append(req)

    if not type_names:
        logger.warning("Expression annotator: no panels scored; labeling all Unknown")
        return _write_unknown(target, annotator_id)

    score_mat = np.vstack(score_rows).T
    hit_mat = np.vstack(hit_rows).T
    core_score_mat = np.vstack(core_score_rows).T
    name_to_i = {name: i for i, name in enumerate(type_names)}
    if "Microglia" in name_to_i:
        mi = name_to_i["Microglia"]
        # Direct core-hit / core-score gate, not "Microglia was not the top label".
        # High only when enough core genes are detected and their score clears the threshold.
        microglia_high = (hit_mat[:, mi] >= min_hit_rows[mi]) & (core_score_mat[:, mi] >= thr)
        for col, req in enumerate(requires_flags):
            if req.get("microglia_core_negative_or_low"):
                score_mat[:, col] = np.where(microglia_high, -np.inf, score_mat[:, col])
    n_types = score_mat.shape[1]
    finite = np.where(np.isfinite(score_mat), score_mat, -1e9)
    order = np.argsort(-finite, axis=1)
    best_idx = order[:, 0]
    second_idx = order[:, 1] if n_types >= 2 else np.zeros(score_mat.shape[0], dtype=int)
    row = np.arange(score_mat.shape[0])
    best_scores = score_mat[row, best_idx]
    second_scores = score_mat[row, second_idx] if n_types >= 2 else np.full(score_mat.shape[0], -np.inf)

    labels: list[str] = []
    tissue_out: list[str] = []
    margins: list[float] = []
    core_hits: list[int] = []
    runners: list[str] = []
    doublet_flags: list[str] = []
    for i in range(score_mat.shape[0]):
        sc_best = float(best_scores[i])
        sc_second = float(second_scores[i]) if np.isfinite(second_scores[i]) else float("-inf")
        bi = int(best_idx[i])
        si = int(second_idx[i])
        margin = sc_best - sc_second if np.isfinite(sc_second) else sc_best
        hit_best = int(hit_mat[i, bi])
        if not np.isfinite(sc_best) or sc_best < thr:
            labels.append("Ambiguous")
            tissue_out.append("Other")
            margins.append(0.0 if not np.isfinite(sc_best) else float(margin))
            core_hits.append(0 if not np.isfinite(sc_best) else hit_best)
            runners.append("" if not np.isfinite(sc_second) else type_names[si])
            doublet_flags.append("0")
            continue
        # Core scores, not anti-penalized totals: T and B both stay visible as a doublet
        # when each core is high. Shared markers (PECAM1 on LEC) stay below this bar.
        core_eligible = np.flatnonzero(
            (hit_mat[i] >= np.asarray(min_hit_rows))
            & np.isfinite(score_mat[i])
            & (core_score_mat[i] >= thr)
        )
        core_order = core_eligible[np.argsort(-core_score_mat[i, core_eligible])] if len(core_eligible) else []
        if len(core_order) >= 2:
            c0, c1 = int(core_order[0]), int(core_order[1])
            c_best = float(core_score_mat[i, c0])
            c_second = float(core_score_mat[i, c1])
            same_continuum = any(
                type_names[c0] in group and type_names[c1] in group
                for _, group in _CONTINUUM_GROUPS
            )
            if not same_continuum and c_best > 0 and c_second >= 0.6 * c_best:
                pair = f"{type_names[c0]}|{type_names[c1]}"
                labels.append("Doublet_suspected")
                tissue_out.append("Other")
                margins.append(c_best - c_second)
                core_hits.append(int(hit_mat[i, c0]))
                runners.append(pair)
                doublet_flags.append("1")
                continue
            if not same_continuum and c_best > 0 and c_second >= 0.5 * c_best:
                pair = f"{type_names[c0]}|{type_names[c1]}"
                labels.append("ambiguous_cross_compartment")
                tissue_out.append("Other")
                margins.append(c_best - c_second)
                core_hits.append(int(hit_mat[i, c0]))
                runners.append(pair)
                doublet_flags.append("0")
                continue
        # Same-continuum pairs (mural, myeloid) keep one label plus an alt state.
        same_continuum = any(
            type_names[bi] in group and type_names[si] in group
            for _, group in _CONTINUUM_GROUPS
        )
        competing = np.isfinite(sc_second) and sc_second >= thr and not same_continuum
        if competing and sc_second >= 0.6 * sc_best:
            pair = f"{type_names[bi]}|{type_names[si]}"
            labels.append("Doublet_suspected")
            tissue_out.append("Other")
            margins.append(float(margin))
            core_hits.append(hit_best)
            runners.append(pair)
            doublet_flags.append("1")
            continue
        if competing and sc_second >= 0.5 * sc_best:
            pair = f"{type_names[bi]}|{type_names[si]}"
            labels.append("ambiguous_cross_compartment")
            tissue_out.append("Other")
            margins.append(float(margin))
            core_hits.append(hit_best)
            runners.append(pair)
            doublet_flags.append("0")
            continue
        if np.isfinite(sc_second) and (sc_best - sc_second) < margin_min and sc_best < 0.5:
            labels.append("Ambiguous")
            tissue_out.append("Other")
            margins.append(float(margin))
            core_hits.append(hit_best)
            runners.append(type_names[si])
            doublet_flags.append("0")
            continue
        winner = type_names[bi]
        winner_tissue = tissues[bi]
        winner_runner = "" if not np.isfinite(sc_second) else type_names[si]
        if winner == "C1QC_APOE_macrophage" and bool(neural_prior[i]):
            mi = name_to_i.get("Microglia")
            if mi is not None and int(hit_mat[i, mi]) >= 1:
                winner = "Microglia"
                winner_tissue = tissues[mi]
                winner_runner = "C1QC_APOE_macrophage"
            else:
                winner = "Ambiguous"
                winner_tissue = "Other"
                winner_runner = "C1QC_APOE_macrophage"
        labels.append(winner)
        tissue_out.append(winner_tissue)
        margins.append(float(margin) if np.isfinite(margin) else 0.0)
        core_hits.append(hit_best)
        runners.append(winner_runner)
        doublet_flags.append("0")

    state_out = [""] * len(labels)
    for parent, states in state_by_parent.items():
        parent_state = _assign_fiber_state(labels, scor, states, org, parent=parent)
        for i, value in enumerate(parent_state):
            if value:
                state_out[i] = value
    for i, lab in enumerate(labels):
        if state_out[i]:
            continue
        alt = runners[i]
        for key, group in _CONTINUUM_GROUPS:
            if lab in group and alt in group:
                state_out[i] = f"{key}={lab};alt={alt}"
                break

    target.obs["cell_type"] = pd.Categorical(labels)
    target.obs["tissue"] = pd.Categorical(tissue_out)
    target.obs["celltype_score"] = np.where(np.isfinite(best_scores), best_scores, 0.0)
    target.obs["celltype_margin"] = margins
    target.obs["celltype_core_hits"] = core_hits
    target.obs["celltype_runner_up"] = runners
    target.obs["celltype_doublet_flag"] = doublet_flags
    target.obs["celltype_state"] = state_out
    target.obs["celltype_confidence"] = [
        "low" if lab in low_confidence_types else "ok" for lab in labels
    ]
    target.obs["celltype_annotator"] = annotator_id

    vc = pd.Series(labels).value_counts().to_dict()
    logger.info(
        "Expression cell-type annotation (%s, organism=%s): %s",
        annotator_id,
        org,
        vc,
    )
    print(
        f"  celltype_annotation: {annotator_id} organism={org} "
        f"labels={vc}",
        flush=True,
    )
    return target


def annotation_config_from_tokenizer(tokenizer_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """Read ``tokenizer.celltype_annotation`` block with defaults."""
    raw = (tokenizer_cfg or {}).get("celltype_annotation")
    if raw is False:
        return {"enabled": False}
    if not isinstance(raw, dict):
        raw = {}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "panel": raw.get("panel"),
        "min_score": raw.get("min_score"),
        "min_margin": raw.get("min_margin"),
    }


def is_platform_celltype_annotation(series: pd.Series | None) -> bool:
    """True when obs/dataset carries our expression annotator provenance."""
    if series is None:
        return False
    s = series.astype(str)
    return bool(s.str.startswith("isp_expression").fillna(False).any())


def organism_from_species(species: Mapping[str, Any] | str | None) -> str:
    """Pick mouse/human from a species config mapping or string."""
    if isinstance(species, Mapping):
        return str(species.get("model_organism") or "mouse")
    if isinstance(species, str) and species.strip():
        return species.strip()
    return "mouse"


def ensure_annotation_attrs(tokenizer_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Guarantee platform annotation columns are listed in custom_attr_name_dict."""
    cfg = tokenizer_cfg if isinstance(tokenizer_cfg, dict) else {}
    attr = dict(cfg.get("custom_attr_name_dict") or {})
    for col in ANNOTATION_OBS_COLUMNS:
        attr.setdefault(col, col)
    cfg["custom_attr_name_dict"] = attr
    return cfg


def apply_pre_isp_celltype_annotation(
    adata: ad.AnnData,
    tokenizer_cfg: Mapping[str, Any] | None,
    species: Mapping[str, Any] | str | None = None,
) -> ad.AnnData:
    """Annotate ``adata`` when ``tokenizer.celltype_annotation`` is enabled."""
    cfg = tokenizer_cfg if isinstance(tokenizer_cfg, dict) else {}
    ensure_annotation_attrs(cfg)
    ct_ann = annotation_config_from_tokenizer(cfg)
    if not ct_ann.get("enabled", True):
        return adata
    return annotate_adata_cell_types(
        adata,
        organism=organism_from_species(species),
        panel_path=ct_ann.get("panel"),
        enabled=True,
        min_score=ct_ann.get("min_score"),
        min_margin=ct_ann.get("min_margin"),
        inplace=True,
    )
