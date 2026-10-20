"""Offline over-representation analysis (ORA) on local gene set libraries.

Gene lists stay on this machine: libraries are read from GMT files under
``core/geneformer/dicts/gene_sets`` (``scripts/download_gene_sets.sh``), and the test is
computed here. The statistics follow Enrichr with a user background:

- term genes are restricted to the background;
- ``P-value`` is the one-sided Fisher exact test (hypergeometric upper tail) of the overlap;
- ``Adjusted P-value`` is Benjamini-Hochberg over the terms with at least one overlapping gene.

Symbols are compared upper-cased (the libraries use human-style uppercase symbols; mouse
symbols are matched after upper-casing, as Enrichr does).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parent.parent
RESULT_COLUMNS = [
    "Term", "Overlap", "P-value", "Adjusted P-value", "Odds Ratio",
    "n_overlap", "n_term", "n_query", "n_background", "Genes",
]


def gene_sets_dir() -> Path:
    base = Path(os.environ.get("GENEFORMER_ROOT") or ROOT)
    return base / "core" / "geneformer" / "dicts" / "gene_sets"


def library_manifest(directory: Path | None = None) -> dict:
    d = directory or gene_sets_dir()
    path = d / "libraries.json"
    if not path.exists():
        path = ROOT / "core" / "geneformer" / "dicts" / "gene_sets" / "libraries.json"
    return json.loads(path.read_text())


def default_libraries(species: str, directory: Path | None = None) -> list[str]:
    manifest = library_manifest(directory)
    key = "mouse" if str(species).lower().startswith("mouse") else "human"
    return list(manifest["default_by_species"][key])


def library_label(name: str, directory: Path | None = None) -> str:
    info = library_manifest(directory).get("libraries", {}).get(name, {})
    return str(info.get("label") or name.replace("_", " "))


def read_gmt(path: Path) -> dict[str, set[str]]:
    """``{term: uppercase gene set}``; Enrichr GMTs have an empty description column."""
    out: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or not parts[0]:
                continue
            genes = {g.split(",")[0].strip().upper() for g in parts[2:] if g.strip()}
            if genes:
                out[parts[0]] = out.get(parts[0], set()) | genes
    return out


def load_library(name: str, directory: Path | None = None) -> dict[str, set[str]]:
    path = (directory or gene_sets_dir()) / f"{name}.gmt"
    if not path.exists():
        raise FileNotFoundError(
            f"Gene set library not found: {path}. Run scripts/download_gene_sets.sh "
            "(downloads by library name only; enrichment runs locally)."
        )
    return read_gmt(path)


def bh_adjust(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def ora(
    query: Iterable[str],
    background: Iterable[str],
    gene_sets: Mapping[str, set[str]],
    *,
    min_term_size: int = 1,
) -> pd.DataFrame:
    """ORA of ``query`` against ``gene_sets`` within ``background``.

    Query genes outside the background are dropped. Rows are terms with at least one
    overlapping gene, sorted by ``P-value``.
    """
    bg = {str(g).upper() for g in background if str(g).strip()}
    q = {str(g).upper() for g in query if str(g).strip()} & bg
    n_bg, n_q = len(bg), len(q)
    rows = []
    for term, genes in gene_sets.items():
        term_bg = genes & bg
        k_term = len(term_bg)
        if k_term < int(min_term_size):
            continue
        hit = q & term_bg
        k = len(hit)
        if k == 0:
            continue
        p = float(hypergeom.sf(k - 1, n_bg, k_term, n_q))
        b, c = n_q - k, k_term - k
        d = n_bg - n_q - c
        odds = float("inf") if b * c == 0 else (k * d) / (b * c)
        rows.append({
            "Term": term,
            "Overlap": f"{k}/{k_term}",
            "P-value": p,
            "Odds Ratio": odds,
            "n_overlap": k,
            "n_term": k_term,
            "n_query": n_q,
            "n_background": n_bg,
            "Genes": ";".join(sorted(hit)),
        })
    df = pd.DataFrame(rows, columns=[c for c in RESULT_COLUMNS if c != "Adjusted P-value"])
    df["Adjusted P-value"] = bh_adjust(df["P-value"].to_numpy()) if len(df) else []
    return df[RESULT_COLUMNS].sort_values(["P-value", "Term"]).reset_index(drop=True)
