"""Local brief gene notes for conversion-drop tables (no network)."""
from __future__ import annotations

import csv
import gzip
from functools import lru_cache
from pathlib import Path

DICTS_DIR = Path(__file__).resolve().parent / "dicts"
CURATED_TSV = DICTS_DIR / "gene_brief_function_curated.tsv"
NCBI_TSV_GZ = DICTS_DIR / "gene_brief_function.tsv.gz"

DROP_REASON_NOTES = {
    "ortholog_one2many": "No unique 1:1 ortholog (one-to-many). Dropped by one2one.",
    "ortholog_many2many": "Many-to-many ortholog. Dropped by one2one.",
    "no_ortholog": "No ortholog in the conversion table.",
    "dropped_ambiguous": "Ambiguous mapping; dropped by the active ortholog policy.",
    "dropped_ortholog_one2many": "No unique 1:1 ortholog (one-to-many). Dropped by one2one.",
    "dropped_ortholog_many2many": "Many-to-many ortholog. Dropped by one2one.",
}


def explain_drop_reason(status: str | None) -> str:
    raw = str(status or "").strip()
    if not raw:
        return "Unmapped during ortholog conversion."
    if raw in DROP_REASON_NOTES:
        return DROP_REASON_NOTES[raw]
    key = raw.removeprefix("dropped_")
    if key in DROP_REASON_NOTES:
        return DROP_REASON_NOTES[key]
    return raw.replace("_", " ")


def _read_tsv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict[str, str]] = []
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


@lru_cache(maxsize=1)
def _indexes() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Return (function_by_id, function_by_symbol_upper, symbol_by_id)."""
    by_id: dict[str, str] = {}
    by_symbol: dict[str, str] = {}
    symbol_by_id: dict[str, str] = {}

    def _put(gene_id: str, symbol: str, text: str, *, overwrite: bool) -> None:
        if gene_id and symbol:
            if overwrite or gene_id not in symbol_by_id:
                symbol_by_id[gene_id] = symbol
        if not text:
            return
        if gene_id:
            if overwrite or gene_id not in by_id:
                by_id[gene_id] = text
        if symbol:
            key = symbol.upper()
            if overwrite or key not in by_symbol:
                by_symbol[key] = text

    for row in _read_tsv_rows(NCBI_TSV_GZ):
        text = row.get("brief_function") or row.get("name") or ""
        _put(row.get("gene_id", ""), row.get("symbol", ""), text, overwrite=False)
    for row in _read_tsv_rows(CURATED_TSV):
        text = row.get("brief_function") or row.get("name") or ""
        _put(row.get("gene_id", ""), row.get("symbol", ""), text, overwrite=True)
    return by_id, by_symbol, symbol_by_id


def lookup_brief_function(*keys: str | None) -> str:
    """Best-effort one-line function / official name from local tables."""
    by_id, by_symbol, _symbol_by_id = _indexes()
    for raw in keys:
        token = str(raw or "").strip()
        if not token:
            continue
        if token in by_id:
            return by_id[token]
        if token.upper() in by_symbol:
            return by_symbol[token.upper()]
    return ""


def lookup_symbol(*keys: str | None) -> str:
    """Ensembl / FlyBase ID → gene symbol when present in local tables."""
    _by_id, _by_symbol, symbol_by_id = _indexes()
    for raw in keys:
        token = str(raw or "").strip()
        if token and token in symbol_by_id:
            return symbol_by_id[token]
    return ""


def invert_symbol_table(symbol_table: dict[str, str] | None) -> dict[str, str]:
    """Ensembl/FBgn → a representative symbol (first wins)."""
    out: dict[str, str] = {}
    if not symbol_table:
        return out
    for symbol, gene_id in symbol_table.items():
        gid = str(gene_id or "").strip()
        if gid and gid not in out:
            out[gid] = str(symbol).strip()
    return out
