"""
WebUI helpers: show genes dropped by ortholog conversion.

Reads tokenize artifacts on disk (no job execution). Brief function notes come
from local TSVs via geneformer.gene_brief_function.
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
_CORE = ROOT / "core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))


def _load_brief_mod():
    import importlib.util

    path = _CORE / "geneformer" / "gene_brief_function.py"
    spec = importlib.util.spec_from_file_location("gene_brief_function_ui", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_brief = _load_brief_mod()
explain_drop_reason = _brief.explain_drop_reason
lookup_brief_function = _brief.lookup_brief_function
lookup_symbol = _brief.lookup_symbol

_REPORT_NAMES = (
    "conversion_report.json",
    "conversion_unmapped_genes.tsv",
    "conversion_gene_detail.tsv",
)


@dataclass
class DroppedGeneTable:
    report_json: Path | None
    unmapped_tsv: Path | None
    detail_tsv: Path | None
    conversion_pair: str | None
    ortholog_policy: str | None
    input_genes: int
    mapped: int
    unmapped: int
    mapped_pct: float
    rows: list[dict[str, str]]


def find_conversion_artifact_dir(search_roots: Sequence[str | Path | None]) -> Path | None:
    """Newest directory that contains a conversion report JSON or unmapped TSV."""
    candidates: list[Path] = []
    for root in search_roots:
        if not root:
            continue
        base = Path(str(root)).expanduser()
        if not base.exists():
            continue
        if base.is_file() and base.name in _REPORT_NAMES:
            candidates.append(base.parent)
            continue
        if not base.is_dir():
            continue
        for name in _REPORT_NAMES:
            candidates.extend(p.parent for p in base.rglob(name))
    if not candidates:
        return None
    unique = {p.resolve() for p in candidates}
    return max(unique, key=lambda p: p.stat().st_mtime)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f, delimiter="\t")]


def _load_dropped_table(art_dir: Path) -> DroppedGeneTable:
    report_json = art_dir / "conversion_report.json"
    unmapped_tsv = art_dir / "conversion_unmapped_genes.tsv"
    detail_tsv = art_dir / "conversion_gene_detail.tsv"
    report = _read_json(report_json) if report_json.is_file() else {}
    unmapped_rows = _read_tsv(unmapped_tsv) if unmapped_tsv.is_file() else []
    detail_rows = _read_tsv(detail_tsv) if detail_tsv.is_file() else []

    detail_by_id: dict[str, dict[str, str]] = {}
    for row in detail_rows:
        status = row.get("status") or ""
        if status.startswith("mapped") or status in ("curated_bridge", "mapped_native"):
            continue
        for key in (row.get("source_id"), row.get("input_id")):
            if key:
                detail_by_id[key] = row

    ids: list[str] = []
    seen: set[str] = set()
    for row in unmapped_rows:
        gid = row.get("source_id") or row.get("input_id") or ""
        if gid and gid not in seen:
            seen.add(gid)
            ids.append(gid)
    if not ids:
        for gid in report.get("unmapped_gene_ids") or []:
            token = str(gid).strip()
            if token and token not in seen:
                seen.add(token)
                ids.append(token)
    if not ids:
        ids = list(detail_by_id.keys())

    unmapped_by_id = {
        (row.get("source_id") or row.get("input_id") or ""): row for row in unmapped_rows
    }

    table_rows: list[dict[str, str]] = []
    for gid in ids:
        raw = unmapped_by_id.get(gid) or {}
        detail = detail_by_id.get(gid) or {}
        symbol = raw.get("symbol") or lookup_symbol(gid)
        if symbol == gid:
            symbol = lookup_symbol(gid) or ""
        status = detail.get("status") or ""
        drop_reason = raw.get("drop_reason") or explain_drop_reason(status)
        function = raw.get("brief_function") or lookup_brief_function(
            gid, symbol, detail.get("input_id")
        )
        if not function:
            function = "(no local note)"
        table_rows.append(
            {
                "Gene ID": gid,
                "Symbol": symbol,
                "Why dropped": drop_reason,
                "Brief function": function,
            }
        )

    unmapped_count = int(report.get("unmapped") or len(table_rows) or 0)
    return DroppedGeneTable(
        report_json=report_json if report_json.is_file() else None,
        unmapped_tsv=unmapped_tsv if unmapped_tsv.is_file() else None,
        detail_tsv=detail_tsv if detail_tsv.is_file() else None,
        conversion_pair=report.get("conversion_pair"),
        ortholog_policy=report.get("ortholog_policy"),
        input_genes=int(report.get("input_genes") or 0),
        mapped=int(report.get("mapped") or 0),
        unmapped=unmapped_count,
        mapped_pct=float(report.get("mapped_pct") or 0.0),
        rows=table_rows,
    )


def load_dropped_gene_table(
    search_roots: Sequence[str | Path | None],
) -> DroppedGeneTable | None:
    art_dir = find_conversion_artifact_dir(search_roots)
    if art_dir is None:
        return None
    return _load_dropped_table(art_dir)
