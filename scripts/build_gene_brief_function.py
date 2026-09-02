#!/usr/bin/env python3
"""Build gene_brief_function.tsv.gz from NCBI gene_info (official full names).

Usage (network required):

    python3 scripts/build_gene_brief_function.py
"""
from __future__ import annotations

import csv
import gzip
import io
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "core" / "geneformer" / "dicts" / "gene_brief_function.tsv.gz"

NCBI = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO"
SOURCES = (
    f"{NCBI}/Mammalia/Homo_sapiens.gene_info.gz",
    f"{NCBI}/Mammalia/Mus_musculus.gene_info.gz",
    f"{NCBI}/Invertebrates/Drosophila_melanogaster.gene_info.gz",
)


def _parse_xrefs(dbxrefs: str) -> list[str]:
    ids: list[str] = []
    for part in (dbxrefs or "").split("|"):
        if ":" not in part:
            continue
        db, value = part.split(":", 1)
        db_l = db.lower()
        value = value.strip()
        if db_l == "ensembl" and (
            value.startswith("ENSG") or value.startswith("ENSMUSG")
        ):
            ids.append(value.split(".")[0])
        elif db_l == "flybase" and value.startswith("FBgn"):
            ids.append(value)
    return ids


def _iter_rows(url: str) -> list[tuple[str, str, str]]:
    print(f"Downloading {url} …")
    with urllib.request.urlopen(url, timeout=180) as resp:
        raw = gzip.decompress(resp.read())
    text = io.StringIO(raw.decode("utf-8", errors="replace"))
    reader = csv.DictReader(text, delimiter="\t")
    out: list[tuple[str, str, str]] = []
    for row in reader:
        symbol = (row.get("Symbol") or "").strip()
        name = (row.get("description") or "").strip()
        if not symbol or not name:
            continue
        gene_type = (row.get("type_of_gene") or "").strip()
        if gene_type and gene_type not in ("protein-coding", "pseudo", "ncRNA"):
            continue
        xrefs = _parse_xrefs(row.get("dbXrefs") or "")
        if not xrefs:
            continue
        for gid in xrefs:
            out.append((gid, symbol, name))
    return out


def main() -> None:
    seen: set[tuple[str, str]] = set()
    rows: list[tuple[str, str, str]] = []
    for url in SOURCES:
        for gid, symbol, name in _iter_rows(url):
            key = (gid, symbol.upper())
            if key in seen:
                continue
            seen.add(key)
            rows.append((gid, symbol, name))
    rows.sort(key=lambda r: (r[0], r[1]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["gene_id", "symbol", "brief_function"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows → {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
