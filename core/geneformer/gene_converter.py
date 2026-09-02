"""
Ortholog / gene-ID conversion for cross-species Geneformer runs (P0).

When species.model_organism differs from the native organism of species.model,
genes are mapped from the input species to the model vocabulary before
tokenization and ISP.

Resolution priority (shared by convert_gene_ids / remap_adata / resolve_gene_for_model):
  1. Ortholog table lookup on the user input (symbol or Ensembl, including curated).
  2. Input-organism symbol → Ensembl (symbol table) → ortholog lookup.
  3. Passthrough if already a model-native Ensembl / fly ID.

Loom remapping and ISP must use this same path so symbol vs Ensembl inputs cannot diverge.

Curated TSVs (*_curated.tsv) are merged after main tables and override Ensembl
pairs when biology or smoke-test genes need correction (e.g. Igfbp2).

Optional project overlays (analysis-scoped bridges) are applied *after* the
platform curated TSV when explicitly selected via:
  - load_ortholog_table(..., curated_overlay=...)
  - species.ortholog_curated_overlay (file or directory)
  - env GENEFORMER_ORTHOLOG_CURATED_OVERLAY (file or directory)
  - CLI --ortholog-curated-overlay (propagated into species / env)

Directory overlays resolve ``curated_bridge_{pair}.tsv``
(e.g. curated_bridge_human_to_mouse.tsv). Overlays do not modify the global
*_curated.tsv files under core/geneformer/dicts/orthologs/.

Many-to-many handling is controlled by species.ortholog_policy:
  - one2one (default): keep reciprocal 1:1 pairs only; never sum expression
  - best_of_n: N→1 keeps the highest-median source gene (no sum)
  - legacy_sum: last-write-wins table collapse + expression sum (old behavior)
"""
from __future__ import annotations

import csv
import logging
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

from .backends.registry import (
    BackendSpec,
    ModelId,
    Organism,
    OrthologPolicy,
    get_backend,
    needs_gene_conversion,
    parse_species_config,
)

logger = logging.getLogger(__name__)

ORTHOLOGS_DIR = Path(__file__).resolve().parent / "dicts" / "orthologs"
DROSOPHILA_DICTS_DIR = Path(__file__).resolve().parent / "dicts" / "drosophila"
_TABLE_CACHE: dict[str, dict[str, str]] = {}

# Re-export for callers that import policy from gene_converter.
__all_policies__ = OrthologPolicy


class ConversionPair(str, Enum):
    MOUSE_TO_HUMAN = "mouse_to_human"
    HUMAN_TO_MOUSE = "human_to_mouse"
    DROSOPHILA_TO_HUMAN = "drosophila_to_human"
    DROSOPHILA_TO_MOUSE = "drosophila_to_mouse"


_PAIR_TO_FILE = {
    ConversionPair.MOUSE_TO_HUMAN: "mouse_to_human.tsv",
    ConversionPair.HUMAN_TO_MOUSE: "human_to_mouse.tsv",
    ConversionPair.DROSOPHILA_TO_HUMAN: "drosophila_to_human.tsv",
    ConversionPair.DROSOPHILA_TO_MOUSE: "drosophila_to_mouse.tsv",
}

_CURATED_SUFFIX = "_curated.tsv"
_ONE2ONE_TYPES = frozenset({"ortholog_one2one", "one2one"})


def normalize_gene_id(gene: str) -> str:
    """Strip Ensembl version suffixes (e.g. ENSMUSG00000039323.2 → ENSMUSG00000039323)."""
    gene = str(gene).strip()
    if gene.startswith("ENS"):
        return gene.split(".", 1)[0]
    return gene


def is_ensembl_id(gene: str) -> bool:
    return str(gene).startswith("ENS")


def is_fly_id(gene: str) -> bool:
    return str(gene).startswith("FBgn")


def is_primary_gene_id(gene: str) -> bool:
    return is_ensembl_id(gene) or is_fly_id(gene)


@dataclass
class ConversionResult:
    mapped: dict[str, str] = field(default_factory=dict)
    unmapped: list[str] = field(default_factory=list)

    @property
    def kept_count(self) -> int:
        return len(self.mapped)

    @property
    def dropped_count(self) -> int:
        return len(self.unmapped)


@dataclass
class GeneMappingInspection:
    """Per-gene ortholog audit row (pre/post policy collapse)."""

    source_id: str
    input_id: str
    candidates: list[str] = field(default_factory=list)
    orthology_types: list[str] = field(default_factory=list)
    selected_target: str | None = None
    status: str = "no_ortholog"
    via_curated_overlay: bool = False
    via_platform_curated: bool = False


@dataclass(frozen=True)
class _OrthologRow:
    source: str
    target: str
    orthology_type: str = ""


def parse_ortholog_policy(
    species: Mapping[str, str] | None = None,
    *,
    policy: OrthologPolicy | str | None = None,
) -> OrthologPolicy:
    if policy is not None:
        return OrthologPolicy(policy) if isinstance(policy, str) else policy
    parsed = parse_species_config(species)
    return OrthologPolicy(parsed["ortholog_policy"])


def conversion_pair(model_organism: Organism | str, model_id: ModelId | str) -> ConversionPair | None:
    org = Organism(model_organism) if isinstance(model_organism, str) else model_organism
    mid = ModelId(model_id) if isinstance(model_id, str) else model_id
    if not needs_gene_conversion(org, mid):
        return None
    target = get_backend({"model": mid.value}).native_organism
    key = (org, target)
    mapping = {
        (Organism.MOUSE, Organism.HUMAN): ConversionPair.MOUSE_TO_HUMAN,
        (Organism.HUMAN, Organism.MOUSE): ConversionPair.HUMAN_TO_MOUSE,
        (Organism.DROSOPHILA, Organism.HUMAN): ConversionPair.DROSOPHILA_TO_HUMAN,
        (Organism.DROSOPHILA, Organism.MOUSE): ConversionPair.DROSOPHILA_TO_MOUSE,
    }
    pair = mapping.get(key)
    if pair is None:
        raise ValueError(f"Unsupported conversion: {org.value} → {target.value}")
    return pair


def should_convert(species: Mapping[str, str] | None) -> bool:
    if not species:
        return False
    parsed = parse_species_config(species)
    return needs_gene_conversion(parsed["model_organism"], parsed["model"])


def _normalize_source_id(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    return normalize_gene_id(raw) if is_ensembl_id(raw) else raw


def _normalize_target_id(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    return normalize_gene_id(raw) if is_ensembl_id(raw) else raw


def _read_ortholog_rows(path: Path) -> list[_OrthologRow]:
    rows: list[_OrthologRow] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            src = _normalize_source_id(row.get("source_id") or "")
            tgt = _normalize_target_id(row.get("target_id") or "")
            otype = (row.get("orthology_type") or row.get("homology_type") or "").strip().lower()
            if src and tgt:
                rows.append(_OrthologRow(source=src, target=tgt, orthology_type=otype))
    return rows


def _collapse_legacy_sum(rows: Sequence[_OrthologRow]) -> dict[str, str]:
    """Last-write-wins: one target per source (old behavior)."""
    table: dict[str, str] = {}
    for row in rows:
        if row.source in table and table[row.source] != row.target:
            logger.debug(
                "Ortholog legacy_sum: overwriting %s (%s → %s)",
                row.source,
                table[row.source],
                row.target,
            )
        table[row.source] = row.target
    return table


def _collapse_best_of_n(rows: Sequence[_OrthologRow]) -> dict[str, str]:
    """
    Deterministic 1→N pick (lexicographically smallest target); allow N→1.
    Expression disambiguation for N→1 happens later in remap_adata_ensembl_ids.
    """
    by_source: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_source[row.source].add(row.target)
    table: dict[str, str] = {}
    for src, tgts in by_source.items():
        chosen = sorted(tgts)[0]
        if len(tgts) > 1:
            logger.debug(
                "Ortholog best_of_n: source %s has %d targets; keeping %s",
                src,
                len(tgts),
                chosen,
            )
        table[src] = chosen
    return table


def _collapse_one2one(rows: Sequence[_OrthologRow]) -> dict[str, str]:
    """
    Keep reciprocal 1:1 pairs only.

    Prefer BioMart orthology_type == ortholog_one2one when that column is present;
    otherwise require unique source and unique target among primary IDs.
    """
    typed = [r for r in rows if r.orthology_type]
    if typed:
        candidates = [r for r in rows if r.orthology_type in _ONE2ONE_TYPES]
        if not candidates:
            # Column present but no one2one labels — fall back to uniqueness on all rows.
            candidates = list(rows)
    else:
        candidates = list(rows)

    src_to_tgts: dict[str, set[str]] = defaultdict(set)
    tgt_to_srcs: dict[str, set[str]] = defaultdict(set)
    for row in candidates:
        if is_primary_gene_id(row.source):
            src_to_tgts[row.source].add(row.target)
            tgt_to_srcs[row.target].add(row.source)

    table: dict[str, str] = {}
    dropped_ambiguous = 0
    for row in candidates:
        if is_primary_gene_id(row.source):
            if len(src_to_tgts[row.source]) != 1 or len(tgt_to_srcs[row.target]) != 1:
                dropped_ambiguous += 1
                continue
        # Symbol / alias rows: keep as alternate keys when unique enough.
        if row.source in table and table[row.source] != row.target:
            dropped_ambiguous += 1
            continue
        table[row.source] = row.target

    if dropped_ambiguous:
        logger.info(
            "Ortholog one2one: dropped %d ambiguous row(s); kept %d mappings",
            dropped_ambiguous,
            len(table),
        )
    return table


def _apply_curated_overrides(
    table: dict[str, str],
    curated_rows: Sequence[_OrthologRow],
    *,
    policy: OrthologPolicy,
) -> dict[str, str]:
    out = dict(table)
    curated_primary: set[str] = set()
    for row in curated_rows:
        if row.source in out and out[row.source] != row.target:
            logger.debug(
                "Curated override %s: %s → %s",
                row.source,
                out[row.source],
                row.target,
            )
        out[row.source] = row.target
        if is_primary_gene_id(row.source):
            curated_primary.add(row.source)

    if policy is OrthologPolicy.ONE2ONE:
        out = _drop_primary_target_collisions(out, prefer=curated_primary)
    return out


def _drop_primary_target_collisions(
    table: Mapping[str, str],
    *,
    prefer: set[str] | None = None,
) -> dict[str, str]:
    """
    Drop primary-ID sources that share a target (non-injective after curated merge).

    When ``prefer`` contains exactly one colliding source (typical curated Ensembl),
    keep that source and drop the others.
    """
    prefer = prefer or set()
    tgt_to_srcs: dict[str, list[str]] = defaultdict(list)
    for src, tgt in table.items():
        if is_primary_gene_id(src):
            tgt_to_srcs[tgt].append(src)

    drop: set[str] = set()
    for tgt, srcs in tgt_to_srcs.items():
        if len(srcs) <= 1:
            continue
        preferred = [s for s in srcs if s in prefer]
        if len(preferred) == 1:
            keep = preferred[0]
            drop.update(s for s in srcs if s != keep)
            logger.info(
                "Ortholog one2one: target %s claimed by %d sources; "
                "keeping curated %s, dropping %s",
                tgt,
                len(srcs),
                keep,
                ", ".join(sorted(s for s in srcs if s != keep)[:8]),
            )
        else:
            drop.update(srcs)
            logger.info(
                "Ortholog one2one: dropping %d primary sources for shared target %s: %s",
                len(srcs),
                tgt,
                ", ".join(sorted(srcs)[:8]),
            )

    if not drop:
        return dict(table)
    return {k: v for k, v in table.items() if k not in drop}


def resolve_curated_overlay_path(
    pair: ConversionPair,
    *,
    curated_overlay: str | Path | Mapping[str, str] | None = None,
) -> Path | None:
    """
    Resolve an optional project-scoped curated overlay TSV for ``pair``.

    ``curated_overlay`` may be:
      - a TSV file path (applied for this pair as given),
      - a directory containing ``curated_bridge_{pair}.tsv``,
      - a mapping of pair name → TSV path.

    When ``curated_overlay`` is None/empty, falls back to
    ``GENEFORMER_ORTHOLOG_CURATED_OVERLAY``.
    """
    raw: str | Path | Mapping[str, str] | None = curated_overlay
    if raw is None or raw == "":
        env = os.environ.get("GENEFORMER_ORTHOLOG_CURATED_OVERLAY", "").strip()
        raw = env or None
    if raw is None or raw == "":
        return None

    if isinstance(raw, Mapping):
        cand = raw.get(pair.value) or raw.get(pair.value.replace("_to_", "2"))
        if not cand:
            return None
        path = Path(str(cand)).expanduser()
        return path if path.is_file() else None

    path = Path(str(raw)).expanduser()
    if path.is_dir():
        for name in (
            f"curated_bridge_{pair.value}.tsv",
            f"{pair.value}_overlay.tsv",
            f"{pair.value}_curated_overlay.tsv",
        ):
            cand = path / name
            if cand.is_file():
                return cand
        return None
    if path.is_file():
        return path
    logger.warning("Ortholog curated overlay not found: %s", path)
    return None


def load_ortholog_rows(
    pair: ConversionPair,
    *,
    orthologs_dir: Path | None = None,
    curated_overlay: str | Path | Mapping[str, str] | None = None,
) -> tuple[list[_OrthologRow], list[_OrthologRow], list[_OrthologRow], Path, Path | None]:
    """
    Load raw BioMart / curated / overlay rows without policy collapse.

    Returns ``(main_rows, platform_curated_rows, overlay_rows, main_path, overlay_path)``.
    """
    base = orthologs_dir or ORTHOLOGS_DIR
    path = base / _PAIR_TO_FILE[pair]
    if not path.is_file():
        raise FileNotFoundError(f"Ortholog table not found: {path}")
    main_rows = _read_ortholog_rows(path)
    curated_path = base / _PAIR_TO_FILE[pair].replace(".tsv", _CURATED_SUFFIX)
    curated_rows = _read_ortholog_rows(curated_path) if curated_path.is_file() else []
    overlay_path = resolve_curated_overlay_path(pair, curated_overlay=curated_overlay)
    overlay_rows = _read_ortholog_rows(overlay_path) if overlay_path is not None else []
    return main_rows, curated_rows, overlay_rows, path, overlay_path


def load_ortholog_table(
    pair: ConversionPair,
    *,
    orthologs_dir: Path | None = None,
    policy: OrthologPolicy | str | None = None,
    curated_overlay: str | Path | Mapping[str, str] | None = None,
) -> dict[str, str]:
    resolved_policy = parse_ortholog_policy(policy=policy)
    overlay_path = resolve_curated_overlay_path(pair, curated_overlay=curated_overlay)
    overlay_key = str(overlay_path.resolve()) if overlay_path is not None else ""
    cache_key = (
        f"{pair.value}|{resolved_policy.value}@"
        f"{orthologs_dir or ORTHOLOGS_DIR}|overlay={overlay_key}"
    )
    if cache_key in _TABLE_CACHE:
        return _TABLE_CACHE[cache_key]

    base = orthologs_dir or ORTHOLOGS_DIR
    path = base / _PAIR_TO_FILE[pair]
    if not path.is_file():
        raise FileNotFoundError(f"Ortholog table not found: {path}")

    main_rows = _read_ortholog_rows(path)
    if resolved_policy is OrthologPolicy.LEGACY_SUM:
        table = _collapse_legacy_sum(main_rows)
    elif resolved_policy is OrthologPolicy.BEST_OF_N:
        table = _collapse_best_of_n(main_rows)
    else:
        table = _collapse_one2one(main_rows)

    curated_path = base / _PAIR_TO_FILE[pair].replace(".tsv", _CURATED_SUFFIX)
    if curated_path.is_file():
        curated_rows = _read_ortholog_rows(curated_path)
        table = _apply_curated_overrides(table, curated_rows, policy=resolved_policy)
    elif resolved_policy is OrthologPolicy.ONE2ONE:
        table = _drop_primary_target_collisions(table)

    if overlay_path is not None:
        overlay_rows = _read_ortholog_rows(overlay_path)
        table = _apply_curated_overrides(table, overlay_rows, policy=resolved_policy)
        logger.info(
            "Ortholog curated overlay applied (%s): %s (%d row(s))",
            pair.value,
            overlay_path,
            len(overlay_rows),
        )

    _TABLE_CACHE[cache_key] = table
    return table


def _index_rows_by_source(rows: Sequence[_OrthologRow]) -> dict[str, list[_OrthologRow]]:
    by_src: dict[str, list[_OrthologRow]] = defaultdict(list)
    for row in rows:
        by_src[row.source].append(row)
        if is_ensembl_id(row.source) or is_fly_id(row.source):
            by_src[normalize_gene_id(row.source)].append(row)
    return by_src


def _classify_unmapped_status(types: Sequence[str]) -> str:
    lowered = [t.lower() for t in types if t]
    if any("one2many" in t or "one_to_many" in t for t in lowered):
        return "ortholog_one2many"
    if any("many2many" in t or "many_to_many" in t for t in lowered):
        return "ortholog_many2many"
    if lowered:
        return "dropped_ambiguous"
    return "no_ortholog"


def inspect_gene_mappings(
    gene_ids: Sequence[str],
    species: Mapping[str, str] | None,
    *,
    orthologs_dir: Path | None = None,
    input_symbol_table: Mapping[str, str] | None = None,
) -> list[GeneMappingInspection]:
    """
    Inspect each input gene against raw ortholog rows and the policy-collapsed table.

    Does not change collapse behavior; used by ``ortholog_loss_gate``.
    """
    parsed = parse_species_config(species)
    pair = conversion_pair(parsed["model_organism"], parsed["model"])
    policy = parse_ortholog_policy(parsed)
    overlay = _curated_overlay_from_species(parsed)
    unique_input = []
    seen: set[str] = set()
    for g in gene_ids:
        gid = str(g).strip()
        if not gid or gid in seen:
            continue
        seen.add(gid)
        unique_input.append(gid)

    if pair is None:
        return [
            GeneMappingInspection(
                source_id=normalize_gene_id(g) if is_ensembl_id(g) else g,
                input_id=g,
                selected_target=normalize_gene_id(g) if is_ensembl_id(g) or is_fly_id(g) else g,
                status="mapped_native",
            )
            for g in unique_input
        ]

    main_rows, curated_rows, overlay_rows, _main_path, _overlay_path = load_ortholog_rows(
        pair, orthologs_dir=orthologs_dir, curated_overlay=overlay
    )
    table = load_ortholog_table(
        pair,
        orthologs_dir=orthologs_dir,
        policy=policy,
        curated_overlay=overlay,
    )
    main_idx = _index_rows_by_source(main_rows)
    curated_idx = _index_rows_by_source(curated_rows)
    overlay_idx = _index_rows_by_source(overlay_rows)
    backend = _backend_for_model(parsed["model"], parsed)

    out: list[GeneMappingInspection] = []
    for gene in unique_input:
        lookup_keys = [gene]
        source_primary = gene
        if is_ensembl_id(gene) or is_fly_id(gene):
            source_primary = normalize_gene_id(gene)
            lookup_keys = [source_primary, gene]
        elif input_symbol_table:
            ens = _symbol_lookup(input_symbol_table, gene)
            if ens:
                source_primary = normalize_gene_id(ens)
                lookup_keys = [source_primary, gene, ens]

        cand_rows: list[_OrthologRow] = []
        for key in lookup_keys:
            cand_rows.extend(main_idx.get(key, []))
            cand_rows.extend(curated_idx.get(key, []))
            cand_rows.extend(overlay_idx.get(key, []))
        # Deduplicate while preserving order
        seen_rt: set[tuple[str, str, str]] = set()
        uniq_rows: list[_OrthologRow] = []
        for row in cand_rows:
            sig = (row.source, row.target, row.orthology_type)
            if sig in seen_rt:
                continue
            seen_rt.add(sig)
            uniq_rows.append(row)

        candidates = sorted({r.target for r in uniq_rows})
        types = sorted({r.orthology_type for r in uniq_rows if r.orthology_type})
        overlay_hit = False
        for k in lookup_keys:
            nk = normalize_gene_id(k) if is_ensembl_id(k) else k
            if k in overlay_idx or nk in overlay_idx:
                overlay_hit = True
                break
        via_platform = False
        for k in lookup_keys:
            nk = normalize_gene_id(k) if is_ensembl_id(k) else k
            if k in curated_idx or nk in curated_idx:
                via_platform = True
                break

        selected = _resolve_gene_id(
            gene,
            pair=pair,
            table=table,
            backend=backend,
            input_symbol_table=input_symbol_table,
        )
        if selected:
            if overlay_hit or (
                via_platform
                and any(
                    normalize_gene_id(r.source) == normalize_gene_id(source_primary)
                    or r.source == gene
                    for r in curated_rows
                )
            ):
                status = "curated_bridge"
            else:
                status = "mapped_one2one" if policy is OrthologPolicy.ONE2ONE else "mapped"
        else:
            status = _classify_unmapped_status(types)

        out.append(
            GeneMappingInspection(
                source_id=source_primary,
                input_id=gene,
                candidates=candidates,
                orthology_types=types,
                selected_target=selected,
                status=status,
                via_curated_overlay=overlay_hit,
                via_platform_curated=via_platform,
            )
        )
    return out


def _curated_overlay_from_species(
    species: Mapping[str, str] | None,
) -> str | None:
    if not species:
        return None
    raw = species.get("ortholog_curated_overlay")
    if raw is None or str(raw).strip() in ("", "null", "None"):
        return None
    return str(raw).strip()


def load_fly_symbol_table(*, dicts_dir: Path | None = None) -> dict[str, str]:
    """Fly gene symbol → FBgn (input-organism IDs for ortholog lookup)."""
    path = (dicts_dir or DROSOPHILA_DICTS_DIR) / "fly_symbol_to_fbgn.tsv"
    if not path.is_file():
        return {}
    table: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sym = (row.get("symbol") or "").strip()
            fbgn = (row.get("fbgn_id") or "").strip()
            if sym and fbgn:
                table[sym] = fbgn
    return table


def _load_symbol_to_ensembl(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def _symbol_lookup(symbol_table: Mapping[str, str], symbol: str) -> str | None:
    if symbol in symbol_table:
        return symbol_table[symbol]
    for variant in (symbol.upper(), symbol.lower(), symbol.capitalize()):
        if variant in symbol_table:
            return symbol_table[variant]
    return None


def _resolve_target_to_model_ensembl(target: str, backend: BackendSpec) -> str | None:
    """Map ortholog table target to model-vocabulary Ensembl (or fly ID)."""
    target = str(target).strip()
    if not target:
        return None
    normalized = normalize_gene_id(target)
    if is_ensembl_id(normalized) or is_fly_id(normalized):
        return normalized
    sym_path = backend.gene_symbol_to_ensembl
    if sym_path and sym_path.is_file():
        sym_table = _load_symbol_to_ensembl(sym_path)
        ens = _symbol_lookup(sym_table, target)
        if ens:
            return normalize_gene_id(ens)
    return None


def lookup_ortholog(table: Mapping[str, str], gene: str) -> str | None:
    """Lookup in ortholog table: exact key, then normalized Ensembl."""
    gene = str(gene).strip()
    if not gene:
        return None
    if gene in table:
        return table[gene]
    if is_ensembl_id(gene):
        return table.get(normalize_gene_id(gene))
    return None


def _backend_for_model(model_id: ModelId | str, species: Mapping[str, str] | None = None) -> BackendSpec:
    parsed = parse_species_config(species or {"model": model_id})
    cfg = dict(parsed)
    if isinstance(model_id, ModelId):
        cfg["model"] = model_id.value
    else:
        cfg["model"] = str(model_id)
    if not cfg.get("human_variant"):
        cfg["human_variant"] = "v2_104m"
    if not cfg.get("mouse_variant"):
        cfg["mouse_variant"] = "base"
    return get_backend(cfg)


def _passthrough_model_native_id(gene: str, backend: BackendSpec) -> str | None:
    normalized = normalize_gene_id(gene)
    native = backend.native_organism
    if native is Organism.HUMAN and normalized.startswith("ENSG"):
        return normalized
    if native is Organism.MOUSE and normalized.startswith("ENSMUSG"):
        return normalized
    if native is Organism.DROSOPHILA and is_fly_id(normalized):
        return normalized
    return None


def _resolve_gene_id(
    gene: str,
    *,
    pair: ConversionPair | None,
    table: Mapping[str, str],
    backend: BackendSpec,
    input_symbol_table: Mapping[str, str] | None = None,
) -> str | None:
    """
    Shared gene resolution for loom remap and ISP.

    Priority:
      1. Ortholog table on the raw input (Ensembl or curated symbol).
      2. Input-organism symbol → Ensembl → ortholog.
      3. Already model-native ID (or same-species Ensembl / symbol→Ensembl).
    """
    gene = str(gene).strip()
    if not gene:
        return None

    if pair is None:
        if not is_ensembl_id(gene) and input_symbol_table:
            ens = _symbol_lookup(input_symbol_table, gene)
            if ens:
                return normalize_gene_id(ens)
        if is_ensembl_id(gene) or is_fly_id(gene):
            return normalize_gene_id(gene)
        # Same-species: keep raw symbol so callers can still attempt vocab lookup.
        return gene

    # 1. Direct ortholog lookup on user input (curated symbols / corrected Ensembl).
    target = lookup_ortholog(table, gene)
    if target:
        resolved = _resolve_target_to_model_ensembl(target, backend)
        if resolved:
            return resolved

    # 2. Input symbol → input Ensembl → ortholog.
    if not is_ensembl_id(gene) and input_symbol_table:
        source_ens = _symbol_lookup(input_symbol_table, gene)
        if source_ens:
            target = lookup_ortholog(table, source_ens)
            if target:
                resolved = _resolve_target_to_model_ensembl(target, backend)
                if resolved:
                    return resolved

    # 3. Already model-native ID (re-entry after prior conversion).
    return _passthrough_model_native_id(gene, backend)


def convert_gene_ids(
    gene_ids: Sequence[str],
    *,
    model_organism: Organism | str,
    model_id: ModelId | str,
    ortholog_table: Mapping[str, str] | None = None,
    species: Mapping[str, str] | None = None,
    policy: OrthologPolicy | str | None = None,
    input_symbol_table: Mapping[str, str] | None = None,
) -> ConversionResult:
    """
    Map input gene IDs to the model vocabulary.

    Uses the same resolution rules as ``resolve_gene_for_model`` so loom
    remapping and ISP perturbation resolution cannot diverge.
    """
    pair = conversion_pair(model_organism, model_id)
    species_cfg = dict(species or {})
    if isinstance(model_organism, Organism):
        species_cfg.setdefault("model_organism", model_organism.value)
    else:
        species_cfg.setdefault("model_organism", str(model_organism))
    if isinstance(model_id, ModelId):
        species_cfg.setdefault("model", model_id.value)
    else:
        species_cfg.setdefault("model", str(model_id))

    if pair is None:
        backend = _backend_for_model(model_id, species_cfg)
        mapped: dict[str, str] = {}
        unmapped: list[str] = []
        for gene in gene_ids:
            resolved = _resolve_gene_id(
                gene,
                pair=None,
                table={},
                backend=backend,
                input_symbol_table=input_symbol_table,
            )
            if resolved is not None:
                mapped[gene] = resolved
            else:
                unmapped.append(gene)
        return ConversionResult(mapped=mapped, unmapped=unmapped)

    resolved_policy = parse_ortholog_policy(species_cfg, policy=policy)
    overlay = _curated_overlay_from_species(species_cfg)
    table = (
        ortholog_table
        if ortholog_table is not None
        else load_ortholog_table(
            pair, policy=resolved_policy, curated_overlay=overlay
        )
    )
    backend = _backend_for_model(model_id, species_cfg)
    mapped = {}
    unmapped = []
    for gene in gene_ids:
        resolved = _resolve_gene_id(
            gene,
            pair=pair,
            table=table,
            backend=backend,
            input_symbol_table=input_symbol_table,
        )
        if resolved:
            mapped[gene] = resolved
        else:
            unmapped.append(gene)
    return ConversionResult(mapped=mapped, unmapped=unmapped)


def summarize_conversion(result: ConversionResult, *, pair: ConversionPair | None) -> str:
    if pair is None:
        return "gene conversion: not required"
    return (
        f"gene conversion ({pair.value}): "
        f"kept={result.kept_count}, dropped={result.dropped_count}"
    )


def get_backend_for_species(species: Mapping[str, str] | None) -> BackendSpec:
    return get_backend(parse_species_config(species))


def resolve_gene_for_model(
    gene: str,
    species: Mapping[str, str] | None,
    *,
    input_symbol_table: Mapping[str, str] | None = None,
    ortholog_table: Mapping[str, str] | None = None,
) -> str | None:
    """
    Resolve a perturbation gene (symbol or Ensembl ID) to the model vocabulary ID.
    Returns None if the gene cannot be mapped.
    """
    parsed = parse_species_config(species)
    gene = str(gene).strip()
    if not gene:
        return None

    backend = get_backend(parsed)
    pair = conversion_pair(parsed["model_organism"], parsed["model"])
    table: Mapping[str, str] = {}
    if pair is not None:
        if ortholog_table is not None:
            table = ortholog_table
        else:
            policy = OrthologPolicy(parsed["ortholog_policy"])
            table = load_ortholog_table(
                pair,
                policy=policy,
                curated_overlay=_curated_overlay_from_species(parsed),
            )

    return _resolve_gene_id(
        gene,
        pair=pair,
        table=table,
        backend=backend,
        input_symbol_table=input_symbol_table,
    )


def convert_perturbation_genes(
    genes: Sequence[str],
    species: Mapping[str, str] | None,
    *,
    input_symbol_table: Mapping[str, str] | None = None,
) -> tuple[list[str], ConversionResult]:
    """Convert a list of ISP perturbation genes; drops unmapped genes."""
    parsed = parse_species_config(species)
    pair = conversion_pair(parsed["model_organism"], parsed["model"])
    resolved: list[str] = []
    unmapped: list[str] = []
    mapped_pairs: dict[str, str] = {}
    seen: set[str] = set()

    for gene in genes:
        target = resolve_gene_for_model(gene, parsed, input_symbol_table=input_symbol_table)
        if target and target not in seen:
            resolved.append(target)
            seen.add(target)
            mapped_pairs[gene] = target
        else:
            unmapped.append(gene)

    result = ConversionResult(mapped=mapped_pairs, unmapped=unmapped)
    if pair and unmapped:
        logger.warning(
            "%s — dropped unmapped perturbation genes: %s",
            summarize_conversion(result, pair=pair),
            unmapped,
        )
    return resolved, result


def _median_expression(column) -> float:
    import numpy as np

    arr = np.asarray(column).ravel()
    if arr.size == 0:
        return float("-inf")
    return float(np.median(arr))


def _collapse_expression_columns(
    adata,
    targets: list[str],
    *,
    policy: OrthologPolicy,
):
    """
    Collapse genes that share a model-vocabulary target ID.

    Returns (new_X or None if no collapse needed, unique_targets, keep_source_indices).
    keep_source_indices[j] lists original column indices contributing to unique target j
    for logging; for one2one/best_of_n only one index is kept per target.
    """
    import numpy as np
    import scipy.sparse as sp

    unique_targets: list[str] = []
    groups: dict[str, list[int]] = {}
    for j, t in enumerate(targets):
        if t not in groups:
            groups[t] = []
            unique_targets.append(t)
        groups[t].append(j)

    if len(unique_targets) == len(targets):
        return None, unique_targets, [[i] for i in range(len(targets))]

    n_rows = adata.n_obs
    n_cols = len(unique_targets)
    X = adata.X
    keep_per_target: list[list[int]] = []

    if policy is OrthologPolicy.LEGACY_SUM:
        row_index = [0] * len(targets)
        for new_j, t in enumerate(unique_targets):
            keep_per_target.append(groups[t])
            for old_j in groups[t]:
                row_index[old_j] = new_j
        if sp.issparse(X):
            coo = X.tocoo()
            new_cols = np.array([row_index[c] for c in coo.col], dtype=np.int32)
            new_X = sp.coo_matrix((coo.data, (coo.row, new_cols)), shape=(n_rows, n_cols)).tocsr()
        else:
            agg = np.zeros((n_rows, n_cols), dtype=X.dtype)
            for old_j, new_j in enumerate(row_index):
                agg[:, new_j] += np.asarray(X[:, old_j]).ravel()
            new_X = agg
        return new_X, unique_targets, keep_per_target

    # one2one / best_of_n: never sum — pick a single source column per target.
    chosen: list[int] = []
    for t in unique_targets:
        members = groups[t]
        if len(members) == 1 or policy is OrthologPolicy.ONE2ONE:
            # one2one: if collisions slipped through, keep first and drop the rest.
            pick = members[0]
            if len(members) > 1 and policy is OrthologPolicy.ONE2ONE:
                logger.warning(
                    "Ortholog one2one: unexpected N→1 for %s (%d sources); keeping first column",
                    t,
                    len(members),
                )
        else:
            # best_of_n: highest median expression across cells
            medians = []
            for old_j in members:
                col = X[:, old_j]
                if sp.issparse(col):
                    col = col.toarray()
                medians.append(_median_expression(col))
            pick = members[int(np.argmax(medians))]
        chosen.append(pick)
        keep_per_target.append([pick])

    new_X = X[:, chosen]
    return new_X, unique_targets, keep_per_target


def remap_adata_ensembl_ids(adata, species: Mapping[str, str] | None):
    """
    Remap adata.var ensembl_id through ortholog table when conversion is required.

    Gene IDs are resolved with the same rules as ISP (``resolve_gene_for_model``),
    including input-organism symbol→Ensembl when a symbol table is available.
    Collapse behavior depends on species.ortholog_policy (see module docstring).

    Returns (adata, ConversionResult).
    """
    import anndata as ad
    import numpy as np

    parsed = parse_species_config(species)
    pair = conversion_pair(parsed["model_organism"], parsed["model"])
    if pair is None:
        return adata, ConversionResult(mapped={})

    # Lazy import avoids a circular import with species_context.
    from geneformer.species_context import load_input_symbol_table

    policy = OrthologPolicy(parsed["ortholog_policy"])
    table = load_ortholog_table(
        pair,
        policy=policy,
        curated_overlay=_curated_overlay_from_species(parsed),
    )
    symbol_table = load_input_symbol_table(parsed)
    # Keep raw labels (strip only): Ensembl versions are normalized inside lookup;
    # gene symbols must remain intact for symbol-table resolution.
    source_ids = [str(x).strip() for x in adata.var["ensembl_id"].astype(str).tolist()]
    conv = convert_gene_ids(
        source_ids,
        model_organism=parsed["model_organism"],
        model_id=parsed["model"],
        ortholog_table=table,
        species=parsed,
        policy=policy,
        input_symbol_table=symbol_table or None,
    )

    target_for_row: list[str | None] = []
    for src in source_ids:
        target_for_row.append(conv.mapped.get(src))

    keep_idx = [i for i, t in enumerate(target_for_row) if t]
    if not keep_idx:
        raise ValueError(
            f"No genes remained after {pair.value} conversion ({policy.value}). "
            f"Dropped {len(conv.unmapped)} / {len(source_ids)} genes."
        )

    adata = adata[:, keep_idx].copy()
    targets = [target_for_row[i] for i in keep_idx]

    new_X, unique_targets, _keep = _collapse_expression_columns(adata, targets, policy=policy)
    if new_X is None:
        adata.var["ensembl_id"] = unique_targets
        adata.var_names = unique_targets
        logger.info("%s [policy=%s]", summarize_conversion(conv, pair=pair), policy.value)
        return adata, conv

    new_var = ad.AnnData(np.zeros((1, len(unique_targets)))).var
    new_var["ensembl_id"] = unique_targets
    new_var.index = unique_targets
    adata = ad.AnnData(X=new_X, obs=adata.obs.copy(), var=new_var)
    adata.var_names = unique_targets

    logger.info(
        "%s [policy=%s] (%d → %d genes)",
        summarize_conversion(conv, pair=pair),
        policy.value,
        len(source_ids),
        len(unique_targets),
    )
    return adata, conv
