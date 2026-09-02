"""Pre-tokenize ortholog conversion summary for cross-species runs."""
from __future__ import annotations

import csv
import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .backends.registry import get_backend, parse_species_config
from .gene_brief_function import (
    explain_drop_reason,
    invert_symbol_table,
    lookup_brief_function,
    lookup_symbol,
)
from .gene_converter import (
    conversion_pair,
    convert_gene_ids,
    inspect_gene_mappings,
    load_ortholog_table,
    normalize_gene_id,
    parse_ortholog_policy,
    should_convert,
)


@dataclass
class FileConversionStats:
    path: str
    input_genes: int
    mapped: int
    unmapped: int
    unique_targets: int


@dataclass
class ConversionReport:
    model_organism: str
    model: str
    conversion_pair: str | None
    ortholog_policy: str = "one2one"
    input_genes: int = 0
    mapped: int = 0
    unmapped: int = 0
    mapped_pct: float = 0.0
    unique_targets: int = 0
    collapsed_many_to_one: int = 0
    in_model_vocab: int = 0
    in_model_vocab_pct: float = 0.0
    files: list[FileConversionStats] = field(default_factory=list)
    unmapped_gene_ids: list[str] = field(default_factory=list)
    unmapped_details: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["files"] = [asdict(f) for f in self.files]
        # Per-gene rows live in conversion_unmapped_genes.tsv (can be tens of thousands).
        d.pop("unmapped_details", None)
        return d


def default_report_conversion(species: Mapping[str, str] | None) -> bool:
    """Default on whenever ortholog conversion applies (any cross-species pair)."""
    return should_convert(species)


def resolve_report_conversion(
    species: Mapping[str, str] | None,
    tokenizer_cfg: Mapping[str, Any] | None = None,
    *,
    cli_flag: bool | None = None,
) -> bool:
    if cli_flag is not None:
        return cli_flag
    cfg = (tokenizer_cfg or {}).get("report_conversion")
    if cfg is not None:
        return bool(cfg)
    return default_report_conversion(species)


def _model_vocab_keys(species: Mapping[str, str]) -> set[str]:
    backend = get_backend(species)
    try:
        with open(backend.token_dictionary, "rb") as f:
            tokens = set(pickle.load(f).keys())
        with open(backend.gene_median_dictionary, "rb") as f:
            medians = set(pickle.load(f).keys())
    except Exception:
        return set()
    return tokens & medians


def analyze_gene_conversion(
    gene_ids: Sequence[str],
    species: Mapping[str, str] | None,
    *,
    orthologs_dir: Path | None = None,
) -> ConversionReport:
    parsed = parse_species_config(species)
    pair = conversion_pair(parsed["model_organism"], parsed["model"])
    policy = parse_ortholog_policy(parsed)
    unique_input = sorted({normalize_gene_id(g) for g in gene_ids if str(g).strip()})

    report = ConversionReport(
        model_organism=parsed["model_organism"],
        model=parsed["model"],
        conversion_pair=pair.value if pair else None,
        ortholog_policy=policy.value,
        input_genes=len(unique_input),
    )

    if pair is None or not unique_input:
        report.unmapped_gene_ids = list(unique_input)
        report.unmapped = len(unique_input)
        return report

    table = load_ortholog_table(
        pair,
        orthologs_dir=orthologs_dir,
        policy=policy,
        curated_overlay=(parsed.get("ortholog_curated_overlay") or None),
    )
    # Match loom remap / ISP: resolve symbols via input-organism symbol table.
    from .species_context import load_input_symbol_table

    symbol_table = load_input_symbol_table(parsed)
    conv = convert_gene_ids(
        unique_input,
        model_organism=parsed["model_organism"],
        model_id=parsed["model"],
        ortholog_table=table,
        species=parsed,
        policy=policy,
        input_symbol_table=symbol_table or None,
    )
    targets = list(conv.mapped.values())
    unique_targets = sorted(set(targets))
    vocab = _model_vocab_keys(parsed)
    in_vocab = [t for t in unique_targets if t in vocab]

    report.mapped = len(conv.mapped)
    report.unmapped = len(conv.unmapped)
    report.mapped_pct = round(100.0 * report.mapped / report.input_genes, 2) if report.input_genes else 0.0
    report.unique_targets = len(unique_targets)
    report.collapsed_many_to_one = report.mapped - report.unique_targets
    report.in_model_vocab = len(in_vocab)
    report.in_model_vocab_pct = (
        round(100.0 * report.in_model_vocab / report.unique_targets, 2) if unique_targets else 0.0
    )
    report.unmapped_gene_ids = sorted(set(conv.unmapped))
    report.unmapped_details = _unmapped_details(
        report.unmapped_gene_ids,
        parsed,
        symbol_table,
        orthologs_dir=orthologs_dir,
    )
    return report


def _unmapped_details(
    unmapped_ids: Sequence[str],
    species: Mapping[str, str] | None,
    symbol_table: Mapping[str, str] | None,
    *,
    orthologs_dir: Path | None = None,
) -> list[dict[str, str]]:
    if not unmapped_ids:
        return []
    id_to_symbol = invert_symbol_table(dict(symbol_table or {}))
    status_by_id: dict[str, str] = {}
    try:
        inspections = inspect_gene_mappings(
            list(unmapped_ids),
            species,
            orthologs_dir=orthologs_dir,
            input_symbol_table=symbol_table or None,
        )
        for insp in inspections:
            status_by_id[insp.source_id] = insp.status
            if insp.input_id:
                status_by_id[insp.input_id] = insp.status
    except Exception:
        status_by_id = {}
    rows: list[dict[str, str]] = []
    for gid in unmapped_ids:
        symbol = id_to_symbol.get(gid, "") or lookup_symbol(gid)
        if not symbol and gid in (symbol_table or {}):
            symbol = gid
        status = status_by_id.get(gid, "")
        rows.append(
            {
                "source_id": gid,
                "symbol": symbol,
                "drop_reason": explain_drop_reason(status),
                "brief_function": lookup_brief_function(gid, symbol),
            }
        )
    return rows


def collect_loom_gene_ids(loom_path: Path | str) -> list[str]:
    import loompy as lp

    path = Path(loom_path)
    with lp.connect(str(path)) as data:
        if "ensembl_id" not in data.ra.keys():
            raise KeyError(
                f"{path} is missing required row attribute 'ensembl_id' "
                f"(found: {list(data.ra.keys())}). Refusing to fall back to "
                "unrelated row attributes."
            )
        return [normalize_gene_id(x) for x in data.ra["ensembl_id"]]


def build_report_from_loom_dir(
    loom_directory: Path | str,
    species: Mapping[str, str] | None,
    *,
    file_format: str = "loom",
    orthologs_dir: Path | None = None,
) -> ConversionReport:
    loom_dir = Path(loom_directory)
    parsed = parse_species_config(species)
    pair = conversion_pair(parsed["model_organism"], parsed["model"])
    policy = parse_ortholog_policy(parsed)
    report = ConversionReport(
        model_organism=parsed["model_organism"],
        model=parsed["model"],
        conversion_pair=pair.value if pair else None,
        ortholog_policy=policy.value,
    )

    if pair is None:
        return report

    all_genes: set[str] = set()
    for loom_path in sorted(loom_dir.glob(f"*.{file_format}")):
        gene_ids = collect_loom_gene_ids(loom_path)
        file_report = analyze_gene_conversion(gene_ids, species, orthologs_dir=orthologs_dir)
        report.files.append(
            FileConversionStats(
                path=str(loom_path),
                input_genes=file_report.input_genes,
                mapped=file_report.mapped,
                unmapped=file_report.unmapped,
                unique_targets=file_report.unique_targets,
            )
        )
        all_genes.update(gene_ids)

    if not all_genes:
        return report

    merged = analyze_gene_conversion(sorted(all_genes), species, orthologs_dir=orthologs_dir)
    report.input_genes = merged.input_genes
    report.mapped = merged.mapped
    report.unmapped = merged.unmapped
    report.mapped_pct = merged.mapped_pct
    report.unique_targets = merged.unique_targets
    report.collapsed_many_to_one = merged.collapsed_many_to_one
    report.in_model_vocab = merged.in_model_vocab
    report.in_model_vocab_pct = merged.in_model_vocab_pct
    report.unmapped_gene_ids = merged.unmapped_gene_ids
    report.unmapped_details = merged.unmapped_details
    return report


def format_conversion_report_text(report: ConversionReport) -> str:
    lines = [
        "=" * 72,
        "Gene conversion report (pre-tokenize)",
        "=" * 72,
    ]
    if report.conversion_pair is None:
        lines.append("  gene conversion: not required")
        lines.append("=" * 72)
        return "\n".join(lines)

    lines.extend(
        [
            f"  pair:                 {report.conversion_pair}",
            f"  ortholog_policy:      {report.ortholog_policy}",
            f"  model_organism:       {report.model_organism}",
            f"  model:                {report.model}",
            f"  input genes (unique): {report.input_genes:,}",
            f"  mapped:               {report.mapped:,} ({report.mapped_pct:.1f}%)",
            f"  unmapped (dropped):   {report.unmapped:,}",
            f"  unique model targets: {report.unique_targets:,}",
            f"  many-to-one collapse: {report.collapsed_many_to_one:,}",
            (
                f"  in model vocabulary:  {report.in_model_vocab:,}"
                f" ({report.in_model_vocab_pct:.1f}% of targets)"
            ),
        ]
    )
    if report.files:
        lines.append("  per-file:")
        for f in report.files:
            pct = 100.0 * f.mapped / f.input_genes if f.input_genes else 0.0
            lines.append(
                f"    {Path(f.path).name}: {f.mapped:,}/{f.input_genes:,} mapped"
                f" ({pct:.1f}%), {f.unique_targets:,} unique targets"
            )
    if report.unmapped_gene_ids:
        preview = ", ".join(report.unmapped_gene_ids[:8])
        suffix = "..." if len(report.unmapped_gene_ids) > 8 else ""
        lines.append(f"  unmapped sample:      {preview}{suffix}")
    lines.append("=" * 72)
    return "\n".join(lines)


def write_conversion_report(
    report: ConversionReport,
    output_dir: Path | str,
    *,
    max_unmapped_tsv: int = 50_000,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "conversion_report.json"
    json_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")

    paths: dict[str, str] = {"json": str(json_path)}
    if report.unmapped_gene_ids:
        tsv_path = out / "conversion_unmapped_genes.tsv"
        details = {
            row.get("source_id", ""): row for row in (report.unmapped_details or [])
        }
        with open(tsv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["source_id", "symbol", "drop_reason", "brief_function"])
            for gene in report.unmapped_gene_ids[:max_unmapped_tsv]:
                row = details.get(gene) or {}
                writer.writerow(
                    [
                        gene,
                        row.get("symbol", ""),
                        row.get("drop_reason", ""),
                        row.get("brief_function", ""),
                    ]
                )
        paths["unmapped_tsv"] = str(tsv_path)

    text_path = out / "conversion_report.txt"
    text_path.write_text(format_conversion_report_text(report) + "\n", encoding="utf-8")
    paths["text"] = str(text_path)
    return paths


def run_conversion_report(
    loom_directory: Path | str,
    output_dir: Path | str,
    species: Mapping[str, str] | None,
    *,
    file_format: str = "loom",
) -> ConversionReport | None:
    if not should_convert(species):
        report = ConversionReport(
            model_organism=parse_species_config(species)["model_organism"],
            model=parse_species_config(species)["model"],
            conversion_pair=None,
            ortholog_policy=parse_species_config(species)["ortholog_policy"],
        )
        text = format_conversion_report_text(report)
        print(text)
        write_conversion_report(report, output_dir)
        return report

    report = build_report_from_loom_dir(loom_directory, species, file_format=file_format)
    text = format_conversion_report_text(report)
    print(text)
    paths = write_conversion_report(report, output_dir)
    print(f"  conversion report:    {paths.get('json')}")
    if "unmapped_tsv" in paths:
        print(f"  unmapped genes TSV:   {paths['unmapped_tsv']}")
    return report
