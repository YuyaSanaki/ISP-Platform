"""
10x input layout for tokenize, pipeline, and Streamlit (one rule everywhere).

Canonical layout:

  /data/ExperimentName/Time-Condition-Replicate/*.gz

  - **data.input_dir** = `/data/ExperimentName/` (study root; same as tokenize service)
  - Each **immediate subfolder** of ExperimentName is one sample (name like `1w-Ctrl-Rep1`)
  - Inside each sample folder: `barcodes.tsv.gz`, `features.tsv.gz`, `matrix.mtx.gz`
    (or `filtered_feature_bc_matrix/` with those files)

Multiple experiments under `/data/`:

  /data/
    ExperimentA/1w-Ctrl-Rep1/*.gz
    ExperimentA/1w-Disease-Rep1/*.gz
    ExperimentB/2w-Ctrl-Rep2/*.gz

  Set **data.input_dir** = `/data/` — all sample folders under each experiment are found.
"""
from __future__ import annotations

import re
from pathlib import Path

DEFAULT_DISCOVER_DEPTH = 2  # ExperimentName / sample (API compat; canonical uses fixed depth)

_MATRIX_NAMES = ("matrix.mtx.gz", "matrix.mtx")
_BARCODES_NAMES = ("barcodes.tsv.gz", "barcodes.tsv", "barcode.tsv.gz", "barcode.tsv")
_FEATURES_NAMES = ("features.tsv.gz", "features.tsv", "feature.tsv.gz", "feature.tsv")

# Time-Condition-Replicate (at least three hyphen-separated fields)
CANONICAL_SAMPLE_NAME = re.compile(r"^[^/\\]+-[^/\\]+-[^/\\]+")


def parse_sample_folder_name(folder_name: str) -> dict[str, str]:
    """
    Parse sample folder names like ``1w-Disease-SingleCell`` or ``1w-Ctrl-SingleCell``.

    Returns time, disease/state (for ISP), genotype, replicate/suffix, sample_id.
    """
    name = folder_name.strip()
    parts = name.split("-")
    time = parts[0] if parts else name
    state = parts[1] if len(parts) > 1 else name
    suffix = "-".join(parts[2:]) if len(parts) > 2 else ""
    return {
        "time": time,
        "disease": state,
        "state": state,
        "genotype": state,
        "replicate": suffix,
        "sample_id": name,
    }


def unique_states_from_samples(study_root: Path) -> list[str]:
    """Sorted unique state/condition tokens (2nd hyphen field) from sample folder names."""
    states: set[str] = set()
    for sample in discover_sample_dirs(study_root):
        states.add(parse_sample_folder_name(sample.name)["state"])
    return sorted(states)


# Columns written when extract_metadata_from_path is on (see execute_tokenizer_pipeline).
FOLDER_DERIVED_LABEL_COLUMNS: tuple[str, ...] = (
    "disease",
    "genotype",
    "state",
    "time",
    "replicate",
    "sample_id",
)


def label_column_candidates_from_study(
    study_root: Path | None,
    *,
    extra: list[str] | tuple[str, ...] = (),
) -> list[str]:
    """
    Candidate dataset columns for fine-tune ``label_column`` / ISP ``state_key``.

    Prefers folder-derived metadata that tokenization will attach when the study
    has canonical ``Time-Condition-Replicate`` sample folders. Always includes
    ``cell_type`` and ``disease`` as common fine-tune targets.
    """
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(name: object) -> None:
        col = str(name).strip()
        if col and col not in seen:
            seen.add(col)
            ordered.append(col)

    for name in extra:
        _add(name)

    if study_root is not None:
        root = Path(study_root)
        try:
            if root.is_dir() and discover_sample_dirs(root):
                for name in FOLDER_DERIVED_LABEL_COLUMNS:
                    _add(name)
        except OSError:
            pass

    _add("cell_type")
    _add("disease")
    return ordered


def _any_file(path: Path, names: tuple[str, ...]) -> bool:
    return any((path / n).is_file() for n in names)


def has_10x_mtx_triple(path: Path) -> bool:
    path = path.resolve()
    return (
        _any_file(path, _MATRIX_NAMES)
        and _any_file(path, _BARCODES_NAMES)
        and _any_file(path, _FEATURES_NAMES)
    )


def is_10x_mtx_dir(path: Path) -> bool:
    path = path.resolve()
    if (path / "filtered_feature_bc_matrix").is_dir():
        inner = path / "filtered_feature_bc_matrix"
        return has_10x_mtx_triple(inner) or _any_file(inner, _MATRIX_NAMES)
    return has_10x_mtx_triple(path) or _any_file(path, _MATRIX_NAMES)


def is_canonical_sample_name(name: str) -> bool:
    return bool(CANONICAL_SAMPLE_NAME.match(name.strip()))


def tenx_matrix_directory(sample_dir: Path) -> Path | None:
    sample_dir = sample_dir.resolve()
    filtered = sample_dir / "filtered_feature_bc_matrix"
    if filtered.is_dir() and (has_10x_mtx_triple(filtered) or _any_file(filtered, _MATRIX_NAMES)):
        return filtered
    if has_10x_mtx_triple(sample_dir) or _any_file(sample_dir, _MATRIX_NAMES):
        return sample_dir
    return None


def _child_dirs(base: Path, exclude: Path | None) -> list[Path]:
    exclude_resolved = exclude.resolve() if exclude else None
    out: list[Path] = []
    if not base.is_dir():
        return out
    for child in sorted(base.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if exclude_resolved and child.resolve() == exclude_resolved:
            continue
        out.append(child)
    return out


def discover_sample_dirs(
    study_root: Path,
    *,
    exclude: Path | None = None,
    max_depth: int = DEFAULT_DISCOVER_DEPTH,
) -> list[Path]:
    """
    Sample folders under study_root following ExperimentName / Time-Condition-Replicate.

    - study_root = ExperimentName → children with 10x files are samples.
    - study_root = /data → ExperimentName / Time-Condition-Replicate / *.gz
    """
    del max_depth  # canonical layout uses fixed depth (1 or 2 levels)
    study_root = study_root.resolve()
    exclude_resolved = exclude.resolve() if exclude else None
    if exclude_resolved and study_root == exclude_resolved:
        return []

    direct = [c for c in _child_dirs(study_root, exclude) if is_10x_mtx_dir(c)]
    if direct:
        return direct

    nested: list[Path] = []
    for experiment in _child_dirs(study_root, exclude):
        for sample in _child_dirs(experiment, exclude):
            if is_10x_mtx_dir(sample):
                nested.append(sample)
    if nested:
        return nested

    if is_10x_mtx_dir(study_root):
        return [study_root]

    return []


def sample_label(sample_dir: Path, study_root: Path) -> str:
    """Loom / metadata id (e.g. 1w-Ctrl-Rep1 or ExpA-1w-Ctrl-Rep1)."""
    try:
        rel = sample_dir.resolve().relative_to(study_root.resolve())
        if not rel.parts:
            return sample_dir.name
        if len(rel.parts) == 1:
            return rel.parts[0]
        return "-".join(rel.parts)
    except ValueError:
        return sample_dir.name


def list_sample_dirs(study_root: Path, *, exclude: Path | None = None) -> list[Path]:
    return discover_sample_dirs(study_root, exclude=exclude)


def _common_parent(paths: list[Path]) -> Path:
    if not paths:
        raise ValueError("paths must be non-empty")
    if len(paths) == 1:
        return paths[0].parent
    parts_list = [p.resolve().parts for p in paths]
    common: list[str] = []
    for tokens in zip(*parts_list):
        if len(set(tokens)) == 1:
            common.append(tokens[0])
        else:
            break
    return Path(*common) if common else paths[0].parent


def resolve_single_cell_input_dir(
    input_dir: str | Path,
    loom_temp_dir: str | Path | None = None,
) -> Path:
    """
    Study root for data.input_dir (tokenize + pipeline).

    Use `/data/ExperimentName/` for one experiment, or `/data/` for many.
    If you point at a single sample folder, returns its ExperimentName parent.

    When ``input_dir`` already contains 10x sample folders as direct children
    (including symlink children such as a PIPseq-only view), keep that directory
    and do **not** climb to a symlink-target parent via ``_common_parent``.
    """
    root = Path(input_dir).resolve()
    loom = Path(loom_temp_dir).resolve() if loom_temp_dir else None
    samples = discover_sample_dirs(root, exclude=loom)

    if not samples:
        return root

    if len(samples) == 1 and samples[0].resolve() == root:
        return root

    # Symlink-safe: PIPseq/ with children -> ../full_study/Sample must stay PIPseq/.
    direct = [c for c in _child_dirs(root, loom) if is_10x_mtx_dir(c)]
    if direct:
        return root

    return _common_parent(samples)


def count_single_cell_samples(
    input_dir: str | Path,
    loom_temp_dir: str | Path | None = None,
) -> tuple[int, Path, str]:
    """Returns (count, study_root, layout_hint)."""
    root = Path(input_dir).resolve()
    loom = Path(loom_temp_dir).resolve() if loom_temp_dir else None
    study_root = resolve_single_cell_input_dir(root, loom)
    samples = discover_sample_dirs(study_root, exclude=loom)

    if not samples:
        return 0, study_root, "none detected"

    noncanonical = [s.name for s in samples if not is_canonical_sample_name(s.name)]
    name_hint = ""
    if noncanonical:
        name_hint = f"; naming hint: use Time-Condition-Replicate ({noncanonical[0]} …)"

    direct = [c for c in _child_dirs(study_root, loom) if is_10x_mtx_dir(c)]
    if direct:
        layout = f"{len(samples)} sample(s) under ExperimentName `{study_root.name}`"
    else:
        layout = f"{len(samples)} sample(s) under `/data/` (multiple experiments)"

    preview = ", ".join(s.name for s in samples[:8])
    if len(samples) > 8:
        preview += ", …"
    return len(samples), study_root, f"{layout}: {preview}{name_hint}"


def count_loom_files(loom_dir: str | Path) -> int:
    return len(list(Path(loom_dir).glob("*.loom")))


def describe_expected_layout() -> str:
    """Short hint for CLI errors; full layout is in docs/tokenization.md."""
    return "See docs/tokenization.md for input layout."


def diagnose_input_dir(input_dir: str | Path, loom_temp_dir: str | Path | None = None) -> str:
    root = Path(input_dir).resolve()
    study_root = resolve_single_cell_input_dir(root, loom_temp_dir)
    samples = discover_sample_dirs(
        study_root,
        exclude=Path(loom_temp_dir) if loom_temp_dir else None,
    )
    n = len(samples)
    layout = count_single_cell_samples(root, loom_temp_dir)[2]

    lines = [
        f"YAML input_dir: {root}",
        f"study root (tokenize): {study_root}",
        f"samples detected: {n} ({layout})",
    ]
    if samples:
        lines.append("sample folders (Time-Condition-Replicate):")
        for s in samples[:12]:
            ok = "ok" if is_canonical_sample_name(s.name) else "non-standard name"
            lines.append(f"  - {s}  [{ok}]")
        if len(samples) > 12:
            lines.append(f"  … and {len(samples) - 12} more")
    if study_root != root:
        lines.append(f"Tip: set data.input_dir to `{study_root}`.")
    if n == 0:
        try:
            entries = sorted(p.name for p in root.iterdir() if not p.name.startswith("."))[:20]
            lines.append(f"Top-level entries: {entries or '(empty)'}")
        except OSError:
            pass
    return "\n".join(lines)


__all__ = [
    "CANONICAL_SAMPLE_NAME",
    "DEFAULT_DISCOVER_DEPTH",
    "count_loom_files",
    "count_single_cell_samples",
    "describe_expected_layout",
    "diagnose_input_dir",
    "discover_sample_dirs",
    "has_10x_mtx_triple",
    "is_10x_mtx_dir",
    "is_canonical_sample_name",
    "list_sample_dirs",
    "parse_sample_folder_name",
    "resolve_single_cell_input_dir",
    "sample_label",
    "tenx_matrix_directory",
    "unique_states_from_samples",
    "FOLDER_DERIVED_LABEL_COLUMNS",
    "label_column_candidates_from_study",
]
