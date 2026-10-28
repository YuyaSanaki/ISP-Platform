"""Sample metadata manifest (QIIME2-style TSV) for 10x studies.

When this file is present, tokenize reads cell metadata from it instead of
splitting the sample folder name into time / condition / replicate.

Required columns: sample-id, disease, species.
Any further columns (for example experiment) are copied onto each cell.
``species`` must be a single value for the study (mouse, human, or
drosophila) and must match ``species.model_organism``.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path

# Filenames sought in the study root when metadata_manifest is unset.
MANIFEST_FILENAMES: tuple[str, ...] = (
    "metadata.tsv",
    "sample-metadata.tsv",
    "manifest.tsv",
)

# Same organisms as geneformer.backends.registry.Organism. Kept here so
# contracts/ does not import core/.
ALLOWED_SPECIES: frozenset[str] = frozenset({"mouse", "human", "drosophila"})

_REQUIRED = ("sample_id", "disease", "species")
_ID_HEADERS = frozenset({"sample_id", "sampleid"})
_PATH_HEADERS = frozenset({"filepath", "absolute_filepath"})
_COLUMN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_OBS = frozenset({"n_counts", "ensembl_id", "input_ids", "length", "cell_id"})


class SampleManifestError(ValueError):
    """The metadata manifest is missing, ambiguous, or inconsistent."""


@dataclass(frozen=True)
class SampleManifest:
    """One row per sample folder. ``by_folder`` keys are 10x directory names."""

    path: Path
    columns: tuple[str, ...]
    by_folder: dict[str, dict[str, str]]


def _normalize_header(name: str) -> str:
    key = name.strip().lstrip("\ufeff").lower().replace("-", "_").replace(" ", "_")
    if key in _ID_HEADERS:
        return "sample_id"
    if not key or not _COLUMN_RE.match(key):
        raise SampleManifestError(
            f"column name {name!r} is not a letter, digit, or underscore"
        )
    if key in _RESERVED_OBS:
        raise SampleManifestError(f"column {key!r} is reserved")
    return key


def _folder_name(sample_id: str, filepath: str) -> str:
    if not filepath:
        return sample_id
    path = Path(filepath)
    if path.name == "filtered_feature_bc_matrix":
        return path.parent.name or sample_id
    return path.name or sample_id


def _open_rows(path: Path) -> list[list[str]]:
    text = path.read_text(encoding="utf-8-sig")
    lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise SampleManifestError(f"{path} is empty")
    header_line = lines[0]
    if "\t" in header_line:
        delimiter = "\t"
    elif "," in header_line:
        delimiter = ","
    else:
        raise SampleManifestError(f"{path} must be tab-separated (or comma-separated)")
    rows = list(csv.reader(lines, delimiter=delimiter))
    if rows and rows[0] and rows[0][0].strip().lower() == "#q2:types":
        raise SampleManifestError(f"{path} has a types row before the header")
    if len(rows) >= 2 and rows[1] and rows[1][0].strip().lower() == "#q2:types":
        rows.pop(1)
    return rows


def load_sample_manifest(path: str | Path) -> SampleManifest:
    """Parse a metadata TSV. Raises SampleManifestError on any problem."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise SampleManifestError(f"metadata manifest not found: {manifest_path}")
    rows = _open_rows(manifest_path)
    header_raw = rows[0]
    if not header_raw or not any(cell.strip() for cell in header_raw):
        raise SampleManifestError(f"{manifest_path} has an empty header")

    headers: list[str] = []
    seen: set[str] = set()
    for cell in header_raw:
        key = _normalize_header(cell)
        if key in seen:
            raise SampleManifestError(f"duplicate column {key!r} in {manifest_path}")
        seen.add(key)
        headers.append(key)
    if headers[0] != "sample_id":
        raise SampleManifestError(
            f"the first column of {manifest_path.name} must be sample-id"
        )
    if "disease" not in seen and "state" not in seen:
        raise SampleManifestError(f"{manifest_path.name} needs a disease column")
    if "species" not in seen:
        raise SampleManifestError(f"{manifest_path.name} needs a species column")

    obs_columns: list[str] = []
    for key in headers:
        if key in _PATH_HEADERS:
            continue
        if key == "state" and "disease" not in seen:
            continue
        obs_columns.append(key)
    if "disease" not in obs_columns:
        insert_at = 1 if obs_columns and obs_columns[0] == "sample_id" else 0
        obs_columns.insert(insert_at, "disease")

    by_folder: dict[str, dict[str, str]] = {}
    seen_ids: set[str] = set()
    for row_index, cells in enumerate(rows[1:], start=2):
        if cells and cells[0].strip().lower() == "#q2:types":
            continue
        padded = list(cells) + [""] * (len(headers) - len(cells))
        values = {
            headers[i]: padded[i].strip()
            for i in range(len(headers))
        }
        sample_id = values.get("sample_id", "")
        if not sample_id:
            raise SampleManifestError(f"{manifest_path.name} row {row_index}: sample-id is empty")
        if sample_id in seen_ids:
            raise SampleManifestError(
                f"{manifest_path.name} row {row_index}: duplicate sample-id {sample_id!r}"
            )
        seen_ids.add(sample_id)

        disease = values.get("disease", "")
        if not disease and "state" in values:
            disease = values["state"]
        if not disease:
            raise SampleManifestError(
                f"{manifest_path.name} row {row_index}: disease is empty ({sample_id})"
            )
        species = values.get("species", "").lower()
        if species not in ALLOWED_SPECIES:
            allowed = ", ".join(sorted(ALLOWED_SPECIES))
            raise SampleManifestError(
                f"{manifest_path.name} row {row_index}: species must be one of {allowed}, "
                f"got {values.get('species', '')!r}"
            )
        obs = {
            key: values[key]
            for key in headers
            if key not in _PATH_HEADERS and not (key == "state" and "disease" not in seen)
        }
        obs["sample_id"] = sample_id
        obs["disease"] = disease
        obs["species"] = species
        if "state" in values and "state" in seen and "disease" in seen:
            obs["state"] = values["state"]

        folder = _folder_name(sample_id, values.get("filepath") or values.get("absolute_filepath") or "")
        if not folder:
            raise SampleManifestError(
                f"{manifest_path.name} row {row_index}: no sample folder for {sample_id}"
            )
        if folder in by_folder:
            raise SampleManifestError(
                f"{manifest_path.name} row {row_index}: sample folder {folder!r} is already used"
            )
        by_folder[folder] = obs

    if not by_folder:
        raise SampleManifestError(f"{manifest_path.name} has no sample rows")

    species_values = {row["species"] for row in by_folder.values()}
    if len(species_values) != 1:
        joined = ", ".join(sorted(species_values))
        raise SampleManifestError(
            f"{manifest_path.name} species column has more than one value ({joined}). "
            "One tokenize run uses one species."
        )

    return SampleManifest(
        path=manifest_path.resolve(),
        columns=tuple(obs_columns),
        by_folder=by_folder,
    )


def find_sample_manifest(
    study_root: str | Path,
    *,
    explicit: str | Path | None = None,
) -> Path | None:
    """Return the manifest path, or None when the study has no manifest.

    ``explicit`` must exist. With no explicit path, look for
    ``metadata.tsv``, ``sample-metadata.tsv``, or ``manifest.tsv`` in the
    study root. More than one of those names is an error.
    """
    if explicit is not None and str(explicit).strip().lower() not in {"", "none", "null"}:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise SampleManifestError(f"metadata manifest not found: {path}")
        return path

    root = Path(study_root)
    if not root.is_dir():
        return None
    found = [root / name for name in MANIFEST_FILENAMES if (root / name).is_file()]
    if not found:
        return None
    if len(found) > 1:
        names = ", ".join(path.name for path in found)
        raise SampleManifestError(
            f"{root} has more than one metadata manifest ({names}). "
            "Keep one file, or set single_cell_settings.metadata_manifest."
        )
    return found[0]


def resolve_sample_manifest(
    study_root: str | Path,
    settings: dict | None = None,
) -> SampleManifest | None:
    """Load the study manifest from settings or the study root. None if absent."""
    explicit = None
    if settings:
        explicit = settings.get("metadata_manifest")
    path = find_sample_manifest(study_root, explicit=explicit)
    if path is None:
        return None
    return load_sample_manifest(path)


def manifest_column_names(study_root: str | Path) -> list[str]:
    """Obs columns from a study-root manifest. Empty when no manifest is present."""
    path = find_sample_manifest(study_root)
    if path is None:
        return []
    return list(load_sample_manifest(path).columns)


def manifest_species(manifest: SampleManifest) -> str:
    values = {row["species"] for row in manifest.by_folder.values()}
    if len(values) != 1:
        joined = ", ".join(sorted(values))
        raise SampleManifestError(
            f"species column has more than one value ({joined}). One tokenize run uses one species."
        )
    return next(iter(values))


def check_manifest_covers_samples(manifest: SampleManifest, sample_names: set[str]) -> None:
    """Every 10x folder needs a row, and every row needs a folder."""
    declared = set(manifest.by_folder)
    missing = sorted(sample_names - declared)
    extra = sorted(declared - sample_names)
    parts: list[str] = []
    if missing:
        parts.append("no manifest row for " + ", ".join(missing))
    if extra:
        parts.append("no sample folder for " + ", ".join(extra))
    if parts:
        raise SampleManifestError(". ".join(parts) + f". Manifest: {manifest.path.name}.")


def manifest_field_order(columns: tuple[str, ...] | list[str]) -> list[str]:
    """Metadata fields other than sample_id, with disease and species first."""
    fields = [col for col in columns if col != "sample_id"]
    ordered: list[str] = []
    for name in ("disease", "species"):
        if name in fields:
            ordered.append(name)
    for name in fields:
        if name not in ordered:
            ordered.append(name)
    return ordered


def validate_sample_metadata(samples: list[dict[str, str]]) -> str | None:
    """Return an error message, or None when every sample can be written.

    Each dict needs ``sample_id``, ``disease``, and ``species``.
    ``species`` must be one organism for the whole table.
    """
    if not samples:
        return "Manifest has no samples."
    species_values: set[str] = set()
    seen_ids: set[str] = set()
    for row in samples:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            return "Sample name is empty."
        if sample_id in seen_ids:
            return f"Duplicate sample name {sample_id}."
        seen_ids.add(sample_id)
        if not str(row.get("disease", "")).strip():
            return f"disease is empty for {sample_id}."
        species = str(row.get("species", "")).strip().lower()
        if species not in ALLOWED_SPECIES:
            allowed = ", ".join(sorted(ALLOWED_SPECIES))
            return f"species for {sample_id} must be one of {allowed}."
        species_values.add(species)
    if len(species_values) != 1:
        joined = ", ".join(sorted(species_values))
        return f"species must be the same for every sample ({joined})."
    return None


def samples_to_metadata_tsv(fields: list[str], samples: list[dict[str, str]]) -> str:
    """Tab-separated manifest. ``fields`` are metadata columns, not sample-id."""
    header = ["sample-id", *fields]
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(header)
    for row in samples:
        species = str(row.get("species", "")).strip().lower()
        values = [str(row.get("sample_id", "")).strip()]
        for field in fields:
            cell = species if field == "species" else str(row.get(field, "")).strip()
            values.append(cell)
        writer.writerow(values)
    return buffer.getvalue()
