"""
Import study data from a .zip for the Streamlit UI.

Zip layout (sample folders with 10x .gz inside; optional ``data/`` wrapper in zip):

  1w-Disease-SingleCell/barcodes.tsv.gz  ...
  1w-Ctrl-SingleCell/...

Files are extracted under:

  {upload_session}/{study_name}/1w-Disease-SingleCell/...

If the zip has one extra wrapper (e.g. StudyBatch/), sample folders are hoisted so
``data.input_dir`` points at ``{upload_session}/{study_name}/`` with samples as
direct children.
"""
from __future__ import annotations

import io
import re
import shutil
import zipfile
from pathlib import Path

from data_input_layout import (
    diagnose_input_dir,
    discover_sample_dirs,
    is_10x_mtx_dir,
    resolve_single_cell_input_dir,
    unique_states_from_samples,
)

_JUNK_DIR_NAMES = frozenset({"__MACOSX", ".ipynb_checkpoints"})
_JUNK_FILE_NAMES = frozenset({".DS_Store"})


def normalize_study_name(name: str) -> str:
    """Sanitize experiment/study name; empty string if missing or invalid."""
    return re.sub(r"[^\w.-]+", "_", (name or "").strip()).strip("_")


def study_folder(upload_dir: Path, study_name: str) -> Path:
    return upload_dir.resolve() / normalize_study_name(study_name)


def _zip_example_layout() -> str:
    return (
        "Zip must contain one folder per sample, each with 10x matrix files:\n\n"
        "```\n"
        "1w-Disease-SingleCell/barcodes.tsv.gz\n"
        "1w-Disease-SingleCell/features.tsv.gz\n"
        "1w-Disease-SingleCell/matrix.mtx.gz\n"
        "1w-Ctrl-SingleCell/barcodes.tsv.gz\n"
        "1w-Ctrl-SingleCell/features.tsv.gz\n"
        "1w-Ctrl-SingleCell/matrix.mtx.gz\n"
        "```\n\n"
        "Optional wrappers inside the zip (`data/`, `StudyBatch/`, etc.) are flattened automatically."
    )


def _is_junk_zip_member(filename: str) -> bool:
    parts = Path(filename).parts
    if any(part in _JUNK_DIR_NAMES for part in parts):
        return True
    base = parts[-1] if parts else filename
    if base in _JUNK_FILE_NAMES or base.startswith("._"):
        return True
    return False


def _is_junk_dir_name(name: str) -> bool:
    return name in _JUNK_DIR_NAMES or name.startswith(".")


def extract_zip_to_dir(zip_bytes: bytes, dest: Path) -> None:
    dest = dest.resolve()
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        _extract_zipfile(zf, dest)


def extract_zip_path_to_dir(zip_path: Path, dest: Path) -> None:
    """Extract a zip on disk into dest (same safety rules as extract_zip_to_dir)."""
    dest = dest.resolve()
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        _extract_zipfile(zf, dest)


def _extract_zipfile(zf: zipfile.ZipFile, dest: Path) -> None:
    for member in zf.infolist():
        if member.is_dir() or _is_junk_zip_member(member.filename):
            continue
        target = (dest / member.filename).resolve()
        if dest not in target.parents and target != dest:
            raise ValueError(f"Unsafe path in zip: {member.filename}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)


def _remove_junk_from_tree(root: Path) -> None:
    root = root.resolve()
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir() and path.name in _JUNK_DIR_NAMES:
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_file() and (
            path.name in _JUNK_FILE_NAMES or path.name.startswith("._")
        ):
            path.unlink(missing_ok=True)


def _meaningful_child_dirs(parent: Path) -> list[Path]:
    return [
        p
        for p in sorted(parent.iterdir())
        if p.is_dir() and not _is_junk_dir_name(p.name)
    ]


def _direct_sample_dirs(study_dir: Path) -> list[Path]:
    """Sample folders that are immediate children of study_dir (not nested under a wrapper)."""
    return [p for p in _meaningful_child_dirs(study_dir) if is_10x_mtx_dir(p)]


def _hoist_sample_folders_to_study_root(study_dir: Path) -> None:
    """
    If samples live under a wrapper (e.g. StudyBatch/1w-Disease-SingleCell), move them to study_dir/.
    """
    study_dir = study_dir.resolve()
    if _direct_sample_dirs(study_dir):
        return

    for wrapper in _meaningful_child_dirs(study_dir):
        if is_10x_mtx_dir(wrapper):
            continue
        inner = discover_sample_dirs(wrapper)
        if not inner:
            continue
        for sample in inner:
            dest = study_dir / sample.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(sample), str(dest))
        try:
            wrapper.rmdir()
        except OSError:
            shutil.rmtree(wrapper, ignore_errors=True)
        break


def prepare_study_directory(study_dir: Path) -> None:
    """Remove macOS zip junk and hoist sample folders to the study root."""
    study_dir = study_dir.resolve()
    if not study_dir.is_dir():
        return
    _remove_junk_from_tree(study_dir)
    _hoist_sample_folders_to_study_root(study_dir)
    _remove_junk_from_tree(study_dir)


def _has_samples(study_dir: Path) -> bool:
    prepare_study_directory(study_dir)
    tokenize_dir = resolve_single_cell_input_dir(study_dir)
    return bool(discover_sample_dirs(tokenize_dir))


def ensure_study_folder(upload_dir: Path, study_name: str) -> tuple[Path, str | None]:
    """
    Return the study directory for study_name.

    If the zip was imported under another folder name
    and there is exactly one folder with data in the session, rename it to study_name.
    """
    upload_dir = upload_dir.resolve()
    target = study_folder(upload_dir, study_name)
    target_name = normalize_study_name(study_name)

    if target.is_dir() and _has_samples(target):
        prepare_study_directory(target)
        return target, None

    others: list[Path] = []
    for path in sorted(upload_dir.iterdir()):
        if not path.is_dir() or _is_junk_dir_name(path.name):
            continue
        if path.name == target_name:
            continue
        if _has_samples(path):
            others.append(path)

    if len(others) == 1:
        src = others[0]
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(src), str(target))
        prepare_study_directory(target)
        return target, src.name

    if len(others) > 1:
        names = ", ".join(p.name for p in others)
        raise ValueError(
            f"Multiple study folders in this session ({names}). "
            "Remove extras or pick one study name."
        )

    target.mkdir(parents=True, exist_ok=True)
    return target, None


def resolve_study_tokenize_dir(upload_dir: Path, study_name: str) -> Path | None:
    """Study root for tokenize after prepare/rename; None if no name or no samples."""
    if not normalize_study_name(study_name):
        return None
    try:
        study_dir, _ = ensure_study_folder(upload_dir, study_name)
    except ValueError:
        return None
    prepare_study_directory(study_dir)
    tokenize_dir = resolve_single_cell_input_dir(study_dir)
    if discover_sample_dirs(tokenize_dir):
        return tokenize_dir
    return None


def import_study_zip(
    zip_bytes: bytes,
    upload_dir: Path,
    study_name: str,
) -> tuple[Path, str]:
    """
    Extract zip under upload_dir/{study_name}/, validate, return (tokenize input_dir, summary).
    """
    if not normalize_study_name(study_name):
        raise ValueError(
            "Enter an **experiment name** in Study name before uploading a .zip."
        )
    upload_dir = upload_dir.resolve()
    study_dir = study_folder(upload_dir, study_name)
    extract_zip_to_dir(zip_bytes, study_dir)
    return _finalize_imported_study(study_dir, study_name)


def import_study_zip_path(
    zip_path: Path,
    upload_dir: Path,
    study_name: str,
) -> tuple[Path, str]:
    """Like import_study_zip, but reads a zip file from disk (no full RAM buffer)."""
    if not normalize_study_name(study_name):
        raise ValueError(
            "Enter an **experiment name** in Study name before uploading a .zip."
        )
    upload_dir = upload_dir.resolve()
    study_dir = study_folder(upload_dir, study_name)
    extract_zip_path_to_dir(Path(zip_path), study_dir)
    return _finalize_imported_study(study_dir, study_name)


def _finalize_imported_study(study_dir: Path, study_name: str) -> tuple[Path, str]:
    prepare_study_directory(study_dir)

    tokenize_dir = resolve_single_cell_input_dir(study_dir)
    samples = discover_sample_dirs(tokenize_dir)

    if not samples:
        raise ValueError(
            "Invalid study zip: no sample folders with 10x files were found.\n\n"
            + _zip_example_layout()
            + "\n\n"
            + diagnose_input_dir(study_dir)
        )

    lines = [
        f"**Study name:** `{normalize_study_name(study_name)}`",
        f"**data.input_dir:** `{tokenize_dir}`",
        f"**{len(samples)} sample folder(s):**",
    ]
    for s in samples:
        lines.append(f"- `{s.name}`" + (" ✓" if is_10x_mtx_dir(s) else ""))

    states = unique_states_from_samples(tokenize_dir)
    if states:
        lines.append(f"**States for ISP** (from folder names): {', '.join(states)}")

    return tokenize_dir, "\n".join(lines)


def summarize_existing_study(upload_dir: Path, study_name: str) -> str:
    tokenize_dir = resolve_study_tokenize_dir(upload_dir, study_name)
    if tokenize_dir is None:
        target = study_folder(upload_dir, study_name)
        if not target.is_dir():
            return "Upload a **.zip**, paste a **dataset URL**, or choose an existing study name folder."
        return f"No valid samples under `{target}`."
    samples = discover_sample_dirs(tokenize_dir)
    lines = [
        f"**{len(samples)} sample folder(s)** at `{tokenize_dir}`:",
    ]
    for s in samples:
        lines.append(f"- `{s.name}`")
    states = unique_states_from_samples(tokenize_dir)
    if states:
        lines.append(f"States: **{', '.join(states)}**")
    return "\n".join(lines)
