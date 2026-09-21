"""
Mouse Geneformer — Streamlit control panel (ISP³ Platform Web UI; same image as CLI).

Monorepo: webui/ talks to core/ only through subprocess + YAML (see docs/architecture.md).

Layout: data upload | run type | YAML configuration | execute + live log + outputs.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

import streamlit as st
import yaml

# Friendly labels for species / model selectors (config values stay snake_case).
_ORGANISM_LABELS = {
    "mouse": "Mouse",
    "human": "Human",
    "drosophila": "Drosophila (fruit fly) (Beta)",
}
_MODEL_LABELS = {
    "mouse_geneformer": "Mouse Geneformer",
    "human_geneformer": "Human Geneformer V2",
}
_MOUSE_VARIANT_LABELS = {
    "base": "Base (6L / ~10M)",
    "12l_e20": "Large (12L-E20)",
}
_HUMAN_VARIANT_LABELS = {
    "v2_104m": "Base (V2-104M)",
    "v2_316m": "Large (V2-316M)",
}
_ORTHOLOG_POLICY_LABELS = {
    "one2one": "one2one — recommended (strict 1:1, no summing)",
    "best_of_n": "best_of_n — keep highest-expressing gene only",
    "legacy_sum": "legacy_sum — old behavior (sum counts; can distort ranks)",
}
_STATS_MODE_LABELS = {
    "goal_state_shift": "goal_state_shift — start→goal shift (recommended)",
    "vs_null": "vs_null — compare to a null distribution dataset",
    "mixture_model": "mixture_model — impact vs no-impact mixture",
    "aggregate_data": "aggregate_data — aggregate cosine shifts only",
}
_PERTURB_TYPE_LABELS = {
    "delete": "delete — remove gene from rank encoding",
    "overexpress": "overexpress — move gene to front of ranks",
    "inhibit": "inhibit — move gene toward lower quartile",
    "activate": "activate — move gene toward higher quartile",
}
_FT_TASK_TYPE_LABELS = {
    "disease": "disease — classify disease / condition labels",
    "cell_type": "cell_type — classify cell-type labels",
}
_DEFAULT_FT_EPOCHS = 10
_DEFAULT_FT_LEARNING_RATE = 5e-5
_DEFAULT_FT_NUM_RUNS = 1
_FT_WARMUP_500_STEPS = "500_steps"
_FT_WARMUP_RATIO_005 = "ratio_0.05"
_DEFAULT_FT_WARMUP_MODE = _FT_WARMUP_500_STEPS
_FT_WARMUP_MODE_LABELS = {
    _FT_WARMUP_500_STEPS: "500 steps",
    _FT_WARMUP_RATIO_005: "rate 0.05",
}
_DEFAULT_ISP_MAX_NCELLS = 2000
_DEFAULT_STATS_MODE = "goal_state_shift"
_DEFAULT_ISP_ANALYSIS_ENABLED = True
_DEFAULT_ISP_POSTPROCESS_ENABLED = False
_DEFAULT_ISP_POSTPROCESS_N_CLUSTERS = 4
_DEFAULT_ISP_POSTPROCESS_N_CLUSTERS_MODE = "manual"  # "auto" | "manual"
_DEFAULT_ISP_POSTPROCESS_CELLTYPE = True
_DEFAULT_PERTURB_TYPE = "delete"
_DEFAULT_STATE_KEY = "disease"
_DEFAULT_FT_TASK_TYPE = "disease"
_DEFAULT_FT_LABEL_COLUMN = "disease"
_DEFAULT_MAX_CELLS = 300_000


def _normalize_n_clusters_value(value: object) -> int | str:
    """Return ``\"auto\"`` or an int >= 2 for ``postprocess.n_clusters``."""
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    try:
        k = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _DEFAULT_ISP_POSTPROCESS_N_CLUSTERS
    return max(2, k)


def _n_clusters_from_mode_session(
    mode_key: str, value_key: str, *, default_mode: str = _DEFAULT_ISP_POSTPROCESS_N_CLUSTERS_MODE
) -> int | str:
    mode = str(st.session_state.get(mode_key, default_mode) or default_mode).strip().lower()
    if mode == "auto":
        return "auto"
    return _normalize_n_clusters_value(
        st.session_state.get(value_key, _DEFAULT_ISP_POSTPROCESS_N_CLUSTERS)
    )


def _set_n_clusters_session_from_config(
    mode_key: str, value_key: str, configured: object
) -> None:
    """Seed mode/value session keys from a YAML ``n_clusters`` (auto or int)."""
    normalized = _normalize_n_clusters_value(configured)
    if normalized == "auto":
        st.session_state.setdefault(mode_key, "auto")
        st.session_state.setdefault(value_key, _DEFAULT_ISP_POSTPROCESS_N_CLUSTERS)
    else:
        st.session_state.setdefault(mode_key, "manual")
        st.session_state.setdefault(value_key, int(normalized))


def _repo_root() -> Path:
    """Monorepo root (contains core/, webui/, contracts/)."""
    env = os.environ.get("WEBUI_ROOT")
    if env:
        return Path(env).resolve()
    # webui/streamlit_app/app.py → repo root is parents[2]
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "core").is_dir() and (candidate / "webui").is_dir():
        return candidate
    alt = Path(__file__).resolve().parents[1]
    if (alt.parent / "core").is_dir():
        return alt.parent
    return Path(os.getcwd()).resolve()


ROOT = _repo_root()
CORE = ROOT / "core"
CONTRACTS = ROOT / "contracts"
WEBUI_DIR = ROOT / "webui"

# Ensure monorepo packages resolve even if PYTHONPATH was not set (local streamlit).
for _p in (CORE, CONTRACTS, WEBUI_DIR):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

# Ortholog Block → approval_record UI (CLI gate remains authoritative).
from ortholog_approval_ui import (  # noqa: E402
    OrthologApprovalError as _OrthologApprovalError,
    card_rows_from_request as _ortholog_card_rows,
    create_approval_record_from_ui as _create_ortholog_approval_record,
    find_ortholog_block_artifacts as _find_ortholog_block_artifacts,
    resolve_default_audit_path as _resolve_ortholog_audit_path,
    resolve_default_overlay_path as _resolve_ortholog_overlay_path,
)
from dropped_genes_ui import (  # noqa: E402
    load_dropped_gene_table as _load_dropped_gene_table,
)


def _import_data_input_layout():
    """Load contracts/data_input_layout (avoids stale sys.modules / wrong path)."""
    import importlib.util

    path = CONTRACTS / "data_input_layout.py"
    if not path.is_file():
        raise ImportError(f"Missing {path}")
    spec = importlib.util.spec_from_file_location("data_input_layout", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["data_input_layout"] = mod
    spec.loader.exec_module(mod)
    required = (
        "count_single_cell_samples",
        "diagnose_input_dir",
        "resolve_single_cell_input_dir",
    )
    missing = [n for n in required if not hasattr(mod, n)]
    if missing:
        raise ImportError(
            f"{path} is outdated (missing {', '.join(missing)}). "
            "Restart the webui container after updating the repo."
        )
    return mod


_dil = _import_data_input_layout()
count_single_cell_samples = _dil.count_single_cell_samples
diagnose_input_dir = _dil.diagnose_input_dir
resolve_single_cell_input_dir = _dil.resolve_single_cell_input_dir
unique_states_from_samples = _dil.unique_states_from_samples
label_column_candidates_from_study = _dil.label_column_candidates_from_study

from streamlit_upload import (
    import_study_zip,
    normalize_study_name,
    resolve_study_tokenize_dir,
    study_folder,
    summarize_existing_study,
)
from streamlit_remote_data import RemoteDataError, import_study_from_url

discover_sample_dirs = _dil.discover_sample_dirs
WORKSPACE = Path(os.environ.get("WEBUI_WORKSPACE", ROOT / "data" / "streamlit_workspace")).resolve()

RUN_TYPE_PIPELINE = "Pipeline (E2E)"
RUN_TYPE_FT_BATCH = "FT batch size (calibrate)"
RUN_TYPE_ISP_UMAP = "ISP UMAP"
RUN_TYPE_SEQUENTIAL_ISP = "Sequential ISP"

ISP_UMAP_POSITION_DIRECT = "Direct UMAP (default)"
ISP_UMAP_POSITION_PCA50 = "PCA(50) → UMAP"
ISP_UMAP_PCA_COMPONENTS = 50
ISP_UMAP_PCA_UMAP_SEED = 0
ISP_UMAP_DIRECT_UMAP_SEED = 42

RUN_FILES = {
    RUN_TYPE_PIPELINE: "pipeline.yaml",
    RUN_TYPE_FT_BATCH: "ft_batch_calibrate.yaml",
    RUN_TYPE_ISP_UMAP: "isp_umap.yaml",
    RUN_TYPE_SEQUENTIAL_ISP: "sequential_isp.yaml",
}

# Run types that share Study name / Data input with Pipeline.
_STUDY_RUN_TYPES = frozenset({RUN_TYPE_PIPELINE, RUN_TYPE_FT_BATCH})

_SEQ_ISP_TYPE_LABELS = {
    "overexpress": "overexpress (OE) — move genes to front of ranks",
    "delete": "delete (KD) — remove genes from the rank encoding",
}
_DEFAULT_SEQ_ISP_STEPS = 2
_MAX_SEQ_ISP_STEPS = 12

FT_BATCH_RESULT_FILENAME = "recommended_train_batch_size.json"
FT_BATCH_RESULT_MARKER = "RECOMMENDED_TRAIN_BATCH_SIZE="
DEFAULT_FT_TRAIN_BATCH_SIZE = 6


def _default_config_path(run_label: str) -> Path:
    return CORE / "config" / RUN_FILES[run_label]


def _load_default_yaml() -> None:
    name = st.session_state.get("run_type_sel", RUN_TYPE_PIPELINE)
    path = _default_config_path(name)
    if path.is_file():
        st.session_state["yaml_editor"] = path.read_text(encoding="utf-8")
    else:
        st.session_state["yaml_editor"] = f"# Missing file: {path}\n"
    if name in _STUDY_RUN_TYPES:
        upload_dir = _session_upload_dir()
        _sync_pipeline_form_from_yaml(upload_dir)
        # Keep uploaded study paths when switching Pipeline ↔ FT calibrate.
        _reapply_loaded_study_to_yaml(upload_dir)
        if name == RUN_TYPE_PIPELINE:
            _sync_ft_train_batch_controls()
            _sync_batch_size_controls()
        _sync_species_form_from_yaml()


def _reapply_loaded_study_to_yaml(upload_dir: Path) -> None:
    """After a Run-type template reload, restore data.input_dir from the session study."""
    study_name = normalize_study_name(_raw_study_name())
    if not study_name:
        return
    tokenize_dir = resolve_study_tokenize_dir(upload_dir, study_name)
    if tokenize_dir is None:
        return
    st.session_state["pipeline_tokenize_dir"] = str(tokenize_dir)
    _patch_pipeline_yaml(input_dir=tokenize_dir, output_prefix=study_name)
    try:
        states = unique_states_from_samples(tokenize_dir)
    except Exception:
        states = []
    if states:
        st.session_state["pipeline_detected_states"] = states

def _ensure_workspace() -> Path:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    return WORKSPACE


def _session_upload_dir() -> Path:
    sid = st.session_state.setdefault("upload_session_id", uuid.uuid4().hex[:10])
    d = _ensure_workspace() / "uploads" / sid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tail_log(path: Path, max_bytes: int = 64_000) -> str:
    """Read only the last max_bytes of a log (do not load multi-MB files whole)."""
    if not path.is_file():
        return "(Waiting for log file…)"
    try:
        size = path.stat().st_size
    except OSError:
        return "(Waiting for log file…)"
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(-max_bytes, os.SEEK_END)
            data = b"... (showing tail)\n" + f.read()
        else:
            data = f.read()
    return data.decode("utf-8", errors="replace")


def _pipeline_zip_skip(rel: Path) -> bool:
    """Skip multi-GB artifacts that would OOM the browser download button."""
    parts = rel.parts
    if not parts:
        return False
    name = rel.name
    if name.startswith("conversion_") or name in (
        "critical_gene_audit.tsv",
        "ortholog_loss_summary.json",
        "mapping_provenance.json",
    ):
        return False
    if parts[0] in ("tokenized_dataset", "loom_files"):
        return True
    if any(p.startswith("checkpoint-") for p in parts):
        return True
    return False


def _build_pipeline_run_zip(run_dir: Path) -> tuple[bytes, str] | None:
    """Zip one pipeline_* run for download (skips tokenized data and training checkpoints)."""
    run_dir = run_dir.resolve()
    if not run_dir.is_dir() or not run_dir.name.startswith("pipeline_"):
        return None

    entries: list[tuple[Path, str]] = []
    for fp in sorted(run_dir.rglob("*")):
        if not fp.is_file():
            continue
        rel = fp.relative_to(run_dir)
        if _pipeline_zip_skip(rel):
            continue
        entries.append((fp, f"{run_dir.name}/{rel.as_posix()}"))

    if not entries:
        return None

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp, arcname in entries:
            zf.write(fp, arcname=arcname)
    return buf.getvalue(), f"{run_dir.name}.zip"


def _pipeline_run_summary(run_dir: Path) -> list[str]:
    """Short list of top-level artifacts in a pipeline run folder."""
    if not run_dir.is_dir():
        return []
    items: list[str] = []
    try:
        for p in sorted(run_dir.iterdir(), key=lambda x: x.name):
            if p.name.startswith("."):
                continue
            suffix = "/" if p.is_dir() else ""
            items.append(f"{p.name}{suffix}")
    except OSError:
        return []
    return items[:20]


def _render_dropped_genes_panel(
    search_roots: list[str | Path | None],
    *,
    key_prefix: str,
) -> None:
    """Ortholog conversion drops: whether any, which genes, brief function."""
    table = _load_dropped_gene_table(search_roots)
    st.markdown("**Dropped genes (ortholog conversion)**")
    if table is None:
        st.caption(
            "No conversion report in this folder. Same-species runs do not drop genes "
            "by ortholog mapping; cross-species tokenize writes "
            "`conversion_unmapped_genes.tsv` under `tokenized_dataset/`."
        )
        return
    pair = table.conversion_pair or "cross-species"
    policy = table.ortholog_policy or "—"
    if table.unmapped <= 0 and not table.rows:
        st.success(
            f"No genes dropped by ortholog conversion (`{pair}`, policy=`{policy}`). "
            f"Mapped {table.mapped:,} / {table.input_genes:,} input genes "
            f"({table.mapped_pct:.1f}%)."
        )
        return
    st.warning(
        f"**{table.unmapped:,}** gene(s) dropped (`{pair}`, policy=`{policy}`). "
        f"Mapped {table.mapped:,} / {table.input_genes:,} "
        f"({table.mapped_pct:.1f}%)."
    )
    st.caption(
        "Brief function is a local one-liner (curated note when available, otherwise "
        "NCBI official full name). It is not a pathway analysis."
    )
    query = st.text_input(
        "Filter dropped genes",
        key=f"{key_prefix}_dropped_filter",
        placeholder="Symbol or Ensembl / FlyBase ID",
    )
    rows = table.rows
    q = (query or "").strip().lower()
    if q:
        rows = [
            r
            for r in rows
            if q in r["Gene ID"].lower()
            or q in r["Symbol"].lower()
            or q in r["Why dropped"].lower()
            or q in r["Brief function"].lower()
        ]
    st.caption(f"Showing {len(rows):,} / {len(table.rows):,} dropped gene(s).")
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True, height=360)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
    except Exception:
        st.table(rows[:50])
        csv_bytes = None
        if table.unmapped_tsv and table.unmapped_tsv.is_file():
            csv_bytes = table.unmapped_tsv.read_bytes()
    if csv_bytes:
        st.download_button(
            label="Download dropped-gene table (.csv)",
            data=csv_bytes,
            file_name="dropped_genes.csv",
            mime="text/csv",
            key=f"{key_prefix}_dropped_csv",
        )
    src = table.unmapped_tsv or table.report_json
    if src:
        st.caption(f"Source: `{src}`")


def _build_command_and_env(run_label: str, config_path: Path) -> tuple[list[str], dict[str, str]]:
    cfg = str(config_path)
    env = os.environ.copy()
    env.setdefault("WANDB_DISABLED", "true")
    # Child jobs import geneformer / pipeline_lib / data_input_layout via PYTHONPATH.
    monorepo_path = os.pathsep.join(str(p) for p in (CORE, CONTRACTS, WEBUI_DIR))
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        monorepo_path if not existing else monorepo_path + os.pathsep + existing
    )

    # WebUI → core: subprocess + YAML only (no direct core imports for job execution).
    if run_label == RUN_TYPE_ISP_UMAP:
        source_run = str(st.session_state.get("isp_umap_source_run_dir") or "").strip()
        genes = _normalize_isp_umap_genes(st.session_state.get("isp_umap_genes_text"))
        if not genes:
            genes = _normalize_isp_umap_genes(st.session_state.get("isp_umap_gene"))
        show_arrows = bool(st.session_state.get("isp_umap_show_trajectories", True))
        num_arrows = int(st.session_state.get("isp_umap_num_arrows", 100) or 0)
        pca_components, umap_seed = _isp_umap_pca_components_and_seed()
        if source_run:
            cmd = [
                "python3",
                str(CORE / "run_isp_umap.py"),
                "--run-dir",
                source_run,
            ]
            if genes:
                cmd.extend(["--gene", *genes])
        else:
            cmd = ["python3", str(CORE / "run_isp_umap.py"), "--config", cfg]
        cmd.extend(["--pca-components", str(pca_components)])
        cmd.extend(["--umap-seed", str(umap_seed)])
        if show_arrows:
            cmd.append("--show-trajectory-arrows")
            cmd.extend(["--num-trajectory-arrows", str(max(1, num_arrows))])
        else:
            cmd.append("--no-trajectory-arrows")
        if bool(st.session_state.get("isp_umap_postprocess_enabled", False)):
            cmd.append("--enable-postprocess")
            cmd.extend(
                [
                    "--postprocess-n-clusters",
                    str(
                        _n_clusters_from_mode_session(
                            "isp_umap_postprocess_n_clusters_mode",
                            "isp_umap_postprocess_n_clusters",
                        )
                    ),
                ]
            )
            if bool(
                st.session_state.get(
                    "isp_umap_postprocess_celltype", _DEFAULT_ISP_POSTPROCESS_CELLTYPE
                )
            ):
                cmd.append("--postprocess-celltype")
            else:
                cmd.append("--no-postprocess-celltype")
        else:
            cmd.append("--skip-postprocess")
        env["ISP_UMAP_CONFIG"] = cfg
    elif run_label == RUN_TYPE_SEQUENTIAL_ISP:
        cmd = ["python3", str(CORE / "run_sequential_isp.py"), "--config", cfg]
        env["SEQUENTIAL_ISP_CONFIG"] = cfg
    elif run_label == RUN_TYPE_PIPELINE:
        cmd = ["python3", str(CORE / "run_pipeline.py"), "--config", cfg]
        env["PIPELINE_CONFIG"] = cfg
        env.setdefault("ISP_NUM_GPUS", os.environ.get("ISP_NUM_GPUS", "1"))
    elif run_label == RUN_TYPE_FT_BATCH:
        cmd = ["python3", str(CORE / "run_ft_batch_calibrate.py"), "--config", cfg]
        env["FT_BATCH_CALIBRATE_CONFIG"] = cfg
    else:
        raise ValueError(run_label)
    return cmd, env


def _guess_output_roots(run_label: str, cfg: dict) -> list[Path]:
    roots: list[Path] = []
    if run_label == RUN_TYPE_ISP_UMAP:
        source = str(st.session_state.get("isp_umap_source_run_dir") or "").strip()
        if source:
            roots.append(Path(source))
            roots.append(Path(source) / "isp_umap")
        roots.append(ROOT / "output")
    elif run_label == RUN_TYPE_SEQUENTIAL_ISP:
        source = str(st.session_state.get("seq_isp_source_run_dir") or "").strip()
        if source:
            roots.append(Path(source) / "sequential_isp")
            roots.append(Path(source))
        roots.append(ROOT / "output")
    elif run_label in (RUN_TYPE_PIPELINE, RUN_TYPE_FT_BATCH):
        paths = cfg.get("paths") or {}
        out_root = paths.get("output_root")
        if out_root:
            roots.append(Path(str(out_root)))
        if run_label == RUN_TYPE_PIPELINE:
            roots.extend(_latest_pipeline_run_dirs(paths.get("output_root") or "/app/output"))
    return roots

_FIGURE_SUFFIXES = frozenset({".png", ".pdf", ".svg", ".jpg", ".jpeg", ".webp"})


def _list_figure_files(directory: Path) -> list[Path]:
    directory = directory.resolve()
    if not directory.is_dir():
        return []
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in _FIGURE_SUFFIXES
    )


def _discover_figures_dirs(roots: list[str | Path], run_label: str) -> list[Path]:
    """Find figures/ (or ISP UMAP PNGs) under recent run folders."""
    found: list[Path] = []
    seen: set[str] = set()

    def add(fig_dir: Path) -> None:
        key = str(fig_dir.resolve())
        if key in seen:
            return
        if _list_figure_files(fig_dir):
            seen.add(key)
            found.append(fig_dir)

    for raw in roots:
        root = Path(raw)
        if not root.is_dir():
            continue
        if root.name.startswith(("pipeline_", "isp_", "sequential_isp_")):
            add(root / "figures")
            if run_label == RUN_TYPE_ISP_UMAP:
                umaps = [p for p in root.glob("umap_*.png") if p.is_file()]
                if umaps:
                    add(root)
            continue
        for fig_dir in root.rglob("figures"):
            if fig_dir.is_dir() and fig_dir.parent.name.startswith(
                ("pipeline_", "isp_", "finetune_")
            ):
                add(fig_dir)
        if run_label == RUN_TYPE_ISP_UMAP:
            for run_dir in sorted(
                root.glob("**/isp_umap_*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:3]:
                if run_dir.is_dir() and list(run_dir.glob("umap_*.png")):
                    add(run_dir)

    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found


def _build_figures_zip(fig_dirs: list[Path]) -> tuple[bytes, str] | None:
    """Zip all figure files; arcnames include run folder for clarity."""
    entries: list[tuple[Path, str]] = []
    for fig_dir in fig_dirs:
        run_name = fig_dir.parent.name if fig_dir.name == "figures" else fig_dir.name
        prefix = f"{run_name}/"
        for fp in _list_figure_files(fig_dir):
            try:
                rel = fp.relative_to(fig_dir)
            except ValueError:
                rel = Path(fp.name)
            entries.append((fp, f"{prefix}{rel.as_posix()}"))

    if not entries:
        return None

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp, arcname in entries:
            zf.write(fp, arcname=arcname)

    primary = fig_dirs[0].parent.name if fig_dirs[0].name == "figures" else fig_dirs[0].name
    return buf.getvalue(), f"{primary}_figures.zip"


def _latest_pipeline_run_dirs(output_root: str | Path, limit: int = 5) -> list[Path]:
    """Newest pipeline_* folders under {output_root}/{YYYYMMDD}/."""
    root = Path(str(output_root))
    if not root.is_dir():
        return []
    found: list[Path] = []
    for date_dir in sorted(root.iterdir(), key=lambda p: p.name, reverse=True):
        if not date_dir.is_dir() or len(date_dir.name) != 8 or not date_dir.name.isdigit():
            continue
        for run_dir in date_dir.glob("pipeline_*"):
            if run_dir.is_dir():
                found.append(run_dir)
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found[:limit]


def _isp_umap_search_roots() -> list[Path]:
    """Output roots to scan for completed pipeline ISP runs."""
    roots: list[Path] = [ROOT / "output"]
    # Prefer path from YAML when editing Pipeline config; ISP UMAP yaml may not have it.
    try:
        paths = (_read_pipeline_yaml().get("paths") or {})
        out = paths.get("output_root")
        if out:
            roots.insert(0, Path(str(out)))
    except Exception:
        pass
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[Path] = []
    for r in roots:
        key = str(r.resolve()) if r.exists() else str(r)
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


def _discover_isp_pipeline_runs(limit: int = 40) -> list[Path]:
    """Newest pipeline_* dirs that have stage_configs/isp.yaml (ISP stage was prepared)."""
    found: list[Path] = []
    seen: set[str] = set()
    for root in _isp_umap_search_roots():
        if not root.is_dir():
            continue
        try:
            candidates = root.rglob("stage_configs/isp.yaml")
        except OSError:
            continue
        for isp_cfg in candidates:
            run_dir = isp_cfg.parent.parent
            if not run_dir.name.startswith("pipeline_"):
                continue
            key = str(run_dir.resolve())
            if key in seen:
                continue
            seen.add(key)
            found.append(run_dir)
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found[:limit]


def _load_isp_stage_cfg(run_dir: Path) -> dict:
    path = run_dir / "stage_configs" / "isp.yaml"
    if not path.is_file():
        return {}
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(cfg, dict):
        return {}
    return _remap_isp_stage_paths_to_run_dir(cfg, run_dir)


def _resolve_path_under_pipeline_run(path: str | Path, run_dir: Path) -> Path:
    """If a cluster absolute path is missing, map the suffix after pipeline_* onto run_dir."""
    run_dir = Path(run_dir).expanduser().resolve()
    original = Path(path)
    if original.exists():
        return original
    parts = original.parts
    if run_dir.name in parts:
        idx = parts.index(run_dir.name)
        candidate = run_dir.joinpath(*parts[idx + 1 :])
        if candidate.exists():
            return candidate
    return original


def _remap_isp_stage_paths_to_run_dir(cfg: dict, run_dir: Path) -> dict:
    """Rewrite dataset/model paths from synced Pegasus absolute paths onto this host."""
    out = dict(cfg)
    paths = dict(out.get("paths") or {})
    run_dir = Path(run_dir).expanduser().resolve()
    for key in ("dataset", "geneformer_model", "output_root"):
        raw = paths.get(key)
        if not raw:
            continue
        paths[key] = str(_resolve_path_under_pipeline_run(raw, run_dir))
    model = Path(str(paths.get("geneformer_model") or ""))
    if paths.get("geneformer_model") and not model.is_dir():
        guess = run_dir / "finetune" / "all_run1"
        if guess.is_dir():
            paths["geneformer_model"] = str(guess)
    dataset = Path(str(paths.get("dataset") or ""))
    if paths.get("dataset") and not dataset.exists():
        by_name = run_dir / "tokenized_dataset" / Path(str(paths["dataset"])).name
        if by_name.exists():
            paths["dataset"] = str(by_name)
        else:
            candidates = sorted((run_dir / "tokenized_dataset").glob("*.dataset"))
            if len(candidates) == 1:
                paths["dataset"] = str(candidates[0])
    if not paths.get("output_root") or not Path(str(paths["output_root"])).is_dir():
        paths["output_root"] = str(run_dir)
    out["paths"] = paths
    return out


def _isp_run_gene_suggestions(run_dir: Path, limit: int = 30) -> list[str]:
    """Gene symbols from ISP config and/or ispstats CSVs."""
    import csv

    genes: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        g = str(name or "").strip()
        if not g or g in seen:
            return
        seen.add(g)
        genes.append(g)

    cfg = _load_isp_stage_cfg(run_dir)
    for g in (cfg.get("perturbation") or {}).get("genes_to_perturb") or []:
        add(str(g))

    stats_dir = run_dir / "ispstats_results"
    for fname in (
        "significant_genes.csv",
        "top100_positive_shifters.csv",
        "top100_negative_shifters.csv",
    ):
        csv_path = stats_dir / fname
        if not csv_path.is_file():
            continue
        try:
            with csv_path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    add(row.get("Gene_name") or row.get("gene_name") or "")
                    if len(genes) >= limit:
                        return genes
        except (OSError, csv.Error, UnicodeDecodeError):
            continue
    return genes


def _format_isp_run_label(run_dir: Path) -> str:
    try:
        rel = run_dir.resolve().relative_to((ROOT / "output").resolve())
        label = rel.as_posix()
    except ValueError:
        label = str(run_dir)
    flags: list[str] = []
    if (run_dir / "isp_results").is_dir():
        flags.append("isp")
    if (run_dir / "ispstats_results").is_dir():
        flags.append("stats")
    if (run_dir / "isp_umap").is_dir():
        flags.append("umap")
    cfg = _load_isp_stage_cfg(run_dir)
    pert = cfg.get("perturbation") or {}
    start = pert.get("start_state") or "?"
    end = pert.get("end_state") or "?"
    suffix = f" [{', '.join(flags)}]" if flags else ""
    return f"{label}  ({start}→{end}){suffix}"


def _isp_umap_pca_components_and_seed() -> tuple[int, int]:
    """Map WebUI position-method widget to ``umap.pca_components`` and ``umap.seed``."""
    method = st.session_state.get("isp_umap_position_method", ISP_UMAP_POSITION_DIRECT)
    if method == ISP_UMAP_POSITION_PCA50:
        return ISP_UMAP_PCA_COMPONENTS, ISP_UMAP_PCA_UMAP_SEED
    return 0, ISP_UMAP_DIRECT_UMAP_SEED


def _normalize_isp_umap_genes(genes: str | list[str] | None) -> list[str]:
    if isinstance(genes, list):
        return _parse_genes_to_perturb_text("\n".join(str(g).strip() for g in genes if str(g).strip()))
    return _parse_genes_to_perturb_text(genes)


def _build_isp_umap_yaml_from_pipeline_run(
    run_dir: Path, genes: str | list[str]
) -> tuple[str | None, str | None]:
    """Build isp_umap YAML from a pipeline ISP stage + gene list. Returns (yaml_text, error)."""
    gene_list = _normalize_isp_umap_genes(genes)
    if not gene_list:
        return None, "Choose at least one gene to perturb for the trajectory UMAP."
    cfg = _load_isp_stage_cfg(run_dir)
    if not cfg:
        return None, f"Missing stage_configs/isp.yaml under `{run_dir}`."
    paths = cfg.get("paths") or {}
    dataset = paths.get("dataset")
    model = paths.get("geneformer_model")
    if not dataset or not model:
        return None, "ISP stage config is missing paths.dataset or paths.geneformer_model."
    pert = cfg.get("perturbation") or {}
    runtime = cfg.get("runtime") or {}
    model_block = cfg.get("model") or {}
    umap_block = dict(cfg.get("umap") or {})
    show_arrows = bool(
        st.session_state.get(
            "isp_umap_show_trajectories",
            umap_block.get("show_trajectory_arrows", True),
        )
    )
    num_arrows = int(
        st.session_state.get(
            "isp_umap_num_arrows",
            umap_block.get("num_trajectory_arrows", 100),
        )
    )
    pca_components, umap_seed = _isp_umap_pca_components_and_seed()
    # Standalone isp_umap.yaml uses umap.* + runtime.batch_size (not stages.isp.umap.enabled).
    umap_out = {
        "seed": umap_seed,
        "n_neighbors": int(umap_block.get("n_neighbors", 15)),
        "min_dist": float(umap_block.get("min_dist", 0.1)),
        "pca_components": pca_components,
        "show_trajectory_arrows": show_arrows,
        "num_trajectory_arrows": max(0, num_arrows),
        "max_cells_per_state": int(umap_block.get("max_cells_per_state", 2000)),
        "sample_key": umap_block.get("sample_key", "sample_id"),
    }
    pert_type = str(pert.get("type") or "delete")
    gene_label = "+".join(gene_list)
    out = {
        "paths": {
            "dataset": str(dataset),
            "geneformer_model": str(model),
        },
        "umap": umap_out,
        "perturbation": {
            "genes_to_perturb": gene_list,
            "gene_label": gene_label,
            "type": pert_type,
            "state_key": pert.get("state_key", "disease"),
            "start_state": pert.get("start_state"),
            "end_state": pert.get("end_state"),
        },
        "runtime": {
            "num_classes": int(model_block.get("num_classes", runtime.get("num_classes", 2))),
            "batch_size": int(
                umap_block.get("batch_size", runtime.get("batch_size", 100))
            ),
        },
        "species": dict(cfg.get("species") or {}),
    }
    post_block = dict(cfg.get("postprocess") or {})
    post_enabled = bool(
        st.session_state.get(
            "isp_umap_postprocess_enabled",
            post_block.get("enabled", _DEFAULT_ISP_POSTPROCESS_ENABLED),
        )
    )
    if "isp_umap_postprocess_n_clusters_mode" in st.session_state:
        post_n_clusters = _n_clusters_from_mode_session(
            "isp_umap_postprocess_n_clusters_mode",
            "isp_umap_postprocess_n_clusters",
        )
    else:
        post_n_clusters = _normalize_n_clusters_value(
            post_block.get("n_clusters", _DEFAULT_ISP_POSTPROCESS_N_CLUSTERS)
        )
    post_celltype = bool(
        st.session_state.get(
            "isp_umap_postprocess_celltype",
            post_block.get("celltype_prediction", _DEFAULT_ISP_POSTPROCESS_CELLTYPE),
        )
    )
    out["postprocess"] = {
        "enabled": post_enabled,
        "n_clusters": post_n_clusters,
        "celltype_prediction": post_celltype,
        "prefer_metadata_celltype": bool(
            post_block.get(
                "prefer_metadata_celltype",
                False,
            )
        ),
        "celltype_rank_weights": bool(
            post_block.get(
                "celltype_rank_weights",
                True,
            )
        ),
    }
    return (
        yaml.dump(out, default_flow_style=False, sort_keys=False, allow_unicode=True),
        None,
    )


def _apply_isp_umap_from_pipeline_run(run_dir: Path, genes: str | list[str]) -> str | None:
    """Fill isp_umap.yaml editor from a pipeline ISP stage + chosen genes. Returns error or None.

    Must run before the ``yaml_editor`` widget is instantiated in the same script run
    (e.g. the Apply button in column 1). For Run job, use
    ``_build_isp_umap_yaml_from_pipeline_run`` and pass the text without writing the widget key.
    """
    yaml_text, err = _build_isp_umap_yaml_from_pipeline_run(run_dir, genes)
    if err:
        return err
    gene_list = _normalize_isp_umap_genes(genes)
    st.session_state["yaml_editor"] = yaml_text
    st.session_state["isp_umap_source_run_dir"] = str(run_dir.resolve())
    st.session_state["isp_umap_genes_text"] = _genes_to_perturb_as_text(gene_list)
    st.session_state["isp_umap_gene"] = gene_list[0] if len(gene_list) == 1 else ""
    return None


def _render_isp_umap_plot_options() -> None:
    """ISP UMAP plot knobs (position method, trajectory arrows) — shown in Run type column."""
    st.markdown("**Plot options**")
    st.caption(
        "Figures use the same **white background and L-shaped axes** as Fig.2 endpoint UMAP "
        "(UMAP-1 / UMAP-2; no grid)."
    )
    st.session_state.setdefault("isp_umap_position_method", ISP_UMAP_POSITION_DIRECT)
    st.radio(
        "UMAP position method",
        [ISP_UMAP_POSITION_DIRECT, ISP_UMAP_POSITION_PCA50],
        key="isp_umap_position_method",
        help=(
            "Direct UMAP: fastest (GPU when available); layout uses all embedding dimensions. "
            "PCA(50)→UMAP: slower (PCA + CPU UMAP); scatter layout usually changes. "
            "PCA(50) + seed 0 matches Fig.2/3/4 manuscript coordinates."
        ),
    )
    if st.session_state.get("isp_umap_position_method") == ISP_UMAP_POSITION_PCA50:
        st.caption(
            "PCA(50)→UMAP: **slower** than direct. Point positions **change** "
            "(50-D compression before UMAP; different seed). Matches Fig.2/3/4 when seed is 0."
        )
    else:
        st.caption("Direct UMAP: **fastest** option (GPU when available).")
    st.session_state.setdefault("isp_umap_show_trajectories", True)
    st.session_state.setdefault("isp_umap_num_arrows", 100)
    st.checkbox(
        "Draw trajectory lines",
        key="isp_umap_show_trajectories",
        help=(
            "Navy arrows from each start-state cell to its perturbed position "
            "(`umap.show_trajectory_arrows`). Off = scatter only."
        ),
    )
    if st.session_state.get("isp_umap_show_trajectories"):
        st.number_input(
            "Number of trajectory arrows",
            min_value=1,
            max_value=5000,
            step=10,
            key="isp_umap_num_arrows",
            help="Approximate count of start → perturbed arrows (`umap.num_trajectory_arrows`).",
        )
    else:
        st.caption("Scatter points only (no Start → Perturbed arrows).")

    with st.expander("Cluster / cell-type analysis", expanded=False):
        st.caption(
            "Optional post-UMAP outputs under `cluster_coexpr_analysis/` "
            "(`postprocess.*`). Default **off** — enable only when you need joint "
            "L2 / cluster / cell-type overlays and cell-type trajectory tracking."
        )
        st.session_state.setdefault(
            "isp_umap_postprocess_enabled", _DEFAULT_ISP_POSTPROCESS_ENABLED
        )
        st.session_state.setdefault(
            "isp_umap_postprocess_n_clusters_mode",
            _DEFAULT_ISP_POSTPROCESS_N_CLUSTERS_MODE,
        )
        st.session_state.setdefault(
            "isp_umap_postprocess_n_clusters", _DEFAULT_ISP_POSTPROCESS_N_CLUSTERS
        )
        st.session_state.setdefault(
            "isp_umap_postprocess_celltype", _DEFAULT_ISP_POSTPROCESS_CELLTYPE
        )
        st.checkbox(
            "Enable cluster_coexpr_analysis",
            key="isp_umap_postprocess_enabled",
            help="Write joint UMAP + L2-by-group figures (`postprocess.enabled`).",
        )
        if st.session_state.get("isp_umap_postprocess_enabled"):
            st.radio(
                "n_clusters (KMeans)",
                options=["auto", "manual"],
                format_func=lambda m: (
                    "auto (silhouette)" if m == "auto" else "manual (specify K)"
                ),
                horizontal=True,
                key="isp_umap_postprocess_n_clusters_mode",
                help=(
                    "auto: pick K by max silhouette on start embeddings (k=2..15). "
                    "manual: use the number below (`postprocess.n_clusters`)."
                ),
            )
            if st.session_state.get("isp_umap_postprocess_n_clusters_mode") == "manual":
                st.number_input(
                    "K",
                    min_value=2,
                    max_value=50,
                    step=1,
                    key="isp_umap_postprocess_n_clusters",
                    help="Fixed KMeans cluster count (`postprocess.n_clusters`).",
                )
            st.checkbox(
                "Cell-type prediction (marker genes)",
                key="isp_umap_postprocess_celltype",
                help=(
                    "Species-aware marker panels on start-state input_ids "
                    "(`postprocess.celltype_prediction`). Prefers dataset "
                    "`cell_type` metadata when present; rank-weights marker hits."
                ),
            )


def _render_isp_umap_source_picker() -> None:
    """Column-1 UI for ISP UMAP: pick a past pipeline ISP run + gene (not raw 10x upload)."""
    st.subheader("ISP source run")
    st.caption(
        "Select a completed **Pipeline (E2E)** folder that has ISP "
        "(`stage_configs/isp.yaml`). Trajectory UMAP reuses that run’s tokenized "
        "dataset and fine-tuned model — not a fresh Data input zip."
    )
    runs = _discover_isp_pipeline_runs()
    if not runs:
        st.warning(
            "No pipeline ISP runs found under `/app/output` "
            "(looking for `**/pipeline_*/stage_configs/isp.yaml`). "
            "Finish a **Pipeline (E2E)** job first."
        )
        return

    labels = {_format_isp_run_label(p): str(p.resolve()) for p in runs}
    options = list(labels.keys())
    current_path = str(st.session_state.get("isp_umap_source_run_dir") or "")
    default_idx = 0
    for i, lab in enumerate(options):
        if labels[lab] == current_path:
            default_idx = i
            break

    chosen_label = st.selectbox(
        "Past ISP / pipeline run",
        options,
        index=default_idx,
        key="isp_umap_run_select_label",
        help="Pipeline run directories that include an ISP stage config.",
    )
    run_dir = Path(labels[chosen_label])
    st.session_state["isp_umap_source_run_dir"] = str(run_dir.resolve())
    st.code(str(run_dir), language="text")

    cfg = _load_isp_stage_cfg(run_dir)
    pert = cfg.get("perturbation") or {}
    paths = cfg.get("paths") or {}
    targeted = [str(g).strip() for g in (pert.get("genes_to_perturb") or []) if str(g).strip()]
    suggestions = _isp_run_gene_suggestions(run_dir)
    pert_type = str(pert.get("type") or "delete")
    if targeted:
        st.caption(
            f"Targeted ISP genes in config: **{', '.join(targeted)}** · "
            f"perturbation.type=`{pert_type}` (group KD/OE applied together)"
        )
    else:
        st.caption(
            "This ISP was **genome-wide** (`genes_to_perturb` empty). "
            "Enter gene(s) below (suggestions from ispstats when available)."
        )
    if paths.get("dataset"):
        st.caption(f"dataset: `{paths.get('dataset')}`")
    if paths.get("geneformer_model"):
        st.caption(f"model: `{paths.get('geneformer_model')}`")

    default_genes_text = _genes_to_perturb_as_text(targeted) if targeted else ""
    if not st.session_state.get("isp_umap_genes_text") and default_genes_text:
        st.session_state["isp_umap_genes_text"] = default_genes_text

    st.text_area(
        "Genes to perturb (one per line; group KD/OE)",
        key="isp_umap_genes_text",
        height=120,
        placeholder="Oct4\nSox2\nKlf4\nMyc",
        help=(
            "All listed genes are perturbed together in each cell (`genes_to_perturb`). "
            "Uses perturbation.type from the selected ISP run: delete = KD, overexpress = OE."
        ),
    )
    if suggestions:
        st.caption(f"Suggestions from ispstats: {', '.join(suggestions[:12])}{'…' if len(suggestions) > 12 else ''}")

    selected_genes = _normalize_isp_umap_genes(st.session_state.get("isp_umap_genes_text"))
    st.session_state["isp_umap_gene"] = selected_genes[0] if len(selected_genes) == 1 else ""

    if st.button("Apply run + genes to Config YAML", type="secondary", key="isp_umap_apply_btn"):
        err = _apply_isp_umap_from_pipeline_run(run_dir, selected_genes)
        if err:
            st.error(err)
        else:
            st.success(
                f"Filled ISP UMAP config from `{run_dir.name}` · "
                f"{len(selected_genes)} gene(s) · type=`{pert_type}`."
            )
            st.rerun()


def _seq_isp_collect_steps() -> list[dict]:
    """Read sequential-step widgets into YAML-ready step dicts."""
    n = int(st.session_state.get("seq_isp_n_steps") or _DEFAULT_SEQ_ISP_STEPS)
    n = max(1, min(_MAX_SEQ_ISP_STEPS, n))
    steps: list[dict] = []
    for i in range(n):
        ptype = str(st.session_state.get(f"seq_isp_step_{i}_type") or "overexpress")
        if ptype not in _SEQ_ISP_TYPE_LABELS:
            ptype = "overexpress"
        genes = _parse_genes_to_perturb_text(st.session_state.get(f"seq_isp_step_{i}_genes"))
        name = str(st.session_state.get(f"seq_isp_step_{i}_name") or "").strip()
        step: dict = {"type": ptype, "genes": genes}
        if name:
            step["name"] = name
        steps.append(step)
    return steps


def _seq_isp_selected_batch_size() -> int | str:
    if st.session_state.get("seq_isp_batch_mode", BATCH_MODE_AUTO) == BATCH_MODE_MANUAL:
        return int(st.session_state.get("seq_isp_batch_size", DEFAULT_MANUAL_BATCH_SIZE))
    return "auto"


def _build_sequential_isp_yaml_from_pipeline_run(
    run_dir: Path,
) -> tuple[str | None, str | None]:
    """Build sequential ISP YAML from a pipeline ISP stage + step widgets."""
    steps = _seq_isp_collect_steps()
    if not steps or any(not s.get("genes") for s in steps):
        return None, "Each sequential step needs at least one gene (symbol or Ensembl ID)."
    cfg = _load_isp_stage_cfg(run_dir)
    if not cfg:
        return None, f"Missing stage_configs/isp.yaml under `{run_dir}`."
    paths = cfg.get("paths") or {}
    dataset = paths.get("dataset")
    model = paths.get("geneformer_model")
    if not dataset or not model:
        return None, "ISP stage config is missing paths.dataset or paths.geneformer_model."
    pert = cfg.get("perturbation") or {}
    runtime = cfg.get("runtime") or {}
    model_block = cfg.get("model") or {}
    isp_block = cfg.get("isp") or {}
    max_ncells = int(
        st.session_state.get("seq_isp_max_ncells")
        or isp_block.get("max_ncells")
        or _DEFAULT_ISP_MAX_NCELLS
    )
    save_ds = bool(st.session_state.get("seq_isp_save_datasets", False))
    out = {
        "paths": {
            "dataset": str(dataset),
            "geneformer_model": str(model),
            "output_root": str((run_dir / "sequential_isp").resolve()),
            "output_time_subdir": False,
            "output_date_subdir": False,
        },
        "species": dict(cfg.get("species") or {}),
        "perturbation": {
            "state_key": pert.get("state_key", "disease"),
            "start_state": pert.get("start_state"),
            "end_state": pert.get("end_state"),
            "alt_states": list(pert.get("alt_states") or []),
        },
        "model": {
            "type": model_block.get("type", "CellClassifier"),
            "num_classes": int(model_block.get("num_classes", runtime.get("num_classes", 2))),
        },
        "isp": {
            "max_ncells": max_ncells,
            "emb_layer": int(isp_block.get("emb_layer", 0)),
        },
        "sequential": {
            "save_intermediate_datasets": save_ds,
            "steps": steps,
        },
        "runtime": {
            "forward_batch_size": _seq_isp_selected_batch_size(),
            "nproc": int(runtime.get("nproc", 8)),
        },
    }
    return (
        yaml.dump(out, default_flow_style=False, sort_keys=False, allow_unicode=True),
        None,
    )


def _apply_sequential_isp_from_pipeline_run(run_dir: Path) -> str | None:
    yaml_text, err = _build_sequential_isp_yaml_from_pipeline_run(run_dir)
    if err:
        return err
    st.session_state["yaml_editor"] = yaml_text
    st.session_state["seq_isp_source_run_dir"] = str(run_dir.resolve())
    return None


def _render_sequential_isp_source_picker() -> None:
    """Column-1 UI: pick a past pipeline ISP run (dataset + FT model + states)."""
    st.subheader("ISP source run")
    st.caption(
        "Select a completed **Pipeline (E2E)** folder that has ISP "
        "(`stage_configs/isp.yaml`). Sequential ISP reuses that run’s tokenized "
        "dataset, fine-tuned model, and start/end states — not a fresh Data input zip."
    )
    runs = _discover_isp_pipeline_runs()
    if not runs:
        st.warning(
            "No pipeline ISP runs found under `/app/output` "
            "(looking for `**/pipeline_*/stage_configs/isp.yaml`). "
            "Finish a **Pipeline (E2E)** job first."
        )
        return

    labels = {_format_isp_run_label(p): str(p.resolve()) for p in runs}
    options = list(labels.keys())
    current_path = str(st.session_state.get("seq_isp_source_run_dir") or "")
    default_idx = 0
    for i, lab in enumerate(options):
        if labels[lab] == current_path:
            default_idx = i
            break

    chosen_label = st.selectbox(
        "Past ISP / pipeline run",
        options,
        index=default_idx,
        key="seq_isp_run_select_label",
        help="Pipeline run directories that include an ISP stage config.",
    )
    run_dir = Path(labels[chosen_label])
    st.session_state["seq_isp_source_run_dir"] = str(run_dir.resolve())
    st.code(str(run_dir), language="text")

    cfg = _load_isp_stage_cfg(run_dir)
    pert = cfg.get("perturbation") or {}
    paths = cfg.get("paths") or {}
    st.caption(
        f"ISP states: **{pert.get('start_state', '?')} → {pert.get('end_state', '?')}** "
        f"(state_key=`{pert.get('state_key', 'disease')}`)"
    )
    if paths.get("dataset"):
        st.caption(f"dataset: `{paths.get('dataset')}`")
    if paths.get("geneformer_model"):
        st.caption(f"model: `{paths.get('geneformer_model')}`")


def _render_sequential_isp_controls() -> None:
    """Run-type column: ordered OE / KD steps + batch size."""
    st.info(
        "Chains **group OE and/or KD** on start-state cells. Each step is applied "
        "on the previous step’s rank encoding (token space), then scored as "
        "`goal_state_shift` toward the pipeline end state. Outputs go under "
        "`{pipeline_run}/sequential_isp/`."
    )
    st.caption(
        "OE = length-preserving move-to-front (later OE genes sit leftmost). "
        "KD = delete those genes from the encoding. Mixed OE+KD is allowed."
    )
    st.session_state.setdefault("seq_isp_n_steps", _DEFAULT_SEQ_ISP_STEPS)
    st.session_state.setdefault("seq_isp_save_datasets", False)
    st.session_state.setdefault("seq_isp_max_ncells", _DEFAULT_ISP_MAX_NCELLS)
    st.session_state.setdefault("seq_isp_batch_mode", BATCH_MODE_AUTO)
    st.session_state.setdefault("seq_isp_batch_size", DEFAULT_MANUAL_BATCH_SIZE)

    st.number_input(
        "Number of sequential steps",
        min_value=1,
        max_value=_MAX_SEQ_ISP_STEPS,
        step=1,
        key="seq_isp_n_steps",
        help="Each step is one group perturbation (one or more genes, OE or KD).",
    )
    n_steps = int(st.session_state.get("seq_isp_n_steps") or _DEFAULT_SEQ_ISP_STEPS)
    for i in range(n_steps):
        st.markdown(f"**Step {i + 1}**")
        st.session_state.setdefault(f"seq_isp_step_{i}_type", "overexpress")
        st.session_state.setdefault(f"seq_isp_step_{i}_name", "")
        st.session_state.setdefault(f"seq_isp_step_{i}_genes", "")
        c_type, c_name = st.columns(2)
        with c_type:
            st.selectbox(
                "type",
                list(_SEQ_ISP_TYPE_LABELS.keys()),
                key=f"seq_isp_step_{i}_type",
                format_func=lambda v: _SEQ_ISP_TYPE_LABELS.get(v, v),
            )
        with c_name:
            st.text_input(
                "name (optional)",
                key=f"seq_isp_step_{i}_name",
                placeholder=f"step{i + 1:02d}",
            )
        st.text_area(
            "genes (one per line; group)",
            key=f"seq_isp_step_{i}_genes",
            height=80,
            placeholder="Pou5f1\nSox2\nKlf4\nMyc",
        )

    st.number_input(
        "max_ncells (start-state cells)",
        min_value=1,
        max_value=100_000,
        step=100,
        key="seq_isp_max_ncells",
    )
    st.checkbox(
        "Save intermediate perturbed datasets",
        key="seq_isp_save_datasets",
        help="Off by default (host RAM / disk). Needed only if you will plot UMAP from a step.",
    )
    c_mode, c_size = st.columns(2)
    with c_mode:
        st.radio(
            "GPU batch size",
            [BATCH_MODE_AUTO, BATCH_MODE_MANUAL],
            key="seq_isp_batch_mode",
            horizontal=True,
            help=(
                "Auto measures this GPU with a dual-forward probe matching sequential "
                "scoring (does not reuse genome-wide ISP’s 1-forward cache)."
            ),
        )
    with c_size:
        if st.session_state.get("seq_isp_batch_mode") == BATCH_MODE_MANUAL:
            st.number_input(
                "forward_batch_size",
                min_value=1,
                max_value=4096,
                step=8,
                key="seq_isp_batch_size",
            )
        else:
            st.caption("Measured at startup for sequential dual-forward scoring, then cached.")

    source = st.session_state.get("seq_isp_source_run_dir")
    steps = _seq_isp_collect_steps()
    ready_steps = bool(steps) and all(s.get("genes") for s in steps)
    if source and ready_steps:
        labels = [f"{s['type']}:{'+'.join(s['genes'][:3])}" for s in steps]
        st.success(f"Ready: `{Path(source).name}` · {' → '.join(labels)}")
    elif source:
        st.warning("Enter genes for every sequential step.")
    else:
        st.warning("Select a past ISP / pipeline run in the left column.")

    if st.button("Apply run + steps to Config YAML", type="secondary", key="seq_isp_apply_btn"):
        run_dir = Path(str(source or ""))
        err = _apply_sequential_isp_from_pipeline_run(run_dir) if run_dir.is_dir() else (
            "Select a past Pipeline ISP run in the left column."
        )
        if err:
            st.error(err)
        else:
            st.success(f"Filled Sequential ISP config from `{run_dir.name}`.")
            st.rerun()


def _read_pipeline_yaml() -> dict:
    text = st.session_state.get("yaml_editor", "")
    try:
        cfg = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return {}
    return cfg if isinstance(cfg, dict) else {}


def _read_pipeline_data_fields() -> tuple[str, str | None]:
    """Return (input_dir, output_prefix) from the YAML editor; output_prefix None if unset/null."""
    cfg = _read_pipeline_yaml()
    data = cfg.get("data") or {}
    input_dir = str(data.get("input_dir") or "")
    raw_prefix = data.get("output_prefix")
    if raw_prefix is None or str(raw_prefix).strip().lower() in ("", "null"):
        return input_dir, None
    return input_dir, str(raw_prefix).strip()


def _read_pipeline_perturbation() -> dict:
    return _read_pipeline_yaml().get("perturbation") or {}


BATCH_MODE_AUTO = "Auto"
BATCH_MODE_MANUAL = "Manual"
DEFAULT_MANUAL_BATCH_SIZE = 100


def _sync_batch_size_controls() -> None:
    """Initialize the ISP batch-size widgets from YAML; anything unparsable means Auto."""
    if "pipeline_batch_mode" in st.session_state:
        return
    raw = (_read_pipeline_yaml().get("runtime") or {}).get("forward_batch_size", "auto")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = None
    st.session_state["pipeline_batch_mode"] = (
        BATCH_MODE_AUTO if value is None else BATCH_MODE_MANUAL
    )
    st.session_state["pipeline_batch_size"] = value or DEFAULT_MANUAL_BATCH_SIZE


def _selected_forward_batch_size() -> int | str:
    if st.session_state.get("pipeline_batch_mode", BATCH_MODE_AUTO) == BATCH_MODE_MANUAL:
        return int(st.session_state.get("pipeline_batch_size", DEFAULT_MANUAL_BATCH_SIZE))
    return "auto"


def _apply_batch_size_to_yaml() -> None:
    _patch_pipeline_yaml(forward_batch_size=_selected_forward_batch_size())


def _sync_ft_train_batch_controls() -> None:
    """Initialize Pipeline fine-tune train_batch_size from YAML or last calibration."""
    if "pipeline_ft_train_batch_size" in st.session_state:
        return
    calibrated = st.session_state.get("calibrated_ft_batch_size")
    if isinstance(calibrated, int) and calibrated >= 1:
        st.session_state["pipeline_ft_train_batch_size"] = calibrated
        return
    raw = (_read_pipeline_yaml().get("runtime") or {}).get("train_batch_size")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_FT_TRAIN_BATCH_SIZE
    st.session_state["pipeline_ft_train_batch_size"] = max(1, value)


def _selected_ft_train_batch_size() -> int:
    return int(
        st.session_state.get("pipeline_ft_train_batch_size", DEFAULT_FT_TRAIN_BATCH_SIZE)
    )


def _apply_ft_train_batch_to_yaml() -> None:
    _patch_pipeline_yaml(train_batch_size=_selected_ft_train_batch_size())


def _apply_calibrated_ft_batch_to_pipeline() -> None:
    calibrated = st.session_state.get("calibrated_ft_batch_size")
    if not isinstance(calibrated, int) or calibrated < 1:
        return
    st.session_state["pipeline_ft_train_batch_size"] = calibrated
    _apply_ft_train_batch_to_yaml()


def _parse_ft_batch_from_log(text: str) -> int | None:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(FT_BATCH_RESULT_MARKER):
            raw = line.split("=", 1)[-1].strip()
            try:
                value = int(raw)
            except ValueError:
                continue
            if value >= 1:
                return value
    return None


def _ingest_ft_batch_calibrate_result() -> None:
    """After FT calibrate finishes, store recommended train_batch_size for Pipeline paste."""
    if st.session_state.get("run_type_sel") != RUN_TYPE_FT_BATCH:
        return
    if st.session_state.get("last_exit_code") not in (0, None):
        # Still try to parse; only skip hard failures below.
        pass
    candidates: list[Path] = []
    last_run = st.session_state.get("last_run_dir")
    if last_run:
        candidates.append(Path(last_run) / FT_BATCH_RESULT_FILENAME)
    log_path = st.session_state.get("active_log_path") or st.session_state.get("last_log_path")
    chosen: int | None = None
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        raw = payload.get("recommended_train_batch_size")
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value >= 1:
            chosen = value
            st.session_state["calibrated_ft_batch_result_path"] = str(path)
            break
    if chosen is None and log_path:
        try:
            chosen = _parse_ft_batch_from_log(Path(log_path).read_text(encoding="utf-8"))
        except OSError:
            chosen = None
    if chosen is None:
        return
    st.session_state["calibrated_ft_batch_size"] = chosen
    # Prefill Pipeline paste field for the next switch to Pipeline (E2E).
    st.session_state["pipeline_ft_train_batch_size"] = chosen


def _display_input_dir_line(upload_dir: Path) -> str:
    """Resolved data.input_dir for Run directory panel (YAML / session after Apply)."""
    cfg = _read_pipeline_yaml()
    from_yaml = str((cfg.get("data") or {}).get("input_dir") or "").strip()
    if from_yaml:
        return from_yaml
    cached = st.session_state.get("pipeline_tokenize_dir")
    if cached:
        return str(cached)
    tokenize_dir = _compute_tokenize_input_dir(upload_dir)
    if tokenize_dir:
        return str(tokenize_dir)
    return ""


def _display_output_root_line() -> str:
    """paths.output_root from Config YAML (refreshes after Apply / editor edits)."""
    paths = _read_pipeline_yaml().get("paths") or {}
    raw = paths.get("output_root")
    if raw is None or str(raw).strip().lower() in ("", "null"):
        return "/app/output"
    return str(raw).strip()


def _newest_pipeline_run_since(output_root: str | Path, since_ts: float) -> Path | None:
    """Newest pipeline_* folder under output_root created at or after since_ts (epoch)."""
    for run_dir in _latest_pipeline_run_dirs(output_root, limit=30):
        try:
            if run_dir.stat().st_mtime >= since_ts:
                return run_dir
        except OSError:
            continue
    return None


def _sync_pipeline_output_run_dir() -> None:
    """Set pipeline_output_run_dir after Run job creates DATE/pipeline_<UTC>/."""
    started = st.session_state.get("pipeline_job_started_ts")
    if started is None or st.session_state.get("run_type_sel") != RUN_TYPE_PIPELINE:
        return
    found = _newest_pipeline_run_since(_display_output_root_line(), float(started))
    if found is not None:
        st.session_state["pipeline_output_run_dir"] = str(found.resolve())


def _render_run_directory_input(upload_dir: Path, run_label: str) -> None:
    """data.input_dir block (Pipeline / FT calibrate; call after YAML editor)."""
    if run_label not in _STUDY_RUN_TYPES:
        return
    input_dir_line = _display_input_dir_line(upload_dir)
    if input_dir_line:
        st.code(f"data.input_dir (auto)\n{input_dir_line}", language="text")
    else:
        st.code("data.input_dir (auto)", language="text")


def _render_run_directory_output(run_label: str) -> None:
    """Pipeline / calibrate run folder after Run job (call after YAML editor)."""
    if run_label == RUN_TYPE_PIPELINE:
        pipeline_run = st.session_state.get("pipeline_output_run_dir")
        if pipeline_run:
            st.code(f"pipeline run folder\n{pipeline_run}", language="text")
        st.caption(
            "**Run job** creates `<DATE>/pipeline_<UTC>/` under `paths.output_root` in Config YAML. "
            "The run folder path appears here after the job starts."
        )
    elif run_label == RUN_TYPE_FT_BATCH:
        calibrated = st.session_state.get("calibrated_ft_batch_size")
        if isinstance(calibrated, int):
            st.code(
                f"recommended train_batch_size\n{calibrated}",
                language="text",
            )
        st.caption(
            "**Run job** measures this GPU (~1 min) and prints a recommended "
            "`runtime.train_batch_size`. Switch to **Pipeline (E2E)** and paste it there."
        )
    elif run_label == RUN_TYPE_SEQUENTIAL_ISP:
        source = st.session_state.get("seq_isp_source_run_dir")
        if source:
            st.code(f"Sequential ISP source run\n{source}", language="text")
        steps = _seq_isp_collect_steps()
        if steps:
            lines = []
            for i, s in enumerate(steps, start=1):
                genes = "+".join(s.get("genes") or []) or "(no genes)"
                lines.append(f"{i}. {s.get('type', 'overexpress')} {genes}")
            st.code("steps\n" + "\n".join(lines), language="text")
        st.caption(
            "**Run job** writes per-step `goal_state_shift` CSVs under "
            "`<selected pipeline run>/sequential_isp/`."
        )
    else:
        source = st.session_state.get("isp_umap_source_run_dir")
        genes = _normalize_isp_umap_genes(st.session_state.get("isp_umap_genes_text"))
        if source:
            st.code(f"ISP source run\n{source}", language="text")
        if genes:
            st.code(f"genes_to_perturb\n{_genes_to_perturb_as_text(genes)}", language="text")
        st.caption(
            "**Run job** writes trajectory UMAP under "
            "`<selected pipeline run>/isp_umap/` (via `--run-dir`)."
        )


def _default_isp_start_end(states: list[str]) -> tuple[str, str]:
    if not states:
        return "Disease", "Ctrl"
    if len(states) == 1:
        return states[0], states[0]
    if "Disease" in states and "Ctrl" in states:
        return "Disease", "Ctrl"
    return states[0], states[1]


def _sync_isp_state_selectors(
    states: list[str] | None = None,
    *,
    force: bool = False,
) -> None:
    """Initialize ISP dropdown options; set values only if unset or force=True (new zip)."""
    pert = _read_pipeline_perturbation()
    detected = states if states is not None else st.session_state.get("pipeline_detected_states") or []
    options = sorted({str(s) for s in detected if s} | {str(pert.get("start_state") or "")} | {str(pert.get("end_state") or "")} - {""})
    if not options:
        options = ["Disease", "Ctrl"]

    default_start, default_end = _default_isp_start_end(detected)
    start_val = str(pert.get("start_state") or default_start)
    end_val = str(pert.get("end_state") or default_end)
    if start_val not in options:
        options = sorted(set(options) | {start_val})
    if end_val not in options:
        options = sorted(set(options) | {end_val})

    st.session_state["pipeline_isp_state_options"] = options
    if force or "pipeline_isp_start_state" not in st.session_state:
        st.session_state["pipeline_isp_start_state"] = start_val
    if force or "pipeline_isp_end_state" not in st.session_state:
        st.session_state["pipeline_isp_end_state"] = end_val


def _zip_upload_fingerprint(uploaded_file) -> str:
    return f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"


def _raw_study_name() -> str:
    return str(st.session_state.get("pipeline_study_name") or "").strip()


def _compute_tokenize_input_dir(upload_dir: Path) -> Path | None:
    """Tokenize study root from fixed upload session + editable study name."""
    return resolve_study_tokenize_dir(upload_dir, _raw_study_name())


def _apply_study_settings_to_yaml(upload_dir: Path) -> bool:
    """Write data.input_dir and output_prefix from study name (no manual path edit)."""
    study_name = normalize_study_name(_raw_study_name())
    if not study_name:
        st.session_state["pipeline_set_input_msg"] = (
            "warning",
            "Enter an **experiment name** in Study name, then upload a .zip or click "
            "**Apply setting to Config YAML**.",
        )
        return False
    tokenize_dir = _compute_tokenize_input_dir(upload_dir)
    if tokenize_dir is None:
        target = study_folder(upload_dir, study_name)
        hint = (
            f"No Single-Cell data under `{target}`. Upload a .zip first, "
            "or set **Study name** to match the folder created on import."
        )
        others = [
            p.name
            for p in sorted(upload_dir.iterdir())
            if p.is_dir() and not p.name.startswith(".")
        ]
        if others:
            hint += f" Folders in this session: `{', '.join(others)}`."
        st.session_state["pipeline_set_input_msg"] = ("warning", hint)
        return False
    study_name = normalize_study_name(_raw_study_name())
    states = unique_states_from_samples(tokenize_dir)
    st.session_state["pipeline_detected_states"] = states
    patch_kwargs: dict = {
        "input_dir": tokenize_dir,
        "output_prefix": study_name,
    }
    if "pipeline_batch_mode" in st.session_state:
        patch_kwargs["forward_batch_size"] = _selected_forward_batch_size()
    if "pipeline_ft_train_batch_size" in st.session_state:
        patch_kwargs["train_batch_size"] = _selected_ft_train_batch_size()
    _patch_pipeline_yaml(**patch_kwargs)
    st.session_state["pipeline_tokenize_dir"] = str(tokenize_dir)
    return True


def _process_study_zip_upload(uploaded_file, upload_dir: Path) -> None:
    """Import zip once; do not re-run on every widget rerun."""
    buf = uploaded_file.getbuffer()
    data = buf.getvalue() if hasattr(buf, "getvalue") else bytes(buf)
    tokenize_dir, summary = import_study_zip(data, upload_dir, _raw_study_name())
    _finish_study_import(
        tokenize_dir,
        summary,
        fingerprint=_zip_upload_fingerprint(uploaded_file),
        source_label="uploaded zip",
    )


def _url_fingerprint(url: str) -> str:
    return f"url:{(url or '').strip()}"


def _finish_study_import(
    tokenize_dir: Path,
    summary: str,
    *,
    fingerprint: str,
    source_label: str,
) -> None:
    """Shared post-import bookkeeping for zip upload and URL download."""
    study_name = normalize_study_name(_raw_study_name())
    states = unique_states_from_samples(tokenize_dir)
    st.session_state["pipeline_detected_states"] = states
    st.session_state["pipeline_tokenize_dir"] = str(tokenize_dir)
    isp_start, isp_end = _default_isp_start_end(states)
    _sync_isp_state_selectors(states, force=True)
    patch_kwargs: dict = {
        "input_dir": tokenize_dir,
        "output_prefix": study_name,
        "isp_start_state": isp_start,
        "isp_end_state": isp_end,
    }
    if "pipeline_batch_mode" in st.session_state:
        patch_kwargs["forward_batch_size"] = _selected_forward_batch_size()
    if "pipeline_ft_train_batch_size" in st.session_state:
        patch_kwargs["train_batch_size"] = _selected_ft_train_batch_size()
    _patch_pipeline_yaml(**patch_kwargs)
    st.session_state["processed_zip_fingerprint"] = fingerprint
    st.session_state["last_data_source"] = source_label
    st.success(
        f"Study **{study_name}** imported from {source_label}. "
        f"`data.input_dir` → `{tokenize_dir}`"
    )
    st.markdown(summary)


def _process_study_url_import(url: str, upload_dir: Path) -> None:
    """Download a deposited archive URL and import it as a study."""
    progress = st.progress(0, text="Starting download…")

    def _on_progress(downloaded: int, total: int | None) -> None:
        if total and total > 0:
            frac = min(1.0, downloaded / total)
            mb = downloaded / (1024 * 1024)
            progress.progress(
                frac,
                text=f"Downloading… {mb:.1f} / {total / (1024 * 1024):.1f} MiB",
            )
        else:
            mb = downloaded / (1024 * 1024)
            # Indeterminate-ish: cap visual at 95% until complete.
            progress.progress(min(0.95, mb / 200.0), text=f"Downloading… {mb:.1f} MiB")

    tokenize_dir, summary, filename = import_study_from_url(
        url,
        upload_dir,
        _raw_study_name(),
        progress=_on_progress,
    )
    progress.progress(1.0, text=f"Imported `{filename}`")
    _finish_study_import(
        tokenize_dir,
        summary,
        fingerprint=_url_fingerprint(url),
        source_label=f"URL (`{filename}`)",
    )


def _infer_study_name_from_yaml(upload_dir: Path) -> str:
    input_dir, prefix = _read_pipeline_data_fields()
    try:
        rel = Path(input_dir).resolve().relative_to(upload_dir.resolve())
        if rel.parts:
            return rel.parts[0]
    except ValueError:
        pass
    if prefix:
        return str(prefix)
    return ""


def _sync_pipeline_form_from_yaml(upload_dir: Path) -> None:
    """Initialize pipeline form fields from the current YAML editor text."""
    if "pipeline_study_name" not in st.session_state:
        st.session_state["pipeline_study_name"] = _infer_study_name_from_yaml(upload_dir)
    tokenize_dir = _compute_tokenize_input_dir(upload_dir)
    if tokenize_dir is not None:
        st.session_state["pipeline_tokenize_dir"] = str(tokenize_dir)


def _detected_states_from_upload(upload_dir: Path) -> list[str]:
    tokenize_dir = resolve_study_tokenize_dir(upload_dir, _raw_study_name())
    if tokenize_dir is None:
        return []
    try:
        return unique_states_from_samples(tokenize_dir)
    except Exception:
        return []


def _tokenizer_custom_attr_columns() -> list[str]:
    """Keys from pipeline stages.tokenize or default tokenize.yaml custom_attr_name_dict."""
    cfg = _read_pipeline_yaml()
    custom = _nested_get(
        cfg, "stages", "tokenize", "tokenizer", "custom_attr_name_dict", default=None
    )
    if not isinstance(custom, dict):
        custom = (cfg.get("tokenizer") or {}).get("custom_attr_name_dict")
    if not isinstance(custom, dict):
        # Fall back to shipped tokenize template (usual folder-derived attrs).
        try:
            path = CORE / "config" / "tokenize.yaml"
            if path.is_file():
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                custom = (data.get("tokenizer") or {}).get("custom_attr_name_dict")
        except Exception:
            custom = None
    if not isinstance(custom, dict):
        return []
    return [str(k).strip() for k in custom.keys() if str(k).strip()]


def _label_column_options(upload_dir: Path | None) -> list[str]:
    """Dropdown candidates for fine-tune label_column."""
    study_root = None
    if upload_dir is not None:
        study_root = resolve_study_tokenize_dir(upload_dir, _raw_study_name())
    extra = _tokenizer_custom_attr_columns()
    current = str(st.session_state.get("pipeline_ft_label_column") or "").strip()
    if current:
        extra = [current, *extra]
    options = label_column_candidates_from_study(study_root, extra=extra)
    return options


def _nested_get(mapping: dict | None, *keys, default=None):
    cur = mapping or {}
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _parse_genes_to_perturb_text(text: str | None) -> list[str]:
    """Parse UI text into a gene list (comma or newline separated). Empty → []."""
    if not text:
        return []
    genes: list[str] = []
    seen: set[str] = set()
    for chunk in str(text).replace(",", "\n").splitlines():
        gene = chunk.strip()
        if gene and gene not in seen:
            genes.append(gene)
            seen.add(gene)
    return genes


def _genes_to_perturb_as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(g).strip() for g in value if str(g).strip())
    return str(value).strip()


def _build_patched_pipeline_yaml(
    *,
    input_dir: str | Path | None = None,
    output_prefix: str | None = "__unset__",
    isp_start_state: str | None = None,
    isp_end_state: str | None = None,
    model_organism: str | None = None,
    model: str | None = None,
    human_variant: str | None = None,
    mouse_variant: str | None = None,
    ortholog_policy: str | None = None,
    ortholog_curated_overlay: str | None = None,
    ortholog_approval_record: str | None = None,
    ortholog_audit: str | None = None,
    forward_batch_size: int | str | None = None,
    train_batch_size: int | str | None = None,
    ft_epochs: int | None = None,
    ft_learning_rate: float | None = None,
    ft_num_runs: int | None = None,
    ft_warmup_mode: str | None = None,
    isp_max_ncells: int | None = None,
    isp_stats_mode: str | None = None,
    isp_analysis_enabled: bool | None = None,
    isp_postprocess_enabled: bool | None = None,
    isp_postprocess_n_clusters: int | str | None = None,
    isp_postprocess_celltype: bool | None = None,
    pert_type: str | None = None,
    pert_state_key: str | None = None,
    pert_genes_to_perturb: list[str] | None = None,
    ft_task_type: str | None = None,
    ft_label_column: str | None = None,
    runtime_max_cells: int | None = None,
) -> str:
    """Return patched pipeline YAML text without requiring a live yaml_editor widget write."""
    cfg = _read_pipeline_yaml()
    cfg.setdefault("data", {})
    if input_dir is not None:
        cfg["data"]["input_dir"] = str(input_dir)
    if output_prefix != "__unset__":
        cfg["data"]["output_prefix"] = output_prefix
    if (
        isp_start_state is not None
        or isp_end_state is not None
        or pert_type is not None
        or pert_state_key is not None
        or pert_genes_to_perturb is not None
    ):
        cfg.setdefault("perturbation", {})
        if isp_start_state is not None:
            cfg["perturbation"]["start_state"] = isp_start_state
        if isp_end_state is not None:
            cfg["perturbation"]["end_state"] = isp_end_state
        if pert_type is not None:
            cfg["perturbation"]["type"] = str(pert_type)
        if pert_state_key is not None:
            cfg["perturbation"]["state_key"] = str(pert_state_key).strip() or _DEFAULT_STATE_KEY
        if pert_genes_to_perturb is not None:
            cfg["perturbation"]["genes_to_perturb"] = list(pert_genes_to_perturb)
    if (
        model_organism is not None
        or model is not None
        or human_variant is not None
        or mouse_variant is not None
        or ortholog_policy is not None
        or ortholog_curated_overlay is not None
    ):
        cfg.setdefault("species", {})
        if model_organism is not None:
            cfg["species"]["model_organism"] = model_organism
        if model is not None:
            cfg["species"]["model"] = model
        if human_variant is not None:
            cfg["species"]["human_variant"] = human_variant
        if mouse_variant is not None:
            cfg["species"]["mouse_variant"] = mouse_variant
        if ortholog_policy is not None:
            cfg["species"]["ortholog_policy"] = ortholog_policy
        if ortholog_curated_overlay is not None:
            if str(ortholog_curated_overlay).strip() in ("", "null", "None"):
                cfg["species"].pop("ortholog_curated_overlay", None)
            else:
                cfg["species"]["ortholog_curated_overlay"] = str(
                    ortholog_curated_overlay
                ).strip()
    if ortholog_approval_record is not None or ortholog_audit is not None:
        stages = cfg.setdefault("stages", {})
        tok_stage = stages.setdefault("tokenize", {})
        tok_cfg = tok_stage.setdefault("tokenizer", {})
        if ortholog_audit is not None and str(ortholog_audit).strip():
            tok_cfg["ortholog_audit"] = str(ortholog_audit).strip()
            tok_cfg.setdefault("ortholog_loss_gate", True)
        if ortholog_approval_record is not None and str(ortholog_approval_record).strip():
            tok_cfg["ortholog_approval_record"] = str(ortholog_approval_record).strip()
    if (
        forward_batch_size is not None
        or train_batch_size is not None
        or runtime_max_cells is not None
    ):
        cfg.setdefault("runtime", {})
        if forward_batch_size is not None:
            cfg["runtime"]["forward_batch_size"] = forward_batch_size
        if train_batch_size is not None:
            cfg["runtime"]["train_batch_size"] = train_batch_size
        if runtime_max_cells is not None:
            cfg["runtime"]["max_cells"] = int(runtime_max_cells)

    if (
        ft_epochs is not None
        or ft_learning_rate is not None
        or ft_num_runs is not None
        or ft_warmup_mode is not None
        or ft_task_type is not None
        or ft_label_column is not None
    ):
        stages = cfg.setdefault("stages", {})
        finetune = stages.setdefault("finetune", {})
        if (
            ft_epochs is not None
            or ft_learning_rate is not None
            or ft_num_runs is not None
            or ft_warmup_mode is not None
        ):
            training = finetune.setdefault("training", {})
            if ft_epochs is not None:
                training["epochs"] = int(ft_epochs)
            if ft_learning_rate is not None:
                training["learning_rate"] = float(ft_learning_rate)
            if ft_num_runs is not None:
                training["num_runs"] = int(ft_num_runs)
            if ft_warmup_mode is not None:
                if str(ft_warmup_mode) == _FT_WARMUP_RATIO_005:
                    training["warmup_ratio"] = 0.05
                    training["warmup_steps"] = 500
                else:
                    training["warmup_ratio"] = None
                    training["warmup_steps"] = 500
        if ft_task_type is not None or ft_label_column is not None:
            ft_block = finetune.setdefault("finetune", {})
            if ft_task_type is not None:
                ft_block["task_type"] = str(ft_task_type)
            if ft_label_column is not None:
                ft_block["label_column"] = (
                    str(ft_label_column).strip() or _DEFAULT_FT_LABEL_COLUMN
                )

    if (
        isp_max_ncells is not None
        or isp_stats_mode is not None
        or isp_analysis_enabled is not None
        or isp_postprocess_enabled is not None
        or isp_postprocess_n_clusters is not None
        or isp_postprocess_celltype is not None
    ):
        stages = cfg.setdefault("stages", {})
        isp_stage = stages.setdefault("isp", {})
        if isp_max_ncells is not None:
            isp_stage.setdefault("isp", {})["max_ncells"] = int(isp_max_ncells)
        if isp_stats_mode is not None:
            isp_stage.setdefault("stats", {})["mode"] = str(isp_stats_mode)
        if isp_analysis_enabled is not None:
            isp_stage.setdefault("analysis", {})["enabled"] = bool(isp_analysis_enabled)
        if (
            isp_postprocess_enabled is not None
            or isp_postprocess_n_clusters is not None
            or isp_postprocess_celltype is not None
        ):
            post = isp_stage.setdefault("postprocess", {})
            if isp_postprocess_enabled is not None:
                post["enabled"] = bool(isp_postprocess_enabled)
            if isp_postprocess_n_clusters is not None:
                post["n_clusters"] = _normalize_n_clusters_value(isp_postprocess_n_clusters)
            if isp_postprocess_celltype is not None:
                post["celltype_prediction"] = bool(isp_postprocess_celltype)

    return yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _patch_pipeline_yaml(
    *,
    input_dir: str | Path | None = None,
    output_prefix: str | None = "__unset__",
    isp_start_state: str | None = None,
    isp_end_state: str | None = None,
    model_organism: str | None = None,
    model: str | None = None,
    human_variant: str | None = None,
    mouse_variant: str | None = None,
    ortholog_policy: str | None = None,
    ortholog_curated_overlay: str | None = None,
    ortholog_approval_record: str | None = None,
    ortholog_audit: str | None = None,
    forward_batch_size: int | str | None = None,
    train_batch_size: int | str | None = None,
    ft_epochs: int | None = None,
    ft_learning_rate: float | None = None,
    ft_num_runs: int | None = None,
    ft_warmup_mode: str | None = None,
    isp_max_ncells: int | None = None,
    isp_stats_mode: str | None = None,
    isp_analysis_enabled: bool | None = None,
    isp_postprocess_enabled: bool | None = None,
    isp_postprocess_n_clusters: int | str | None = None,
    isp_postprocess_celltype: bool | None = None,
    pert_type: str | None = None,
    pert_state_key: str | None = None,
    pert_genes_to_perturb: list[str] | None = None,
    ft_task_type: str | None = None,
    ft_label_column: str | None = None,
    runtime_max_cells: int | None = None,
) -> bool:
    """Patch pipeline / FT-calibrate YAML fields in the editor."""
    dumped = _build_patched_pipeline_yaml(
        input_dir=input_dir,
        output_prefix=output_prefix,
        isp_start_state=isp_start_state,
        isp_end_state=isp_end_state,
        model_organism=model_organism,
        model=model,
        human_variant=human_variant,
        mouse_variant=mouse_variant,
        ortholog_policy=ortholog_policy,
        ortholog_curated_overlay=ortholog_curated_overlay,
        ortholog_approval_record=ortholog_approval_record,
        ortholog_audit=ortholog_audit,
        forward_batch_size=forward_batch_size,
        train_batch_size=train_batch_size,
        ft_epochs=ft_epochs,
        ft_learning_rate=ft_learning_rate,
        ft_num_runs=ft_num_runs,
        ft_warmup_mode=ft_warmup_mode,
        isp_max_ncells=isp_max_ncells,
        isp_stats_mode=isp_stats_mode,
        isp_analysis_enabled=isp_analysis_enabled,
        isp_postprocess_enabled=isp_postprocess_enabled,
        isp_postprocess_n_clusters=isp_postprocess_n_clusters,
        isp_postprocess_celltype=isp_postprocess_celltype,
        pert_type=pert_type,
        pert_state_key=pert_state_key,
        pert_genes_to_perturb=pert_genes_to_perturb,
        ft_task_type=ft_task_type,
        ft_label_column=ft_label_column,
        runtime_max_cells=runtime_max_cells,
    )
    st.session_state["yaml_editor"] = dumped
    if input_dir is not None:
        st.session_state["pipeline_tokenize_dir"] = str(input_dir)
    return True


def _prepare_pipeline_yaml_for_run(upload_dir: Path) -> tuple[str | None, str | None]:
    """
    Ensure Pipeline YAML points at the uploaded study + species/model before starting.

    Returns (yaml_text, error_message). On success error_message is None.
    """
    study_name = normalize_study_name(_raw_study_name())
    if not study_name:
        return None, (
            "Set **Study name** to the experiment folder (the name used when uploading the zip), "
            "then click **Apply setting to Config YAML** or Run again."
        )

    tokenize_dir = resolve_study_tokenize_dir(upload_dir, study_name)
    if tokenize_dir is None:
        target = study_folder(upload_dir, study_name)
        others = [
            p.name
            for p in sorted(upload_dir.iterdir())
            if p.is_dir() and not p.name.startswith(".")
        ]
        msg = (
            f"No 10x sample folders under study **{study_name}** "
            f"(`{target}`).\n\n"
            "Upload a `.zip` whose contents are sample folders "
            "(`Time-State-Suffix/…`), with **Study name** matching that experiment."
        )
        if others:
            msg += (
                f"\n\nFolders in this upload session: `{', '.join(others)}` "
                "— set Study name to one of these."
            )
        else:
            msg += "\n\nNo uploads in this session yet."
        return None, msg

    samples = discover_sample_dirs(tokenize_dir)
    if not samples:
        return None, (
            f"Study root `{tokenize_dir}` has no detectable 10x samples. "
            "Check folder names and `barcodes/features/matrix` files."
        )

    start_state = st.session_state.get("pipeline_isp_start_state")
    end_state = st.session_state.get("pipeline_isp_end_state")
    if start_state and end_state and str(start_state) == str(end_state):
        return None, (
            f"ISP **start_state** and **end_state** are both `{start_state}`.\n\n"
            "Goal-state-shift ISP needs two different conditions "
            "(e.g. start=`AD`, end=`WT`). Change the dropdowns and run again."
        )

    available = unique_states_from_samples(tokenize_dir)
    unknown = sorted({str(s) for s in (start_state, end_state) if s} - set(available))
    if available and unknown:
        return None, (
            f"ISP states `{', '.join(unknown)}` do not exist in study **{study_name}**.\n\n"
            f"Sample folders provide: `{', '.join(available)}`. "
            "Pick those in the **ISP start_state / end_state** dropdowns."
        )

    _sync_ft_isp_advanced_from_yaml()
    yaml_text = _build_patched_pipeline_yaml(
        input_dir=tokenize_dir,
        output_prefix=study_name,
        isp_start_state=start_state,
        isp_end_state=end_state,
        model_organism=st.session_state.get("pipeline_model_organism"),
        model=st.session_state.get("pipeline_model"),
        human_variant=st.session_state.get("pipeline_human_variant"),
        mouse_variant=st.session_state.get("pipeline_mouse_variant"),
        ortholog_policy=st.session_state.get("pipeline_ortholog_policy"),
        ortholog_curated_overlay=st.session_state.get("pipeline_ortholog_curated_overlay"),
        ortholog_approval_record=st.session_state.get("pipeline_ortholog_approval_record"),
        ortholog_audit=st.session_state.get("pipeline_ortholog_audit"),
        forward_batch_size=_selected_forward_batch_size(),
        train_batch_size=_selected_ft_train_batch_size(),
        **_ft_isp_advanced_kwargs_from_session(),
    )
    st.session_state["pipeline_tokenize_dir"] = str(tokenize_dir)
    return yaml_text, None


def _prepare_ft_batch_calibrate_yaml_for_run(
    upload_dir: Path,
) -> tuple[str | None, str | None]:
    """Patch FT-calibrate YAML with species (+ study path when available)."""
    study_name = normalize_study_name(_raw_study_name())
    tokenize_dir = (
        resolve_study_tokenize_dir(upload_dir, study_name) if study_name else None
    )
    yaml_text = _build_patched_pipeline_yaml(
        input_dir=tokenize_dir,
        output_prefix=study_name if study_name else "__unset__",
        model_organism=st.session_state.get("pipeline_model_organism"),
        model=st.session_state.get("pipeline_model"),
        human_variant=st.session_state.get("pipeline_human_variant"),
        mouse_variant=st.session_state.get("pipeline_mouse_variant"),
        ortholog_policy=st.session_state.get("pipeline_ortholog_policy"),
    )
    if tokenize_dir is not None:
        st.session_state["pipeline_tokenize_dir"] = str(tokenize_dir)
    return yaml_text, None


def _apply_isp_states_to_yaml() -> None:
    _patch_pipeline_yaml(
        isp_start_state=st.session_state.get("pipeline_isp_start_state"),
        isp_end_state=st.session_state.get("pipeline_isp_end_state"),
    )


def _apply_species_to_yaml() -> None:
    _patch_pipeline_yaml(
        model_organism=st.session_state.get("pipeline_model_organism"),
        model=st.session_state.get("pipeline_model"),
        human_variant=st.session_state.get("pipeline_human_variant"),
        mouse_variant=st.session_state.get("pipeline_mouse_variant"),
        ortholog_policy=st.session_state.get("pipeline_ortholog_policy"),
    )


def _warmup_mode_from_training(ft_training: dict) -> str:
    ratio = ft_training.get("warmup_ratio")
    if ratio is not None:
        try:
            if float(ratio) > 0:
                return _FT_WARMUP_RATIO_005
        except (TypeError, ValueError):
            pass
    return _FT_WARMUP_500_STEPS


def _ft_isp_advanced_kwargs_from_session() -> dict:
    return {
        "ft_epochs": st.session_state.get("pipeline_ft_epochs"),
        "ft_learning_rate": st.session_state.get("pipeline_ft_learning_rate"),
        "ft_num_runs": st.session_state.get("pipeline_ft_num_runs"),
        "ft_warmup_mode": st.session_state.get(
            "pipeline_ft_warmup_mode", _DEFAULT_FT_WARMUP_MODE
        ),
        "isp_max_ncells": st.session_state.get("pipeline_isp_max_ncells"),
        "isp_stats_mode": st.session_state.get("pipeline_isp_stats_mode"),
        "isp_analysis_enabled": st.session_state.get("pipeline_isp_analysis_enabled"),
        "isp_postprocess_enabled": st.session_state.get("pipeline_isp_postprocess_enabled"),
        "isp_postprocess_n_clusters": _n_clusters_from_mode_session(
            "pipeline_isp_postprocess_n_clusters_mode",
            "pipeline_isp_postprocess_n_clusters",
        ),
        "isp_postprocess_celltype": st.session_state.get(
            "pipeline_isp_postprocess_celltype"
        ),
        "ft_task_type": st.session_state.get("pipeline_ft_task_type"),
        "ft_label_column": st.session_state.get("pipeline_ft_label_column"),
        "runtime_max_cells": st.session_state.get("pipeline_max_cells"),
        "pert_type": st.session_state.get("pipeline_pert_type"),
        "pert_state_key": st.session_state.get("pipeline_pert_state_key"),
        "pert_genes_to_perturb": _parse_genes_to_perturb_text(
            st.session_state.get("pipeline_genes_to_perturb_text")
        ),
    }


def _apply_ft_isp_advanced_to_yaml() -> None:
    _patch_pipeline_yaml(**_ft_isp_advanced_kwargs_from_session())


def _apply_perturbation_controls_to_yaml() -> None:
    _patch_pipeline_yaml(
        pert_type=st.session_state.get("pipeline_pert_type"),
        pert_state_key=st.session_state.get("pipeline_pert_state_key"),
        pert_genes_to_perturb=_parse_genes_to_perturb_text(
            st.session_state.get("pipeline_genes_to_perturb_text")
        ),
        isp_start_state=st.session_state.get("pipeline_isp_start_state"),
        isp_end_state=st.session_state.get("pipeline_isp_end_state"),
    )


def _sync_ft_isp_advanced_from_yaml() -> None:
    cfg = _read_pipeline_yaml()
    stages = cfg.get("stages") or {}
    ft_training = _nested_get(stages, "finetune", "training", default={}) or {}
    ft_block = _nested_get(stages, "finetune", "finetune", default={}) or {}
    isp_stage = _nested_get(stages, "isp", default={}) or {}
    pert = cfg.get("perturbation") or {}
    runtime = cfg.get("runtime") or {}

    epochs = ft_training.get("epochs", _DEFAULT_FT_EPOCHS)
    lr = ft_training.get("learning_rate", _DEFAULT_FT_LEARNING_RATE)
    num_runs = ft_training.get("num_runs", _DEFAULT_FT_NUM_RUNS)
    max_ncells = _nested_get(isp_stage, "isp", "max_ncells", default=_DEFAULT_ISP_MAX_NCELLS)
    stats_mode = _nested_get(isp_stage, "stats", "mode", default=_DEFAULT_STATS_MODE)
    analysis_enabled = _nested_get(
        isp_stage, "analysis", "enabled", default=_DEFAULT_ISP_ANALYSIS_ENABLED
    )
    postprocess_enabled = _nested_get(
        isp_stage, "postprocess", "enabled", default=_DEFAULT_ISP_POSTPROCESS_ENABLED
    )
    postprocess_n_clusters = _nested_get(
        isp_stage,
        "postprocess",
        "n_clusters",
        default=_DEFAULT_ISP_POSTPROCESS_N_CLUSTERS,
    )
    postprocess_celltype = _nested_get(
        isp_stage,
        "postprocess",
        "celltype_prediction",
        default=_DEFAULT_ISP_POSTPROCESS_CELLTYPE,
    )
    task_type = ft_block.get("task_type", _DEFAULT_FT_TASK_TYPE)
    label_column = ft_block.get("label_column", _DEFAULT_FT_LABEL_COLUMN)
    max_cells = runtime.get("max_cells", _DEFAULT_MAX_CELLS)
    pert_type = pert.get("type", _DEFAULT_PERTURB_TYPE)
    state_key = pert.get("state_key", _DEFAULT_STATE_KEY)
    genes_text = _genes_to_perturb_as_text(pert.get("genes_to_perturb"))

    st.session_state.setdefault("pipeline_ft_epochs", int(epochs))
    try:
        st.session_state.setdefault("pipeline_ft_learning_rate", float(lr))
    except (TypeError, ValueError):
        st.session_state.setdefault("pipeline_ft_learning_rate", _DEFAULT_FT_LEARNING_RATE)
    st.session_state.setdefault("pipeline_ft_num_runs", int(num_runs))
    warmup_mode = _warmup_mode_from_training(ft_training)
    st.session_state.setdefault("pipeline_ft_warmup_mode", warmup_mode)
    st.session_state.setdefault("pipeline_isp_max_ncells", int(max_ncells))
    mode = str(stats_mode) if stats_mode in _STATS_MODE_LABELS else _DEFAULT_STATS_MODE
    st.session_state.setdefault("pipeline_isp_stats_mode", mode)
    st.session_state.setdefault("pipeline_isp_analysis_enabled", bool(analysis_enabled))
    st.session_state.setdefault(
        "pipeline_isp_postprocess_enabled", bool(postprocess_enabled)
    )
    _set_n_clusters_session_from_config(
        "pipeline_isp_postprocess_n_clusters_mode",
        "pipeline_isp_postprocess_n_clusters",
        postprocess_n_clusters,
    )
    st.session_state.setdefault(
        "pipeline_isp_postprocess_celltype", bool(postprocess_celltype)
    )
    task = str(task_type) if task_type in _FT_TASK_TYPE_LABELS else _DEFAULT_FT_TASK_TYPE
    st.session_state.setdefault("pipeline_ft_task_type", task)
    st.session_state.setdefault(
        "pipeline_ft_label_column", str(label_column or _DEFAULT_FT_LABEL_COLUMN)
    )
    try:
        st.session_state.setdefault("pipeline_max_cells", int(max_cells))
    except (TypeError, ValueError):
        st.session_state.setdefault("pipeline_max_cells", _DEFAULT_MAX_CELLS)
    ptype = str(pert_type) if pert_type in _PERTURB_TYPE_LABELS else _DEFAULT_PERTURB_TYPE
    st.session_state.setdefault("pipeline_pert_type", ptype)
    st.session_state.setdefault(
        "pipeline_pert_state_key", str(state_key or _DEFAULT_STATE_KEY)
    )
    st.session_state.setdefault("pipeline_genes_to_perturb_text", genes_text)


def _render_perturbation_controls() -> None:
    """ISP perturbation type / state_key / genes (writes perturbation.*)."""
    _sync_ft_isp_advanced_from_yaml()
    st.markdown("**ISP perturbation**")
    st.caption(
        "Writes `perturbation.*` (copied into the ISP stage). "
        "Empty gene list = perturb **all** genes (slow)."
    )
    c_type, c_key = st.columns(2)
    with c_type:
        st.selectbox(
            "perturbation.type",
            list(_PERTURB_TYPE_LABELS.keys()),
            key="pipeline_pert_type",
            format_func=lambda v: _PERTURB_TYPE_LABELS.get(v, v),
            on_change=_apply_perturbation_controls_to_yaml,
            help="How the gene is changed in the rank encoding (`perturbation.type`).",
        )
    with c_key:
        st.text_input(
            "state_key",
            key="pipeline_pert_state_key",
            on_change=_apply_perturbation_controls_to_yaml,
            help=(
                "Dataset column for start/end labels (`perturbation.state_key`). "
                "Usually `disease` (folder-derived metadata)."
            ),
        )
    st.text_area(
        "genes_to_perturb (one per line; empty = all)",
        key="pipeline_genes_to_perturb_text",
        height=90,
        on_change=_apply_perturbation_controls_to_yaml,
        help=(
            "Symbol or Ensembl IDs. Empty list runs genome-wide ISP. "
            "Targeted lists (e.g. Igfbp2) are needed for ISP UMAP trajectories."
        ),
        placeholder="Igfbp2\n# or leave empty for all genes",
    )


def _render_ft_isp_advanced_controls(upload_dir: Path | None = None) -> None:
    """Fine-tune training + ISP stats/cell-cap/UMAP/analysis (writes stages.*)."""
    _sync_ft_isp_advanced_from_yaml()

    st.markdown("**Fine-tune task**")
    st.caption(
        "Writes `stages.finetune.finetune.*`. "
        "`label_column` candidates come from sample-folder metadata "
        "(and tokenizer `custom_attr_name_dict`)."
    )
    c_task, c_label = st.columns(2)
    with c_task:
        st.selectbox(
            "task_type",
            list(_FT_TASK_TYPE_LABELS.keys()),
            key="pipeline_ft_task_type",
            format_func=lambda v: _FT_TASK_TYPE_LABELS.get(v, v),
            on_change=_apply_ft_isp_advanced_to_yaml,
            help="Classification task (`stages.finetune.finetune.task_type`).",
        )
    with c_label:
        label_options = _label_column_options(upload_dir)
        current_label = str(
            st.session_state.get("pipeline_ft_label_column") or _DEFAULT_FT_LABEL_COLUMN
        ).strip()
        if current_label and current_label not in label_options:
            label_options = [current_label, *label_options]
        if not label_options:
            label_options = [_DEFAULT_FT_LABEL_COLUMN]
        st.selectbox(
            "label_column",
            label_options,
            key="pipeline_ft_label_column",
            on_change=_apply_ft_isp_advanced_to_yaml,
            help=(
                "Column used as the classification label "
                "(`stages.finetune.finetune.label_column`). "
                "Detected from uploaded sample folders when available."
            ),
        )
        if upload_dir is not None and resolve_study_tokenize_dir(
            upload_dir, _raw_study_name()
        ):
            st.caption(f"Candidates: {', '.join(label_options)}")

    with st.expander("Advanced options", expanded=False):
        st.markdown("**Fine-tune training**")
        st.caption(
            "Writes `stages.finetune.training.*`. Changing epochs / LR / num_runs / warmup "
            "changes the fine-tuned model and therefore ISP results."
        )
        c_ep, c_lr, c_runs = st.columns(3)
        with c_ep:
            st.number_input(
                "epochs",
                min_value=1,
                max_value=200,
                step=1,
                key="pipeline_ft_epochs",
                on_change=_apply_ft_isp_advanced_to_yaml,
                help="Full passes over the training set (`stages.finetune.training.epochs`).",
            )
        with c_lr:
            st.number_input(
                "learning_rate",
                min_value=1e-7,
                max_value=1e-2,
                step=1e-5,
                format="%.1e",
                key="pipeline_ft_learning_rate",
                on_change=_apply_ft_isp_advanced_to_yaml,
                help="AdamW learning rate (`stages.finetune.training.learning_rate`). Default 5e-5.",
            )
        with c_runs:
            st.number_input(
                "num_runs",
                min_value=1,
                max_value=20,
                step=1,
                key="pipeline_ft_num_runs",
                on_change=_apply_ft_isp_advanced_to_yaml,
                help="Independent fine-tune runs with different seeds (`training.num_runs`).",
            )
        st.selectbox(
            "warmup",
            list(_FT_WARMUP_MODE_LABELS.keys()),
            key="pipeline_ft_warmup_mode",
            format_func=lambda v: _FT_WARMUP_MODE_LABELS.get(v, v),
            on_change=_apply_ft_isp_advanced_to_yaml,
            help=(
                "Choose **500 steps** for a virtual genetic screen (ISP gene-effect size). "
                "Choose **rate 0.05** when the fine-tune goal is classification "
                "(cell type / disease accuracy), not per-gene ISP. "
                "Writes `stages.finetune.training.warmup_steps` or `warmup_ratio`."
            ),
        )

        st.markdown("**Tokenize / ISP options**")
        st.caption(
            "`max_cells` caps tokenization; `max_ncells` / `stats.mode` affect ISP rankings; "
            "UMAP / analysis only control optional plots."
        )
        st.number_input(
            "tokenize max_cells",
            min_value=100,
            max_value=5_000_000,
            step=10_000,
            key="pipeline_max_cells",
            on_change=_apply_ft_isp_advanced_to_yaml,
            help=(
                "Max cells tokenized (`runtime.max_cells` → tokenizer.max_cells). "
                "Large values need more RAM; does not change ISP cell cap."
            ),
        )
        c_cells, c_stats = st.columns(2)
        with c_cells:
            st.number_input(
                "max_ncells",
                min_value=1,
                max_value=500_000,
                step=100,
                key="pipeline_isp_max_ncells",
                on_change=_apply_ft_isp_advanced_to_yaml,
                help=(
                    "Cap on cells after filters (`stages.isp.isp.max_ncells`). "
                    "Lower = faster but different gene rankings. Same default as Mouse-Geneformer-WebUI (2000)."
                ),
            )
        with c_stats:
            st.selectbox(
                "stats.mode",
                list(_STATS_MODE_LABELS.keys()),
                key="pipeline_isp_stats_mode",
                format_func=lambda v: _STATS_MODE_LABELS.get(v, v),
                on_change=_apply_ft_isp_advanced_to_yaml,
                help="Post-ISP statistics mode (`stages.isp.stats.mode`). Usual choice: goal_state_shift.",
            )
        c_umap, c_analysis = st.columns(2)
        with c_umap:
            st.checkbox(
                "ISP analysis plots",
                key="pipeline_isp_analysis_enabled",
                on_change=_apply_ft_isp_advanced_to_yaml,
                help="Bar / volcano figures after ISP stats (`stages.isp.analysis.enabled`).",
            )
        with c_analysis:
            st.caption(
                "Cell **trajectory** UMAP: use Run type **ISP UMAP** after a completed "
                "Pipeline (E2E). Pipeline (E2E) also runs **TOP1 significant ISP UMAP** automatically; "
                "Fine-tune UMAP is controlled by `stages.finetune.umap.enabled`."
            )

        with st.expander("Cluster / cell-type analysis", expanded=False):
            st.caption(
                "After ISP UMAP (including E2E TOP1), optionally write "
                "`cluster_coexpr_analysis/` (`stages.isp.postprocess.*`). Default **off**."
            )
            st.checkbox(
                "Enable cluster_coexpr_analysis",
                key="pipeline_isp_postprocess_enabled",
                on_change=_apply_ft_isp_advanced_to_yaml,
                help="`stages.isp.postprocess.enabled`",
            )
            if st.session_state.get("pipeline_isp_postprocess_enabled"):
                st.radio(
                    "n_clusters (KMeans)",
                    options=["auto", "manual"],
                    format_func=lambda m: (
                        "auto (silhouette)" if m == "auto" else "manual (specify K)"
                    ),
                    horizontal=True,
                    key="pipeline_isp_postprocess_n_clusters_mode",
                    on_change=_apply_ft_isp_advanced_to_yaml,
                    help=(
                        "auto: silhouette-optimal K (2..15) on start embeddings. "
                        "manual: fixed K (`stages.isp.postprocess.n_clusters`)."
                    ),
                )
                if (
                    st.session_state.get("pipeline_isp_postprocess_n_clusters_mode")
                    == "manual"
                ):
                    st.number_input(
                        "K",
                        min_value=2,
                        max_value=50,
                        step=1,
                        key="pipeline_isp_postprocess_n_clusters",
                        on_change=_apply_ft_isp_advanced_to_yaml,
                        help="`stages.isp.postprocess.n_clusters`",
                    )
                st.checkbox(
                    "Cell-type prediction (marker genes)",
                    key="pipeline_isp_postprocess_celltype",
                    on_change=_apply_ft_isp_advanced_to_yaml,
                    help="`stages.isp.postprocess.celltype_prediction`",
                )

    with st.expander("What do these options mean?", expanded=False):
        st.markdown(
            """
| Control | Effect |
|---------|--------|
| **perturbation.type** | delete / overexpress / inhibit / activate |
| **state_key** | Metadata column for start/end (usually `disease`) |
| **genes_to_perturb** | Empty = all genes; list = targeted ISP |
| **task_type / label_column** | What the fine-tune classifier learns |
| **tokenize max_cells** | Cap cells at tokenize (RAM / time) |
| **max_ncells** | How many cells ISP uses. Lower ≈ faster; **results change**. |
| **stats.mode** | How cosine shifts are scored. Keep **goal_state_shift** for Disease→WT. |
| **analysis plots** | Post-stats figures (top genes barplot, volcano, etc.). |
| **cluster_coexpr_analysis** | Optional joint UMAP + L2-by-group after ISP UMAP / E2E TOP1 (`stages.isp.postprocess`). |
| **Trajectory UMAP** | Run type **ISP UMAP** (per-cell arrows), after E2E. E2E also runs **TOP1 significant ISP UMAP** automatically; Fine-tune UMAP comes from `stages.finetune.umap.enabled`. |
"""
        )


def _sync_species_form_from_yaml() -> None:
    cfg = _read_pipeline_yaml()
    species = cfg.get("species") or {}
    st.session_state.setdefault(
        "pipeline_model_organism", species.get("model_organism", "mouse")
    )
    st.session_state.setdefault(
        "pipeline_model", species.get("model", "mouse_geneformer")
    )
    st.session_state.setdefault(
        "pipeline_human_variant", species.get("human_variant", "v2_104m")
    )
    st.session_state.setdefault(
        "pipeline_mouse_variant", species.get("mouse_variant", "base")
    )
    st.session_state.setdefault(
        "pipeline_ortholog_policy", species.get("ortholog_policy", "one2one")
    )


def _native_model_organism(model_id: str) -> str:
    return "human" if model_id == "human_geneformer" else "mouse"


def _render_species_model_controls() -> None:
    """Input-data species vs FT/ISP model — clear separation for cross-species runs."""
    _sync_species_form_from_yaml()
    st.markdown("**Input data species**")
    st.selectbox(
        "Species of the uploaded scRNA-seq",
        list(_ORGANISM_LABELS.keys()),
        key="pipeline_model_organism",
        format_func=lambda v: _ORGANISM_LABELS.get(v, v),
        on_change=_apply_species_to_yaml,
        help=(
            "Organism of gene IDs in your 10x zip (not the pretrained model). "
            "Drosophila is Beta — pipeline smoke only; not biologically validated."
        ),
    )

    st.markdown("**Model for fine-tune / ISP**")
    st.selectbox(
        "Pretrained Geneformer",
        list(_MODEL_LABELS.keys()),
        key="pipeline_model",
        format_func=lambda v: _MODEL_LABELS.get(v, v),
        on_change=_apply_species_to_yaml,
        help="Checkpoint used for fine-tuning and in-silico perturbation.",
    )
    model_id = st.session_state.get("pipeline_model", "mouse_geneformer")
    if model_id == "mouse_geneformer":
        st.selectbox(
            "Mouse variant",
            list(_MOUSE_VARIANT_LABELS.keys()),
            key="pipeline_mouse_variant",
            format_func=lambda v: _MOUSE_VARIANT_LABELS.get(v, v),
            on_change=_apply_species_to_yaml,
            help="Base = 6L / ~10M; Large = 12L-E20 (same 2048 vocab). YAML: base | 12l_e20.",
        )
    else:
        st.selectbox(
            "Human variant",
            list(_HUMAN_VARIANT_LABELS.keys()),
            key="pipeline_human_variant",
            format_func=lambda v: _HUMAN_VARIANT_LABELS.get(v, v),
            on_change=_apply_species_to_yaml,
            help="Base = V2-104M (default); Large = V2-316M (more GPU). YAML: v2_104m | v2_316m.",
        )

    organism = st.session_state.get("pipeline_model_organism", "mouse")
    native = _native_model_organism(model_id)
    if organism != native:
        st.markdown("**Ortholog mapping (cross-species only)**")
        st.selectbox(
            "How to map genes when several orthologs exist",
            list(_ORTHOLOG_POLICY_LABELS.keys()),
            key="pipeline_ortholog_policy",
            format_func=lambda v: _ORTHOLOG_POLICY_LABELS.get(v, v),
            on_change=_apply_species_to_yaml,
            help=(
                "Only used when input species ≠ model species. "
                "Default one2one is safest for Geneformer rank tokens."
            ),
        )
        with st.expander("Which ortholog policy should I pick?", expanded=False):
            st.markdown(
                f"""
**This run converts `{organism}` → `{native}`.** Ambiguous orthologs need a rule:

| Choice | When to use | What it does |
|--------|-------------|--------------|
| **one2one** (default) | Almost always | Keep clear 1:1 pairs only. Drop ambiguous genes. **Never sum counts.** |
| **best_of_n** | You want more genes kept | If several genes map to one, keep the **highest-median** one. Still no summing. |
| **legacy_sum** | Reproducing an old run only | Old platform behavior: overwrite collisions + **sum** expression (can distort token ranks). |

**Tip:** Leave **one2one** unless you have a reason to change it.
Writes to YAML as `species.ortholog_policy`.
"""
            )
        st.info(
            f"Cross-species run: ortholog conversion **{organism} → {native}** "
            "runs automatically at tokenize and ISP."
        )
    else:
        st.caption(
            f"Same species as the model vocabulary ({native}) — no ortholog conversion "
            "(ortholog policy is unused)."
        )


def _poll_active_job() -> None:
    proc = st.session_state.get("active_proc")
    if proc is None:
        return
    code = proc.poll()
    if code is not None:
        st.session_state["active_proc"] = None
        st.session_state.pop("active_run_type", None)
        st.session_state["last_exit_code"] = code
        st.session_state["last_job_finished_utc"] = datetime.now(timezone.utc).isoformat()
        log_path = st.session_state.get("active_log_path")
        if log_path:
            st.session_state["last_log_path"] = log_path
        _sync_pipeline_output_run_dir()
        _ingest_ft_batch_calibrate_result()
        # Detect ortholog Block artifacts for the approval card.
        arts = _find_ortholog_block_artifacts(
            [
                st.session_state.get("pipeline_output_run_dir"),
                st.session_state.get("last_run_dir"),
            ]
        )
        if arts is not None:
            st.session_state["ortholog_block_request_path"] = str(arts.request_path)
            st.session_state["ortholog_block_run_dir"] = str(arts.run_dir)


def _ortholog_block_search_roots() -> list[str | Path | None]:
    return [
        st.session_state.get("pipeline_output_run_dir"),
        st.session_state.get("last_run_dir"),
        st.session_state.get("ortholog_block_run_dir"),
    ]


def _render_ortholog_approval_card(upload_dir: Path) -> None:
    """Show Block details and write approval_record.yaml (CLI remains authoritative)."""
    arts = _find_ortholog_block_artifacts(_ortholog_block_search_roots())
    if arts is None:
        return
    if str(arts.request.get("status") or "").lower() != "pending":
        return

    # Hide card after a successful follow-up run with approval.
    if st.session_state.get("ortholog_card_dismissed_for") == str(arts.request_path):
        return

    st.subheader("Ortholog mapping — approval required")
    st.error(
        "Tokenize was **blocked** by `ortholog_loss_gate`. "
        "The UI only records your decision; the CLI gate still verifies hashes."
    )
    st.caption(f"Pending request: `{arts.request_path}`")
    rows = _ortholog_card_rows(arts.request)
    st.table({"Item": [r[0] for r in rows], "Detail": [r[1] for r in rows]})
    st.info(
        "`best_of_n` sensitivity is **not** a one-click UI action — create a separate "
        "run with `ortholog_policy: best_of_n` manually if needed."
    )

    approved_by = st.text_input(
        "Approved by (required)",
        key="ortholog_approved_by",
        help="Audit identity for the approval_record (no login system).",
    )
    reason = st.text_area(
        "Reason (required)",
        key="ortholog_approval_reason",
        height=80,
    )
    action = st.radio(
        "Decision",
        options=[
            "Stop / keep strict one2one",
            "Approve curated bridge (presented candidate only)",
            "Reject candidate",
        ],
        key="ortholog_approval_action",
    )
    overlay = _resolve_ortholog_overlay_path(
        yaml_overlay=(
            ((_read_pipeline_yaml().get("species") or {}).get("ortholog_curated_overlay"))
        ),
        session_overlay=st.session_state.get("pipeline_ortholog_curated_overlay"),
    )
    if action.startswith("Approve"):
        st.caption(
            f"Bridge overlay (no free-form gene IDs): `{overlay}`"
            if overlay
            else "No curated overlay found — place `analysis/ortholog_policy/v1` or set "
            "`species.ortholog_curated_overlay`."
        )

    col_a, col_b = st.columns(2)
    with col_a:
        submit = st.button("Record decision", type="primary", key="ortholog_record_btn")
    with col_b:
        if st.button("Dismiss card", key="ortholog_dismiss_btn"):
            st.session_state["ortholog_card_dismissed_for"] = str(arts.request_path)
            st.rerun()

    if not submit:
        return
    if not str(approved_by or "").strip() or not str(reason or "").strip():
        st.warning("Fill **Approved by** and **Reason** before recording.")
        return

    if action.startswith("Stop"):
        ui_action = "stop"
    elif action.startswith("Approve"):
        ui_action = "curated_bridge"
        if overlay is None:
            st.error("Cannot approve curated bridge without an overlay path on disk.")
            return
    else:
        ui_action = "reject"

    sid = st.session_state.get("upload_session_id")
    try:
        written, record = _create_ortholog_approval_record(
            arts,
            action=ui_action,
            approved_by=approved_by,
            reason=reason,
            overlay_path=overlay if ui_action == "curated_bridge" else None,
            ui_session_id=str(sid) if sid else None,
        )
    except _OrthologApprovalError as exc:
        st.error(f"Approval rejected (fail-closed): {exc}")
        return
    except Exception as exc:
        st.error(f"Failed to write approval_record: {exc}")
        return

    st.success(f"Wrote `{written}` (status={record.get('status')}, decision={record.get('decision')})")
    st.session_state["ortholog_card_dismissed_for"] = str(arts.request_path)

    if ui_action != "curated_bridge":
        st.info("No re-run started (stop / reject). Tokenize remains blocked until a curated_bridge approval is used.")
        return

    audit = _resolve_ortholog_audit_path()
    st.session_state["pipeline_ortholog_approval_record"] = str(written)
    st.session_state["pipeline_ortholog_curated_overlay"] = str(overlay)
    if audit is not None:
        st.session_state["pipeline_ortholog_audit"] = str(audit)
    st.session_state["ortholog_auto_rerun"] = True
    st.info("Starting a **new** Pipeline run with `--ortholog-approval-record` + curated overlay…")
    st.rerun()


_UPLOAD_RUN_GUARD_JS = """
<!-- refresh %(nonce)s clear=%(clear)s -->
<script>
(function () {
  const doc = window.parent.document;
  const BUTTON = ".st-key-run_job_btn button";
  const INPUT = '[data-testid="stFileUploaderDropzoneInput"]';
  const DROPZONE = '[data-testid="stFileUploaderDropzone"]';
  const CLEAR = %(clear)s;
  // Max time the browser-upload lock may stick if the transfer never completes.
  const LOCK_MS = 30 * 60 * 1000;

  function applyLock() {
    const on = !!window.parent.__gfUploading;
    const button = doc.querySelector(BUTTON);
    if (!button) return false;
    // Never override Streamlit's own disabled styling; only the transient
    // "browser is still sending the zip" grey-out uses these inline styles.
    if (button.disabled && on) return true;
    button.style.opacity = on ? "0.4" : "";
    button.style.pointerEvents = on ? "none" : "";
    button.style.cursor = on ? "not-allowed" : "";
    button.title = on ? "Waiting for the upload to finish" : "";
    return true;
  }

  function setUploading(on) {
    window.parent.__gfUploading = !!on;
    window.parent.__gfUploadLockUntil = on ? (Date.now() + LOCK_MS) : 0;
    applyLock();
    return true;
  }

  // Persist the lock across Streamlit reruns (this iframe reloads every time).
  // Python clears it after the zip bytes arrive, when leaving Upload mode, or when
  // Run type is FT calibrate (study data not required). Pipeline keeps the lock
  // mid-upload so Run cannot start before the study is ready.
  if (CLEAR) {
    window.parent.__gfUploading = false;
    window.parent.__gfUploadLockUntil = 0;
  } else if (
    window.parent.__gfUploadLockUntil &&
    Date.now() > window.parent.__gfUploadLockUntil
  ) {
    window.parent.__gfUploading = false;
    window.parent.__gfUploadLockUntil = 0;
  }
  applyLock();

  if (!window.parent.__gfUploadClickGuard) {
    window.parent.__gfUploadClickGuard = true;
    doc.addEventListener(
      "click",
      function (e) {
        if (!window.parent.__gfUploading) return;
        const t = e.target;
        if (!t || !t.closest) return;
        if (t.closest(BUTTON)) {
          e.preventDefault();
          e.stopPropagation();
          e.stopImmediatePropagation();
        }
      },
      true
    );
  }

  function watch(selector, event, handler) {
    doc.querySelectorAll(selector).forEach(function (el) {
      if (el.dataset.uploadRunGuard) return;
      el.dataset.uploadRunGuard = "1";
      el.addEventListener(event, handler);
    });
  }

  function attach() {
    watch(INPUT, "change", function (e) {
      if (e.target.files && e.target.files.length) setUploading(true);
      else setUploading(false);
    });
    watch(DROPZONE, "drop", function () { setUploading(true); });
    applyLock();
  }

  if (!window.parent.__gfUploadRunGuard) {
    window.parent.__gfUploadRunGuard = true;
    attach();
    new MutationObserver(attach).observe(doc.body, {childList: true, subtree: true});
  } else {
    attach();
  }
})();
</script>
"""


def _render_upload_run_guard(*, clear_browser_upload_lock: bool = False) -> None:
    # Nonce forces the iframe to re-run after each Streamlit rerun without
    # re-installing the parent MutationObserver (guarded in JS).
    # clear_browser_upload_lock=True once Streamlit has the zip (or we left Upload mode);
    # False while zip_upload is still None so an in-flight browser transfer stays locked.
    st.iframe(
        _UPLOAD_RUN_GUARD_JS
        % {
            "nonce": time.time_ns(),
            "clear": "true" if clear_browser_upload_lock else "false",
        },
        height=1,
    )


# Short jobs (FT calibrate ~1 min) need frequent completion checks; long
# Pipeline / ISP runs keep a sparse interval so the UI stays quiet.
_LOG_AUTO_REFRESH_SHORT = timedelta(seconds=5)
_LOG_AUTO_REFRESH_LONG = timedelta(minutes=10)
_SHORT_POLL_RUN_TYPES = frozenset({RUN_TYPE_FT_BATCH})


def _render_log_body() -> None:
    """Show log tail + optional last-exit status (shared by busy / idle paths)."""
    log_path = st.session_state.get("active_log_path")
    if isinstance(log_path, Path):
        st.code(_tail_log(log_path), language="text")
    if st.session_state.get("last_exit_code") is not None:
        code = st.session_state["last_exit_code"]
        if code == 0:
            st.success(f"Last job finished OK (exit {code}).")
        elif isinstance(code, int) and code < 0:
            sig = -code
            st.warning(
                f"Last job was stopped by signal {sig} (exit {code}), "
                "not a pipeline failure."
            )
        else:
            st.error(f"Last job failed (exit {code}).")


def _live_log_busy_body(*, refresh_hint: str) -> None:
    proc = st.session_state.get("active_proc")
    if proc is not None and proc.poll() is not None:
        _poll_active_job()
        st.rerun()

    st.subheader("Logs & status")
    st.warning(
        f"Job running… ({refresh_hint} — click **Refresh log** for the latest lines)"
    )
    st.button("Refresh log", key="refresh_log_btn")
    _render_log_body()


@st.fragment(run_every=_LOG_AUTO_REFRESH_SHORT)
def _render_live_log_while_busy_short() -> None:
    """Frequent poll for short jobs (FT batch calibrate)."""
    _live_log_busy_body(refresh_hint="status refreshes every few seconds")


@st.fragment(run_every=_LOG_AUTO_REFRESH_LONG)
def _render_live_log_while_busy_long() -> None:
    """Sparse poll for long Pipeline / ISP jobs."""
    _live_log_busy_body(refresh_hint="auto-refreshes every 10 min")


def _render_live_log(busy: bool) -> None:
    if busy:
        active = st.session_state.get("active_run_type") or st.session_state.get(
            "run_type_sel", ""
        )
        if active in _SHORT_POLL_RUN_TYPES:
            _render_live_log_while_busy_short()
        else:
            _render_live_log_while_busy_long()
        return
    st.subheader("Logs & status")
    log_path = st.session_state.get("active_log_path")
    if isinstance(log_path, Path) and log_path.is_file():
        if st.button("Refresh log", key="refresh_log_idle_btn"):
            st.rerun()
    _render_log_body()


WEBUI_REPO_URL = "https://github.com/YuyaSanaki/ISP-Platform"

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})
_OUTPUT_BROWSER_CSV_NAMES = (
    "significant_genes.csv",
    "nominal_significant_genes_p0.05.csv",
    "top100_positive_shifters.csv",
    "top100_negative_shifters.csv",
)


def _default_output_root() -> Path:
    """Fixed platform output tree: /app/output in Docker, else repo output/."""
    docker_out = Path("/app/output")
    if docker_out.is_dir():
        return docker_out
    return (ROOT / "output").resolve()


def _resolve_under_output_root(path: Path, output_root: Path) -> Path | None:
    """Return resolved path only if it stays under output_root; else None."""
    try:
        resolved = path.resolve()
        root = output_root.resolve()
        resolved.relative_to(root)
        return resolved
    except (OSError, ValueError):
        return None


def _discover_browsable_runs(output_root: str | Path, limit: int = 80) -> list[Path]:
    """Newest pipeline_* folders under output_root (dated dirs only)."""
    root = Path(str(output_root))
    return _latest_pipeline_run_dirs(root, limit=limit)


def _run_browser_label(run_dir: Path) -> str:
    """Human-readable selectbox label for a past pipeline run."""
    date_part = run_dir.parent.name if run_dir.parent.name.isdigit() else ""
    prefix = f"{date_part}/" if date_part else ""
    n_figs = len(_list_figure_files(run_dir / "figures"))
    umap_dir = run_dir / "isp_umap"
    n_umap = len(_list_figure_files(umap_dir)) if umap_dir.is_dir() else 0
    bits: list[str] = []
    if n_figs:
        bits.append(f"{n_figs} fig")
    if n_umap:
        bits.append(f"{n_umap} umap")
    meta = run_dir / "pipeline_run_metadata.yaml"
    status = ""
    study = ""
    if meta.is_file():
        try:
            data = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
            status = str(data.get("run_status") or "").strip()
            resolved = data.get("resolved_paths") or {}
            study = str(resolved.get("study_name") or "").strip()
            if not study:
                cfg = run_dir / "pipeline_config_used.yaml"
                if cfg.is_file():
                    cfg_data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
                    study = str((cfg_data.get("data") or {}).get("output_prefix") or "").strip()
        except Exception:
            pass
    detail = ", ".join(bits) if bits else "no figures"
    extra = []
    if study:
        extra.append(study)
    if status:
        extra.append(status)
    tail = f" — {' · '.join(extra)}" if extra else ""
    return f"{prefix}{run_dir.name} ({detail}){tail}"


def _load_run_metadata_summary(run_dir: Path) -> dict[str, str]:
    """Small key→value map for the Output browser header."""
    out: dict[str, str] = {"path": str(run_dir)}
    meta = run_dir / "pipeline_run_metadata.yaml"
    if meta.is_file():
        try:
            data = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
            resolved = data.get("resolved_paths") or {}
            species = resolved.get("species") or {}
            out["study"] = str(resolved.get("study_name") or "")
            out["status"] = str(data.get("run_status") or "")
            out["started"] = str(data.get("started_at_utc") or "")
            out["finished"] = str(data.get("finished_at_utc") or "")
            out["organism"] = str(species.get("model_organism") or "")
            out["model"] = str(species.get("model") or "")
        except Exception:
            pass
    cfg = run_dir / "pipeline_config_used.yaml"
    if cfg.is_file() and not out.get("study"):
        try:
            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            out["study"] = str((data.get("data") or {}).get("output_prefix") or "")
            pert = data.get("perturbation") or {}
            if pert:
                out["perturbation"] = (
                    f"{pert.get('type', '')} "
                    f"{pert.get('start_state', '')}→{pert.get('end_state', '')}"
                ).strip()
        except Exception:
            pass
    return out


def _list_run_csv_previews(run_dir: Path, limit: int = 4) -> list[Path]:
    """Prefer known ISP stats CSVs; otherwise a few CSVs under ispstats_results."""
    stats = run_dir / "ispstats_results"
    found: list[Path] = []
    if stats.is_dir():
        for name in _OUTPUT_BROWSER_CSV_NAMES:
            fp = stats / name
            if fp.is_file():
                found.append(fp)
        if len(found) < limit:
            for fp in sorted(stats.glob("*.csv")):
                if fp not in found:
                    found.append(fp)
                if len(found) >= limit:
                    break
    return found[:limit]


def _render_figure_gallery(title: str, fig_dir: Path, key_prefix: str) -> None:
    """Show PNGs (and friends) inline; offer a figures zip when present."""
    files = [
        p
        for p in _list_figure_files(fig_dir)
        if p.suffix.lower() in _IMAGE_SUFFIXES
    ]
    if not files:
        st.caption(f"No image files under `{fig_dir}`.")
        return

    st.markdown(f"**{title}** (`{fig_dir.name}/` — {len(files)} file(s))")
    zip_payload = _build_figures_zip([fig_dir])
    if zip_payload:
        zip_bytes, zip_name = zip_payload
        st.download_button(
            label=f"Download {fig_dir.name} (.zip)",
            data=zip_bytes,
            file_name=zip_name,
            mime="application/zip",
            key=f"{key_prefix}_fig_zip",
        )

    # Prefer a stable display order: volcano / waterfall / top genes first when present.
    priority = (
        "volcano",
        "waterfall",
        "top_significant",
        "top_genes",
        "shift_distribution",
        "preview_sig",
        "preview_shift",
        "umap",
    )

    def sort_key(p: Path) -> tuple[int, str]:
        name = p.name.lower()
        for i, token in enumerate(priority):
            if token in name:
                return (i, name)
        return (len(priority), name)

    files = sorted(files, key=sort_key)
    cols = st.columns(2)
    for i, fp in enumerate(files):
        with cols[i % 2]:
            try:
                st.image(str(fp), caption=fp.name, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display `{fp.name}`: {e}")


def _render_output_browser() -> None:
    """Browse past pipeline runs under /app/output only (survives session disconnect)."""
    root = _default_output_root()
    st.caption(
        f"Past **Pipeline (E2E)** runs under `{root}` only "
        "(no other directories). Use this tab if the Analysis session was lost — "
        "figures and downloads are read from disk, not from the live job state."
    )
    st.code(str(root), language="text")
    runs = _discover_browsable_runs(root, limit=80)
    if not runs:
        st.info(
            f"No `pipeline_*` folders found under `{root}`. "
            "Run **Pipeline (E2E)** from the Analysis tab first."
        )
        return

    labels = [_run_browser_label(r) for r in runs]
    label_to_run = dict(zip(labels, runs))
    # Keep selection stable across refreshes when the same path still exists.
    prev = str(st.session_state.get("output_browser_selected_path") or "")
    default_idx = 0
    for i, r in enumerate(runs):
        safe = _resolve_under_output_root(r, root)
        if safe is not None and str(safe) == prev:
            default_idx = i
            break
    chosen_label = st.selectbox(
        "Past run",
        labels,
        index=default_idx,
        key="output_browser_run_select",
    )
    run_dir = label_to_run[chosen_label]
    safe_run = _resolve_under_output_root(run_dir, root)
    if safe_run is None:
        st.error(f"Refusing path outside `{root}`.")
        return
    run_dir = safe_run
    st.session_state["output_browser_selected_path"] = str(run_dir)

    summary = _load_run_metadata_summary(run_dir)
    c_meta, c_dl = st.columns([2, 1])
    with c_meta:
        bits = [f"`{run_dir}`"]
        if summary.get("study"):
            bits.append(f"study=`{summary['study']}`")
        if summary.get("status"):
            bits.append(f"status=`{summary['status']}`")
        if summary.get("organism") or summary.get("model"):
            bits.append(
                f"species=`{summary.get('organism', '')}` / model=`{summary.get('model', '')}`"
            )
        if summary.get("perturbation"):
            bits.append(summary["perturbation"])
        st.markdown(" · ".join(bits))
        if summary.get("started") or summary.get("finished"):
            st.caption(
                f"started={summary.get('started', '—')} · finished={summary.get('finished', '—')}"
            )
        for item in _pipeline_run_summary(run_dir):
            st.caption(f"  · {item}")
    with c_dl:
        prepared = st.session_state.get("output_browser_zip_for") == str(run_dir)
        if not prepared:
            if st.button(
                "Prepare download zip",
                type="secondary",
                key="output_browser_prep_zip",
                help="Figures, ISP stats, logs, configs, fine-tuned weights "
                "(skips tokenized_dataset / checkpoint-*).",
            ):
                with st.spinner("Building zip…"):
                    zip_payload = _build_pipeline_run_zip(run_dir)
                if zip_payload:
                    st.session_state["output_browser_zip_bytes"] = zip_payload[0]
                    st.session_state["output_browser_zip_name"] = zip_payload[1]
                    st.session_state["output_browser_zip_for"] = str(run_dir)
                else:
                    st.session_state.pop("output_browser_zip_bytes", None)
                    st.session_state.pop("output_browser_zip_name", None)
                    st.session_state.pop("output_browser_zip_for", None)
                    st.warning("No downloadable files found in this run folder.")
                st.rerun()
        else:
            st.download_button(
                label=f"Download run (.zip) — {st.session_state['output_browser_zip_name']}",
                data=st.session_state["output_browser_zip_bytes"],
                file_name=st.session_state["output_browser_zip_name"],
                mime="application/zip",
                key="output_browser_dl_zip",
            )
            if st.button("Clear prepared zip", type="secondary", key="output_browser_clear_zip"):
                st.session_state.pop("output_browser_zip_bytes", None)
                st.session_state.pop("output_browser_zip_name", None)
                st.session_state.pop("output_browser_zip_for", None)
                st.rerun()

    st.divider()
    _render_dropped_genes_panel([run_dir], key_prefix="output_browser")

    st.divider()
    fig_dir = run_dir / "figures"
    if fig_dir.is_dir():
        _render_figure_gallery("ISP analysis figures", fig_dir, "output_browser_main")
    else:
        st.caption("No `figures/` folder in this run yet.")

    umap_dir = run_dir / "isp_umap"
    if umap_dir.is_dir() and _list_figure_files(umap_dir):
        st.divider()
        _render_figure_gallery("ISP UMAP figures", umap_dir, "output_browser_umap")
    else:
        # UMAP may write umap_*.png directly under dated subfolders.
        nested = sorted(
            (
                p
                for p in umap_dir.glob("*")
                if p.is_dir() and _list_figure_files(p)
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ) if umap_dir.is_dir() else []
        for i, sub in enumerate(nested[:3]):
            st.divider()
            _render_figure_gallery(f"ISP UMAP — {sub.name}", sub, f"output_browser_umap_{i}")

    csvs = _list_run_csv_previews(run_dir)
    if csvs:
        st.divider()
        st.markdown("**ISP stats tables**")
        for fp in csvs:
            st.caption(f"`{fp.relative_to(run_dir)}`")
            try:
                import pandas as pd

                df = pd.read_csv(fp, nrows=50)
                st.dataframe(df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"Could not preview `{fp.name}`: {e}")
            st.download_button(
                label=f"Download {fp.name}",
                data=fp.read_bytes(),
                file_name=fp.name,
                mime="text/csv",
                key=f"output_browser_csv_{fp.name}",
            )


def main() -> None:
    st.set_page_config(page_title="ISP³ Platform", layout="wide")
    st.title("ISP³ Platform")
    st.caption(
        "Tokenize → fine-tune → ISP for Mouse or Human Geneformer. "
        "Choose the **input data species** and the **model for FT/ISP** separately."
    )
    st.markdown(f"[{WEBUI_REPO_URL}]({WEBUI_REPO_URL})")

    _poll_active_job()

    tab_analysis, tab_output = st.tabs(["Analysis", "Output"])
    with tab_analysis:
        _render_analysis_panel()
    with tab_output:
        _render_output_browser()


def _render_analysis_panel() -> None:
    """Current job UI: data input, run type, execute, live log, session outputs."""
    upload_dir = _session_upload_dir()
    st.session_state["upload_session_dir"] = str(upload_dir)
    if "pipeline_study_name" not in st.session_state:
        st.session_state["pipeline_study_name"] = _infer_study_name_from_yaml(upload_dir)

    # Run-guard flags (set in column 1 / 2; consumed by Execute).
    import_pending = False
    imported_now = False
    isp_states_invalid = False
    # True once Streamlit has the zip (or we are not in Upload mode). False while
    # zip_upload is still None so the browser-side transfer lock can persist.
    clear_browser_upload_lock = True

    c1, c2, c3 = st.columns(3)
    with c1:
        run_sel_c1 = st.session_state.get("run_type_sel", RUN_TYPE_PIPELINE)
        if run_sel_c1 == RUN_TYPE_ISP_UMAP:
            _render_isp_umap_source_picker()
        elif run_sel_c1 == RUN_TYPE_SEQUENTIAL_ISP:
            _render_sequential_isp_source_picker()
        else:
            st.subheader("Study name")
            st.text_input(
                "Study name",
                key="pipeline_study_name",
                label_visibility="collapsed",
                placeholder="Experiment name (e.g. MyExperiment)",
            )
            st.caption(
                "Enter the **experiment name** before uploading (becomes the study folder + "
                "`output_prefix`). Must match the name you use for Run."
            )
            st.subheader("Data input")
            st.caption(
                "Provide **10x sample folders** as a `.zip` (or `.tar.gz`): "
                "`Time-State-Suffix/` with `barcodes.tsv.gz`, `features.tsv.gz`, "
                "`matrix.mtx.gz` (optional `filtered_feature_bc_matrix/`). "
                "Zenodo / Figshare / GitHub Release **file** links work; "
                "Google Drive / Dropbox share links are rewritten to direct download. "
                "GEO / SRA / CELLxGENE **pages** need a supplementary archive URL."
            )
            data_source = st.radio(
                "Source",
                ["Upload .zip", "Download from URL"],
                horizontal=True,
                key="data_source_mode",
                label_visibility="collapsed",
            )

            if data_source == "Upload .zip":
                zip_upload = st.file_uploader(
                    "data.zip file",
                    type=["zip"],
                    key="zip_upload",
                    label_visibility="collapsed",
                )
                if zip_upload is not None:
                    # Bytes reached Streamlit — release the browser-transfer lock;
                    # import_pending / disabled=run_blocked cover the import phase.
                    clear_browser_upload_lock = True
                    zip_fp = _zip_upload_fingerprint(zip_upload)
                    if st.session_state.get("processed_zip_fingerprint") != zip_fp:
                        try:
                            with st.spinner("Importing study from zip…"):
                                _process_study_zip_upload(zip_upload, upload_dir)
                            imported_now = True
                        except (zipfile.BadZipFile, ValueError) as e:
                            st.error(str(e))
                        except Exception as e:
                            st.error(f"Upload failed: {e}")
                    import_pending = st.session_state.get("processed_zip_fingerprint") != zip_fp
                    if import_pending:
                        st.warning(
                            "This zip is not imported, so **Run job** is disabled. "
                            "Upload a valid zip, or remove this file to run on an earlier study."
                        )
                else:
                    # File picker empty: either idle, or browser still sending the zip.
                    # Do not clear the JS lock here or Run becomes clickable mid-upload.
                    clear_browser_upload_lock = False
                    study_name = normalize_study_name(_raw_study_name())
                    if study_name and study_folder(upload_dir, study_name).is_dir():
                        st.markdown(summarize_existing_study(upload_dir, study_name))
            else:
                st.text_input(
                    "Dataset URL",
                    key="remote_data_url",
                    placeholder="https://zenodo.org/records/.../files/study.zip?download=1",
                    help="Direct http(s) link to a .zip / .tar.gz of 10x sample folders.",
                )
                url_val = str(st.session_state.get("remote_data_url") or "").strip()
                url_busy = bool(st.session_state.get("url_fetch_active"))
                fetch_clicked = st.button(
                    "Download & import",
                    type="secondary",
                    key="url_fetch_btn",
                    disabled=url_busy or not url_val,
                )
                if fetch_clicked and url_val:
                    st.session_state["url_fetch_active"] = True
                    st.session_state["url_fetch_target"] = url_val
                    st.rerun()

                if st.session_state.get("url_fetch_active"):
                    import_pending = True
                    target_url = str(st.session_state.get("url_fetch_target") or url_val)
                    try:
                        with st.spinner("Downloading and importing study…"):
                            _process_study_url_import(target_url, upload_dir)
                        imported_now = True
                    except RemoteDataError as e:
                        st.error(str(e))
                    except (zipfile.BadZipFile, ValueError) as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Download / import failed: {e}")
                    finally:
                        st.session_state["url_fetch_active"] = False
                        st.session_state.pop("url_fetch_target", None)

                study_name = normalize_study_name(_raw_study_name())
                if (
                    not imported_now
                    and study_name
                    and study_folder(upload_dir, study_name).is_dir()
                ):
                    st.markdown(summarize_existing_study(upload_dir, study_name))
                last_src = st.session_state.get("last_data_source")
                if last_src:
                    st.caption(f"Last import: {last_src}")

    with c2:
        st.subheader("Run type")
        run_types = list(RUN_FILES.keys())
        default_idx = (
            run_types.index(RUN_TYPE_PIPELINE) if RUN_TYPE_PIPELINE in run_types else 0
        )
        st.selectbox(
            "Run type",
            run_types,
            index=default_idx,
            key="run_type_sel",
            label_visibility="collapsed",
            on_change=_load_default_yaml,
        )
        if "yaml_editor" not in st.session_state:
            _load_default_yaml()

        run_sel = st.session_state.get("run_type_sel", RUN_TYPE_PIPELINE)
        if run_sel == RUN_TYPE_PIPELINE:
            st.info(
                "Runs **Tokenize → Fine-tune → ISP** in one job. Set paths, species/model, "
                "and ISP states below (or edit full YAML). Outputs under "
                "`{paths.output_root}/{DATE}/pipeline_<UTC>/`."
            )
            _sync_pipeline_form_from_yaml(upload_dir)
            upload_states = _detected_states_from_upload(upload_dir)
            if upload_states:
                st.session_state["pipeline_detected_states"] = upload_states
            if "pipeline_isp_start_state" not in st.session_state:
                _sync_isp_state_selectors(
                    st.session_state.get("pipeline_detected_states"),
                    force=False,
                )
            detected = st.session_state.get("pipeline_detected_states") or []
            if detected:
                st.caption(f"Detected states from sample folders: **{', '.join(detected)}**")
            isp_options = st.session_state.get("pipeline_isp_state_options") or [
                "Disease",
                "Ctrl",
            ]
            c_start, c_end = st.columns(2)
            with c_start:
                st.selectbox(
                    "ISP start_state",
                    isp_options,
                    key="pipeline_isp_start_state",
                    on_change=_apply_isp_states_to_yaml,
                    help="Perturb cells in this condition (writes `perturbation.start_state`).",
                )
            with c_end:
                st.selectbox(
                    "ISP end_state",
                    isp_options,
                    key="pipeline_isp_end_state",
                    on_change=_apply_isp_states_to_yaml,
                    help="Goal state for in-silico shift (writes `perturbation.end_state`).",
                )
            pert = _read_pipeline_perturbation()
            sel_start = st.session_state.get("pipeline_isp_start_state")
            sel_end = st.session_state.get("pipeline_isp_end_state")
            st.caption(
                f"ISP: start=`{sel_start}` end=`{sel_end}` "
                f"(YAML end=`{pert.get('end_state', '')}` — updates when you change the dropdowns)"
            )
            if sel_start and sel_end and str(sel_start) == str(sel_end):
                isp_states_invalid = True
                st.error(
                    f"start_state and end_state are both `{sel_start}`. "
                    "Pick different values (e.g. WT → AD). **Run job** is disabled."
                )

            _render_perturbation_controls()

            _sync_ft_train_batch_controls()
            st.markdown("**Fine-tune train_batch_size**")
            st.caption(
                "Paste the value from **FT batch size (calibrate)** here. "
                "Once chosen for a study, **do not change it** — FT batch size changes "
                "optimization dynamics and makes results hard to compare "
                "(writes `runtime.train_batch_size`)."
            )
            c_ft_batch, c_ft_apply = st.columns([2, 1])
            with c_ft_batch:
                st.number_input(
                    "train_batch_size",
                    min_value=1,
                    max_value=1024,
                    step=1,
                    key="pipeline_ft_train_batch_size",
                    on_change=_apply_ft_train_batch_to_yaml,
                    help="Per-device fine-tune batch size (integer). Prefer the calibrated value.",
                )
            with c_ft_apply:
                calibrated = st.session_state.get("calibrated_ft_batch_size")
                if isinstance(calibrated, int) and calibrated >= 1:
                    # on_click runs before widgets are instantiated, so we can safely
                    # write pipeline_ft_train_batch_size (same key as the number_input).
                    st.button(
                        f"Use calibrated ({calibrated})",
                        type="secondary",
                        key="apply_calibrated_ft_batch_btn",
                        help="Copy the last FT batch calibration result into this field.",
                        on_click=_apply_calibrated_ft_batch_to_pipeline,
                    )
                else:
                    st.caption("No calibration yet")

            _sync_batch_size_controls()
            c_mode, c_size = st.columns(2)
            with c_mode:
                st.radio(
                    "ISP GPU batch size",
                    [BATCH_MODE_AUTO, BATCH_MODE_MANUAL],
                    key="pipeline_batch_mode",
                    horizontal=True,
                    on_change=_apply_batch_size_to_yaml,
                    help=(
                        "Auto measures this GPU when ISP starts and picks the largest batch "
                        "that still speeds things up (writes `runtime.forward_batch_size`)."
                    ),
                )
            with c_size:
                if st.session_state.get("pipeline_batch_mode") == BATCH_MODE_MANUAL:
                    st.number_input(
                        "forward_batch_size",
                        min_value=1,
                        max_value=4096,
                        step=8,
                        key="pipeline_batch_size",
                        on_change=_apply_batch_size_to_yaml,
                    )
                else:
                    st.caption(
                        "Measured at ISP startup (~1 min), then cached per GPU and model. "
                        "Calibrate on an idle GPU; a busy GPU yields a smaller batch."
                    )

            _render_species_model_controls()
            _render_ft_isp_advanced_controls(upload_dir)

            if st.button("Apply setting to Config YAML", type="secondary"):
                if _apply_study_settings_to_yaml(upload_dir):
                    _apply_species_to_yaml()
                    _apply_isp_states_to_yaml()
                    _apply_perturbation_controls_to_yaml()
                    _apply_batch_size_to_yaml()
                    _apply_ft_train_batch_to_yaml()
                    _apply_ft_isp_advanced_to_yaml()
                    st.session_state["pipeline_set_input_msg"] = (
                        "success",
                        f"Applied settings for `{st.session_state['pipeline_study_name']}` "
                        "to Config YAML.",
                    )
                st.rerun()
            msg = st.session_state.pop("pipeline_set_input_msg", None)
            if msg:
                kind, text = msg
                if kind == "success":
                    st.success(text)
                else:
                    st.warning(text)

        elif run_sel == RUN_TYPE_ISP_UMAP:
            st.info(
                "Per-cell **trajectory UMAP** for one gene, using a **past Pipeline ISP run** "
                "(left column). Outputs go under `{pipeline_run}/isp_umap/`. "
                "Plots match Fig.2 endpoint style (white background, UMAP-1 / UMAP-2 L-axes)."
            )
            st.caption(
                "Pick the source run + gene on the left, then **Apply run + gene to Config YAML** "
                "(optional preview) and **Run job**. Genome-wide ISP needs an explicit gene "
                "(suggestions come from `ispstats_results` when present)."
            )
            _render_isp_umap_plot_options()
            source = st.session_state.get("isp_umap_source_run_dir")
            gene = st.session_state.get("isp_umap_gene")
            if source and gene:
                st.success(f"Ready: `{Path(source).name}` · gene=`{gene}`")
            elif source:
                st.warning("Select a **gene** in the left column.")
            else:
                st.warning("Select a past ISP / pipeline run in the left column.")

        elif run_sel == RUN_TYPE_SEQUENTIAL_ISP:
            _render_sequential_isp_controls()

        elif run_sel == RUN_TYPE_FT_BATCH:
            st.info(
                "Measures this GPU and recommends a **fine-tune `train_batch_size`** "
                "(~1 min, no training; **study data not required** — only the pretrained "
                "model). **GPU compute must be idle** (util ~0%; stop any "
                "Pipeline / fine-tune first). Soft memory use on this host is OK. Then "
                "switch to **Pipeline (E2E)** and paste the value. Study **Data input** "
                "stays loaded when you switch."
            )
            _ingest_ft_batch_calibrate_result()
            _sync_pipeline_form_from_yaml(upload_dir)
            _render_species_model_controls()
            calibrated = st.session_state.get("calibrated_ft_batch_size")
            if isinstance(calibrated, int) and calibrated >= 1:
                st.success(
                    f"Recommended **train_batch_size = {calibrated}**. "
                    "Switch Run type to **Pipeline (E2E)** and use "
                    f"**Use calibrated ({calibrated})** (or paste `{calibrated}`)."
                )
                st.code(str(calibrated), language="text")
            else:
                st.caption(
                    "After **Run job** finishes, the recommended integer appears here "
                    "for copy/paste into Pipeline."
                )
            if st.button(
                "Apply setting to Config YAML",
                type="secondary",
                key="ft_calibrate_apply_btn",
            ):
                if _apply_study_settings_to_yaml(upload_dir):
                    _apply_species_to_yaml()
                    st.session_state["pipeline_set_input_msg"] = (
                        "success",
                        "Applied study / species settings to Config YAML.",
                    )
                else:
                    # Species alone is enough for calibration.
                    _apply_species_to_yaml()
                    st.session_state["pipeline_set_input_msg"] = (
                        "success",
                        "Applied species/model settings to Config YAML "
                        "(study data optional for calibration).",
                    )
                st.rerun()
            msg = st.session_state.pop("pipeline_set_input_msg", None)
            if msg:
                kind, text = msg
                if kind == "success":
                    st.success(text)
                else:
                    st.warning(text)

    with c3:
        st.subheader("Run directory")
        run_input_slot = st.empty()
        execute_slot = st.empty()
        run_output_slot = st.empty()
        if st.button("Reset YAML to template on disk", type="secondary"):
            _load_default_yaml()
            st.rerun()

    st.text_area("YAML", key="yaml_editor", height=420)

    if st.session_state.get("active_proc") is not None:
        _sync_pipeline_output_run_dir()

    run_label = st.session_state.get("run_type_sel", RUN_TYPE_PIPELINE)
    with run_input_slot.container():
        _render_run_directory_input(upload_dir, run_label)

    proc = st.session_state.get("active_proc")
    busy = proc is not None and proc.poll() is None
    url_busy = bool(st.session_state.get("url_fetch_active"))
    # FT calibrate / ISP UMAP / Sequential ISP do not need a fresh 10x zip upload.
    study_data_required = run_label not in (
        RUN_TYPE_FT_BATCH,
        RUN_TYPE_ISP_UMAP,
        RUN_TYPE_SEQUENTIAL_ISP,
    )
    if not study_data_required:
        clear_browser_upload_lock = True
    isp_umap_ready = True
    if run_label == RUN_TYPE_ISP_UMAP:
        isp_umap_ready = bool(
            str(st.session_state.get("isp_umap_source_run_dir") or "").strip()
            and str(st.session_state.get("isp_umap_gene") or "").strip()
        )
    seq_isp_ready = True
    if run_label == RUN_TYPE_SEQUENTIAL_ISP:
        seq_steps = _seq_isp_collect_steps()
        seq_isp_ready = bool(
            str(st.session_state.get("seq_isp_source_run_dir") or "").strip()
            and seq_steps
            and all(s.get("genes") for s in seq_steps)
        )
    # Block Run only while import/URL/job is actually in progress — not after
    # a successful load (imported_now). Disabling on imported_now left Run stuck
    # until the user changed Run type to force a rerun.
    run_blocked = busy or isp_states_invalid or not isp_umap_ready or not seq_isp_ready
    if study_data_required:
        run_blocked = run_blocked or import_pending or url_busy
    run_clicked = False
    with execute_slot.container():
        st.subheader("Execute")
        run_clicked = st.button(
            "Run job", type="primary", key="run_job_btn", disabled=run_blocked
        )
        if busy:
            st.caption("Disabled while a job is running.")
        elif run_label == RUN_TYPE_ISP_UMAP and not isp_umap_ready:
            st.caption(
                "Disabled until you select a past ISP / pipeline run and a gene "
                "in the left column."
            )
        elif run_label == RUN_TYPE_SEQUENTIAL_ISP and not seq_isp_ready:
            st.caption(
                "Disabled until you select a past ISP / pipeline run and enter "
                "genes for every sequential step."
            )
        elif not study_data_required:
            st.caption(
                "This run type does not need study Data input — **Run job** stays available "
                "while a zip uploads or imports in the background."
            )
        elif url_busy or (
            import_pending and st.session_state.get("data_source_mode") == "Download from URL"
        ):
            st.caption("Disabled while the dataset URL is downloading / importing.")
        elif import_pending:
            st.caption(
                "Disabled until the uploaded zip finishes importing. "
                "If this persists, clear the file from **Data input** and try again."
            )
        elif isp_states_invalid:
            st.caption("Disabled while ISP start_state and end_state are the same.")
    _render_upload_run_guard(clear_browser_upload_lock=clear_browser_upload_lock)

    with run_output_slot.container():
        _render_run_directory_output(run_label)

    # Drop a Run click that was queued in the browser while import was in progress
    # (previous page still had an enabled button during the spinner).
    # Skip for FT calibrate — upload/import must not cancel an intentional Run.
    if imported_now and study_data_required:
        st.session_state["discard_run_click_once"] = True
    elif st.session_state.pop("discard_run_click_once", False):
        if run_clicked:
            st.warning(
                "Data import just finished, so the previous **Run job** click was ignored. "
                "Press **Run job** again when you are ready."
            )
        run_clicked = False

    if run_clicked and isp_states_invalid:
        st.warning(
            "ISP start_state and end_state must differ. "
            "Fix the dropdowns, then click **Run job** again."
        )
        run_clicked = False
    elif run_clicked and run_label == RUN_TYPE_ISP_UMAP and not isp_umap_ready:
        st.warning(
            "Select a past Pipeline ISP run and a gene in the left column, "
            "then click **Run job** again."
        )
        run_clicked = False
    elif run_clicked and run_label == RUN_TYPE_SEQUENTIAL_ISP and not seq_isp_ready:
        st.warning(
            "Select a past Pipeline ISP run and enter genes for every sequential step, "
            "then click **Run job** again."
        )
        run_clicked = False
    elif run_clicked and study_data_required and (import_pending or url_busy):
        st.warning(
            "The study was still importing when **Run job** was clicked. "
            "Check the detected states below, then click **Run job** again."
        )
        run_clicked = False

    # Ortholog Approve curated_bridge → auto-launch a new Pipeline run.
    if st.session_state.pop("ortholog_auto_rerun", False):
        run_label = RUN_TYPE_PIPELINE
        st.session_state["run_type_sel"] = RUN_TYPE_PIPELINE
        run_clicked = True

    if run_clicked:
        run_label = st.session_state["run_type_sel"]
        yaml_text = st.session_state.get("yaml_editor", "")
        cfg_obj = None
        prep_failed = False

        if run_label == RUN_TYPE_PIPELINE:
            prepared, prep_err = _prepare_pipeline_yaml_for_run(upload_dir)
            if prep_err:
                st.error(prep_err)
                prep_failed = True
            else:
                yaml_text = prepared or yaml_text
        elif run_label == RUN_TYPE_FT_BATCH:
            prepared, prep_err = _prepare_ft_batch_calibrate_yaml_for_run(upload_dir)
            if prep_err:
                st.error(prep_err)
                prep_failed = True
            else:
                yaml_text = prepared or yaml_text
        elif run_label == RUN_TYPE_ISP_UMAP:
            source = Path(str(st.session_state.get("isp_umap_source_run_dir") or ""))
            gene = str(st.session_state.get("isp_umap_gene") or "").strip()
            # Do not write yaml_editor here — the text_area widget already exists
            # in this run (StreamlitAPIException if we mutate its key).
            prepared, apply_err = _build_isp_umap_yaml_from_pipeline_run(source, gene)
            if apply_err:
                st.error(apply_err)
                prep_failed = True
            else:
                yaml_text = prepared or yaml_text
                if source.is_dir():
                    st.session_state["isp_umap_source_run_dir"] = str(source.resolve())
                st.session_state["isp_umap_gene"] = gene
        elif run_label == RUN_TYPE_SEQUENTIAL_ISP:
            source = Path(str(st.session_state.get("seq_isp_source_run_dir") or ""))
            prepared, apply_err = _build_sequential_isp_yaml_from_pipeline_run(source)
            if apply_err:
                st.error(apply_err)
                prep_failed = True
            else:
                yaml_text = prepared or yaml_text
                if source.is_dir():
                    st.session_state["seq_isp_source_run_dir"] = str(source.resolve())

        if not prep_failed:
            try:
                cfg_obj = yaml.safe_load(yaml_text) or {}
            except yaml.YAMLError as e:
                st.error(f"Invalid YAML: {e}")
                cfg_obj = None

        if cfg_obj is not None:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
            run_dir = _ensure_workspace() / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            cfg_path = run_dir / "config.yaml"
            cfg_path.write_text(yaml_text, encoding="utf-8")
            log_path = run_dir / "console.log"

            try:
                cmd, env = _build_command_and_env(run_label, cfg_path)
            except ValueError as e:
                st.error(str(e))
                cmd, env = [], os.environ.copy()

            if cmd:
                st.session_state["last_run_dir"] = str(run_dir)
                st.session_state["last_output_roots"] = [
                    str(p) for p in _guess_output_roots(run_label, cfg_obj)
                ]
                st.session_state["last_exit_code"] = None
                if run_label == RUN_TYPE_PIPELINE:
                    st.session_state["pipeline_job_started_ts"] = time.time()
                    st.session_state.pop("pipeline_output_run_dir", None)
                    input_shown = str((cfg_obj.get("data") or {}).get("input_dir") or "")
                    species = cfg_obj.get("species") or {}
                    st.info(
                        f"Using `data.input_dir` = `{input_shown}` · "
                        f"organism=`{species.get('model_organism', '')}` · "
                        f"model=`{species.get('model', '')}` · "
                        f"train_batch_size=`{(cfg_obj.get('runtime') or {}).get('train_batch_size', '')}`"
                    )
                elif run_label == RUN_TYPE_FT_BATCH:
                    species = cfg_obj.get("species") or {}
                    st.info(
                        "Calibrating fine-tune batch size for "
                        f"organism=`{species.get('model_organism', '')}` · "
                        f"model=`{species.get('model', '')}` "
                        "(no training; ~1 min on an idle GPU)."
                    )
                elif run_label == RUN_TYPE_ISP_UMAP:
                    source = st.session_state.get("isp_umap_source_run_dir")
                    gene = st.session_state.get("isp_umap_gene")
                    st.info(
                        f"ISP UMAP from pipeline run `{source}` · gene=`{gene}` "
                        "(writes under that run’s `isp_umap/`)."
                    )
                elif run_label == RUN_TYPE_SEQUENTIAL_ISP:
                    source = st.session_state.get("seq_isp_source_run_dir")
                    n_steps = len((cfg_obj.get("sequential") or {}).get("steps") or [])
                    st.info(
                        f"Sequential ISP from pipeline run `{source}` · {n_steps} step(s) "
                        "(writes under that run’s `sequential_isp/`)."
                    )
                log_file = open(log_path, "w", encoding="utf-8", buffering=1)
                try:
                    p = subprocess.Popen(
                        cmd,
                        cwd=str(ROOT),
                        env=env,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                finally:
                    log_file.close()
                st.session_state["active_proc"] = p
                st.session_state["active_log_path"] = log_path
                st.session_state["active_run_type"] = run_label
                st.success(f"Started. Run folder: `{run_dir}`")
                st.rerun()

    _render_live_log(busy)

    st.subheader("Outputs")
    run_label = st.session_state.get("run_type_sel", "")
    if run_label == RUN_TYPE_FT_BATCH:
        calibrated = st.session_state.get("calibrated_ft_batch_size")
        if isinstance(calibrated, int) and calibrated >= 1:
            st.markdown(f"**Recommended fine-tune `train_batch_size`:** `{calibrated}`")
            st.code(str(calibrated), language="text")
            result_path = st.session_state.get("calibrated_ft_batch_result_path")
            if result_path:
                st.caption(f"Result file: `{result_path}`")
            st.caption(
                "Switch to **Pipeline (E2E)** and paste this integer into "
                "**Fine-tune train_batch_size** (or click **Use calibrated**). "
                "Keep it fixed afterward for comparable FT results."
            )
        else:
            st.caption(
                "No calibration result yet. Click **Run job**, then the recommended "
                "batch size appears here."
            )
    elif run_label == RUN_TYPE_PIPELINE:
        # Ortholog loss gate Block → approval card (CLI remains authoritative).
        _render_ortholog_approval_card(upload_dir)
        pipeline_run = st.session_state.get("pipeline_output_run_dir")
        if pipeline_run:
            run_path = Path(pipeline_run)
            st.markdown(f"**Pipeline run folder:** `{run_path}`")
            for item in _pipeline_run_summary(run_path):
                st.caption(f"  · {item}")
            st.caption(
                "Download excludes `tokenized_dataset/` and training `checkpoint-*` "
                "(multi-GB). Copy those from the run folder on the server if needed."
            )
            prepared = st.session_state.get("pipeline_zip_for") == str(run_path)
            if not prepared:
                if st.button("Prepare download zip (results + figures)", type="secondary"):
                    with st.spinner("Building zip…"):
                        zip_payload = _build_pipeline_run_zip(run_path)
                    if zip_payload:
                        st.session_state["pipeline_zip_bytes"] = zip_payload[0]
                        st.session_state["pipeline_zip_name"] = zip_payload[1]
                        st.session_state["pipeline_zip_for"] = str(run_path)
                    else:
                        st.session_state.pop("pipeline_zip_bytes", None)
                        st.session_state.pop("pipeline_zip_name", None)
                        st.session_state.pop("pipeline_zip_for", None)
                        st.warning("No downloadable files found in this run folder.")
                    st.rerun()
            else:
                st.download_button(
                    label=f"Download pipeline run (.zip) — {st.session_state['pipeline_zip_name']}",
                    data=st.session_state["pipeline_zip_bytes"],
                    file_name=st.session_state["pipeline_zip_name"],
                    mime="application/zip",
                    key="dl_pipeline_run_zip",
                    help=(
                        "Figures, ISP stats/results, logs, configs, and the fine-tuned "
                        "model weights (no tokenized dataset / intermediate checkpoints)."
                    ),
                )
                if st.button("Clear prepared zip from memory", type="secondary"):
                    st.session_state.pop("pipeline_zip_bytes", None)
                    st.session_state.pop("pipeline_zip_name", None)
                    st.session_state.pop("pipeline_zip_for", None)
                    st.rerun()
        _render_dropped_genes_panel(
            [pipeline_run, st.session_state.get("last_run_dir")],
            key_prefix="analysis_outputs",
        )

    roots = st.session_state.get("last_output_roots") or []
    if roots:
        st.markdown("**Output roots from YAML / recent pipeline runs:**")
        seen: set[str] = set()
        for r in roots:
            p = Path(r)
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            st.write(f"- `{p}` — exists: {p.is_dir()}")
            if p.is_dir():
                try:
                    subs = sorted(
                        p.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True
                    )[:12]
                    for s in subs:
                        st.caption(f"  · `{s.name}`")
                    if p.name.startswith("pipeline_"):
                        for sub in ("isp_results", "finetune", "tokenized_dataset"):
                            sp = p / sub
                            if sp.exists():
                                st.caption(f"  → `{sub}/`")
                except OSError:
                    pass
    last_run = st.session_state.get("last_run_dir")
    fig_dirs = _discover_figures_dirs(roots, run_label)
    zip_payload = _build_figures_zip(fig_dirs) if fig_dirs else None
    if zip_payload:
        zip_bytes, zip_name = zip_payload
        st.download_button(
            label=f"Download figures (.zip) — {zip_name}",
            data=zip_bytes,
            file_name=zip_name,
            mime="application/zip",
            key="dl_figures_zip",
            help=(
                "PNG/PDF figures from the newest pipeline or ISP run folder "
                "under the output roots above."
            ),
        )
        for fd in fig_dirs[:3]:
            n = len(_list_figure_files(fd))
            st.caption(f"Included: `{fd}` ({n} file(s))")
    elif roots and st.session_state.get("last_exit_code") == 0:
        st.caption(
            "No figure files found yet under the listed output roots "
            "(expected `figures/*.png` under `pipeline_*` or `isp_*`, "
            "or `umap_*.png` for ISP UMAP)."
        )

    if last_run:
        st.markdown(f"**Last run metadata:** `{last_run}`")
        for name in ("config.yaml", "console.log"):
            fp = Path(last_run) / name
            if fp.is_file():
                st.download_button(
                    label=f"Download {name}",
                    data=fp.read_bytes(),
                    file_name=f"{Path(last_run).name}_{name}",
                    key=f"dl_{name}",
                )


if __name__ == "__main__":
    main()
