"""
WebUI helpers for ortholog_loss_gate Block → approval_record (CLI remains authoritative).

This module must not bypass hash checks: it only discovers artifacts, presents
choices, and asks ``geneformer.ortholog_loss_gate`` to write immutable records.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parent.parent
_CORE = ROOT / "core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

from geneformer.ortholog_loss_gate import (  # noqa: E402
    OrthologApprovalError,
    POU5F1B_ENSEMBL,
    assert_request_matches_disk,
    build_approved_record,
    write_approval_record,
)

DEFAULT_PROJECT_OVERLAY = ROOT / "analysis" / "ortholog_policy" / "v1"
DEFAULT_PROJECT_AUDIT = ROOT / "analysis" / "analysis_manifest.yaml"


@dataclass
class OrthologBlockArtifacts:
    request_path: Path
    summary_path: Path | None
    audit_tsv: Path | None
    run_dir: Path
    request: dict[str, Any]
    summary: dict[str, Any]


def find_ortholog_block_artifacts(
    search_roots: Sequence[str | Path | None],
) -> OrthologBlockArtifacts | None:
    """
    Locate the newest ``ortholog_approval_request.yaml`` under the given roots
    (pipeline run folder, tokenized_dataset, WEBUI workspace run, …).
    """
    candidates: list[Path] = []
    for root in search_roots:
        if not root:
            continue
        base = Path(str(root)).expanduser()
        if not base.exists():
            continue
        if base.is_file() and base.name == "ortholog_approval_request.yaml":
            candidates.append(base)
            continue
        if base.is_dir():
            candidates.extend(base.rglob("ortholog_approval_request.yaml"))
    if not candidates:
        return None
    req_path = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        request = assert_request_matches_disk(req_path)
    except (OrthologApprovalError, OSError, ValueError):
        with open(req_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, Mapping) or str(raw.get("status")).lower() != "pending":
            return None
        request = dict(raw)

    run_dir = req_path.parent
    summary_path = run_dir / "ortholog_loss_summary.json"
    if not summary_path.is_file():
        alt = req_path.parent.parent / "ortholog_loss_summary.json"
        summary_path = alt if alt.is_file() else None
    audit_tsv = run_dir / "critical_gene_audit.tsv"
    if not audit_tsv.is_file():
        audit_tsv = None

    summary: dict[str, Any] = {}
    if summary_path and summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}

    return OrthologBlockArtifacts(
        request_path=req_path,
        summary_path=summary_path,
        audit_tsv=audit_tsv,
        run_dir=run_dir,
        request=request,
        summary=summary,
    )


def card_rows_from_request(request: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Table rows for the Streamlit approval card."""
    blocked = request.get("blocked_run") or {}
    genes = list(request.get("blocked_genes") or [])
    primary = request.get("blocked_gene") if isinstance(request.get("blocked_gene"), Mapping) else None
    if primary is None and genes:
        primary = genes[0]
    primary = primary or {}
    direction = blocked.get("direction") or "?"
    src = primary.get("source_symbol") or primary.get("source_id") or "?"
    src_id = primary.get("source_id") or ""
    status = primary.get("mapping_status") or "?"
    cand = primary.get("candidate_target_id") or ""
    cands = primary.get("candidate_target_ids") or ([] if not cand else [cand])
    cand_disp = ", ".join(str(x) for x in cands if x) or "(none)"
    return [
        ("Direction", str(direction)),
        ("Critical gene lost", f"{src} / {src_id}".strip(" /")),
        ("Present in input", "Yes (Block = present + policy drop)"),
        ("one2one conversion", status),
        ("Candidate target(s)", cand_disp),
        ("Excluded paralog", f"POU5F1B / {POU5F1B_ENSEMBL}"),
        (
            "Impact",
            "OSKM landmark may be absent from swapped tokenization / ISP endpoints.",
        ),
    ]


def resolve_default_overlay_path(
    *,
    yaml_overlay: str | None = None,
    session_overlay: str | None = None,
) -> Path | None:
    for raw in (session_overlay, yaml_overlay, str(DEFAULT_PROJECT_OVERLAY)):
        if not raw:
            continue
        p = Path(str(raw)).expanduser()
        if p.exists():
            return p.resolve()
    return None


def resolve_default_audit_path() -> Path | None:
    if DEFAULT_PROJECT_AUDIT.is_file():
        return DEFAULT_PROJECT_AUDIT.resolve()
    return None


def create_approval_record_from_ui(
    artifacts: OrthologBlockArtifacts,
    *,
    action: str,
    approved_by: str,
    reason: str,
    overlay_path: str | Path | None = None,
    ui_session_id: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """
    Validate pending request and write ``approval_record.yaml`` beside it.

    ``action``: stop | curated_bridge | reject
    """
    req = assert_request_matches_disk(artifacts.request_path)
    action_norm = str(action).strip().lower()
    if action_norm not in {"stop", "curated_bridge", "reject"}:
        raise OrthologApprovalError(
            f"UI action must be stop|curated_bridge|reject (got {action!r})."
        )

    record = build_approved_record(
        req,
        decision=action_norm,
        approved_by=approved_by,
        reason=reason,
        curated_overlay_path=overlay_path if action_norm == "curated_bridge" else None,
        ui_session_id=ui_session_id,
    )
    out_path = artifacts.run_dir / "approval_record.yaml"
    written = write_approval_record(out_path, record, overwrite=False)
    return written, record
