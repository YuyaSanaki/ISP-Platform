"""
Deterministic Pass / Warn / Block gate for cross-species ortholog conversion.

Block applies only when a critical gene is present in the raw input feature set
but is dropped by the mapping policy (e.g. one2one filtering of ortholog_one2many).
Critical genes absent from the input are recorded as absent_from_input (Warn), never Block.

One-to-many genes are never auto-added to curated overlays. Approval is fail-closed:
Block writes ``ortholog_approval_request.yaml`` with ``status: pending`` (not reusable);
a separately authored ``status: approved`` record must match run fingerprints.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .backends.registry import parse_species_config
from .conversion_report import collect_loom_gene_ids
from .gene_converter import (
    ORTHOLOGS_DIR,
    ConversionPair,
    GeneMappingInspection,
    _PAIR_TO_FILE,
    _curated_overlay_from_species,
    _read_ortholog_rows,
    conversion_pair,
    inspect_gene_mappings,
    is_ensembl_id,
    is_fly_id,
    normalize_gene_id,
    parse_ortholog_policy,
    resolve_curated_overlay_path,
    should_convert,
)
from .species_context import load_input_symbol_table

DEFAULT_PASS_MAPPED_PCT_MIN = 80.0
DEFAULT_WARN_MAPPED_PCT_MIN = 90.0
DEFAULT_BLOCK_MAPPED_PCT_MIN = 50.0

# Human POU5F1B paralog — must never appear in curated overlays.
POU5F1B_ENSEMBL = "ENSG00000212993"

_DROP_STATUSES = frozenset(
    {
        "ortholog_one2many",
        "ortholog_many2many",
        "no_ortholog",
        "dropped_ambiguous",
        "symbol_unresolved",
    }
)
_MAPPED_STATUSES = frozenset(
    {"mapped", "mapped_one2one", "curated_bridge", "mapped_native"}
)

_DECISION_ALIASES = {
    "A": "stop",
    "STOP": "stop",
    "B": "curated_bridge",
    "CURATED_BRIDGE": "curated_bridge",
    "C": "sensitivity_best_of_n",
    "SENSITIVITY_BEST_OF_N": "sensitivity_best_of_n",
    "SENSITIVITY": "sensitivity_best_of_n",
    "D": "mark_non_critical",
    "MARK_NON_CRITICAL": "mark_non_critical",
}


class OrthologApprovalError(ValueError):
    """Raised when an approval record is pending, malformed, or hash-mismatched."""


class OrthologLossGateError(RuntimeError):
    """Raised when the gate blocks tokenization pending user approval."""

    def __init__(self, message: str, *, result: "GateResult"):
        super().__init__(message)
        self.result = result


@dataclass
class CriticalGeneAuditRow:
    gene: str
    critical_set: str
    input_presence: str  # present_in_input | absent_from_input
    mapping_status: str
    source_feature: str = ""
    selected_target: str = ""
    candidates: list[str] = field(default_factory=list)
    orthology_types: list[str] = field(default_factory=list)
    block_eligible: bool = False
    reaches_token: bool = False


@dataclass
class GateResult:
    verdict: str  # pass | warn | block | stopped | not_applicable
    conversion_pair: str | None
    ortholog_policy: str
    mapped_pct: float
    input_genes: int
    mapped: int
    unmapped: int
    critical_rows: list[CriticalGeneAuditRow] = field(default_factory=list)
    gene_inspections: list[GeneMappingInspection] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    approval_request: dict[str, Any] = field(default_factory=dict)
    fingerprints: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    artifact_paths: dict[str, str] = field(default_factory=dict)
    continue_tokenize: bool = True

    @property
    def approval_stub(self) -> dict[str, Any]:
        """Legacy alias for ``approval_request``."""
        return self.approval_request

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "conversion_pair": self.conversion_pair,
            "ortholog_policy": self.ortholog_policy,
            "mapped_pct": self.mapped_pct,
            "input_genes": self.input_genes,
            "mapped": self.mapped,
            "unmapped": self.unmapped,
            "summary": self.summary,
            "message": self.message,
            "continue_tokenize": self.continue_tokenize,
            "fingerprints": self.fingerprints,
            "critical_present_dropped": [
                {
                    "gene": r.gene,
                    "critical_set": r.critical_set,
                    "mapping_status": r.mapping_status,
                    "candidates": r.candidates,
                    "source_feature": r.source_feature,
                }
                for r in self.critical_rows
                if r.block_eligible
            ],
            "critical_absent_from_input": [
                {
                    "gene": r.gene,
                    "critical_set": r.critical_set,
                }
                for r in self.critical_rows
                if r.input_presence == "absent_from_input"
            ],
        }


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _git_commit_short() -> str | None:
    env = os.environ.get("TOKENIZE_GIT_COMMIT") or os.environ.get("GIT_COMMIT")
    if env:
        return str(env).strip()[:40] or None
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            timeout=5,
        )
        return out.strip() or None
    except Exception:
        return None


def _symbol_lookup_ci(table: Mapping[str, str], symbol: str) -> str | None:
    if symbol in table:
        return table[symbol]
    for variant in (symbol.upper(), symbol.lower(), symbol.capitalize()):
        if variant in table:
            return table[variant]
    return None


def normalize_decision(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return _DECISION_ALIASES.get(text.upper(), text.lower())


def input_feature_sha256(gene_ids: Sequence[str]) -> str:
    """SHA-256 of sorted unique input feature IDs (Ensembl IDs normalized)."""
    features: set[str] = set()
    for g in gene_ids:
        raw = str(g).strip()
        if not raw:
            continue
        if is_ensembl_id(raw) or is_fly_id(raw):
            features.add(normalize_gene_id(raw))
        else:
            features.add(raw)
    payload = "\n".join(sorted(features))
    if payload:
        payload += "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_sha256(
    audit_path: str | Path | None = None,
    *,
    audit_cfg: Mapping[str, Any] | None = None,
) -> str:
    """SHA-256 of the audit/manifest file, or of canonical audit_cfg JSON."""
    if audit_path:
        p = Path(str(audit_path)).expanduser()
        if p.is_file():
            digest = _sha256_file(p)
            if digest:
                return digest
    if audit_cfg is not None:
        return hashlib.sha256(_canonical_json(dict(audit_cfg)).encode("utf-8")).hexdigest()
    return ""


def build_run_fingerprints(
    gene_ids: Sequence[str],
    species: Mapping[str, str] | None,
    *,
    audit_path: str | Path | None = None,
    audit_cfg: Mapping[str, Any] | None = None,
    orthologs_dir: Path | None = None,
    critical_gene_audit_sha256: str | None = None,
) -> dict[str, Any]:
    """Fingerprints used in blocked_run / approval matching."""
    parsed = parse_species_config(species)
    pair = conversion_pair(parsed["model_organism"], parsed["model"])
    base = orthologs_dir or ORTHOLOGS_DIR
    main_path = (base / _PAIR_TO_FILE[pair]) if pair else None
    mapping_src = _sha256_file(main_path) if main_path else None
    overlay_raw = _curated_overlay_from_species(parsed)
    overlay_path = (
        resolve_curated_overlay_path(pair, curated_overlay=overlay_raw) if pair else None
    )
    overlay_digest = _sha256_file(overlay_path) if overlay_path else None
    return {
        "direction": pair.value if pair else None,
        "input_feature_sha256": input_feature_sha256(gene_ids),
        "manifest_sha256": manifest_sha256(audit_path, audit_cfg=audit_cfg),
        "converter_git_commit": _git_commit_short(),
        "mapping_source_sha256": mapping_src or "",
        "critical_gene_audit_sha256": critical_gene_audit_sha256 or "",
        "overlay_path": str(overlay_path) if overlay_path else None,
        "overlay_sha256": overlay_digest,
        "model_organism": parsed.get("model_organism"),
        "model": parsed.get("model"),
    }


def mint_request_id(blocked_run: Mapping[str, Any]) -> str:
    """``sha256:`` + hex digest of canonical JSON of blocked_run."""
    # Only stable identity fields participate in the request id.
    payload = {
        "direction": blocked_run.get("direction"),
        "input_feature_sha256": blocked_run.get("input_feature_sha256"),
        "manifest_sha256": blocked_run.get("manifest_sha256"),
        "converter_git_commit": blocked_run.get("converter_git_commit"),
        "mapping_source_sha256": blocked_run.get("mapping_source_sha256"),
        "critical_gene_audit_sha256": blocked_run.get("critical_gene_audit_sha256"),
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def build_approval_request(
    fingerprints: Mapping[str, Any],
    present_dropped: Sequence[CriticalGeneAuditRow],
    *,
    absent: Sequence[CriticalGeneAuditRow] | None = None,
    policy: str | None = None,
) -> dict[str, Any]:
    blocked_run = {
        "direction": fingerprints.get("direction"),
        "input_feature_sha256": fingerprints.get("input_feature_sha256"),
        "manifest_sha256": fingerprints.get("manifest_sha256"),
        "converter_git_commit": fingerprints.get("converter_git_commit"),
        "mapping_source_sha256": fingerprints.get("mapping_source_sha256"),
        "critical_gene_audit_sha256": fingerprints.get("critical_gene_audit_sha256")
        or "",
    }
    blocked_genes = [
        {
            "source_id": r.source_feature or "",
            "source_symbol": r.gene,
            "mapping_status": r.mapping_status,
            "candidate_target_ids": list(r.candidates),
            "candidate_target_id": r.candidates[0] if r.candidates else "",
            "orthology_types": list(r.orthology_types),
            "critical_set": r.critical_set,
        }
        for r in present_dropped
    ]
    req: dict[str, Any] = {
        "status": "pending",
        "request_id": mint_request_id(blocked_run),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "blocked_run": blocked_run,
        "blocked_genes": blocked_genes,
        "critical_gene_not_in_raw_input": [
            {"gene": r.gene, "critical_set": r.critical_set} for r in (absent or [])
        ],
        "policy": policy,
        "decisions": {
            "stop": "Stop: retain strict one2one; skip tokenize (alias A)",
            "curated_bridge": (
                "Approve curated bridge (requires active overlay; alias B); "
                "never auto-writes TSV"
            ),
            "sensitivity_best_of_n": (
                "Sensitivity branch via best_of_n — manual re-tokenize in v1 (alias C)"
            ),
            "mark_non_critical": (
                "Mark listed genes as non-critical for this analysis (alias D)"
            ),
        },
        "notes": (
            "This file has status: pending and MUST NOT be passed to "
            "--ortholog-approval-record. Create a NEW YAML with status: approved, "
            "the same request_id and blocked_run hashes, plus decision / approver."
        ),
    }
    if blocked_genes:
        primary = dict(blocked_genes[0])
        primary["candidate_target_symbol"] = ""
        req["blocked_gene"] = primary
    return req


def validate_overlay_excludes_paralog(
    species: Mapping[str, str] | None,
    *,
    forbidden_ids: Sequence[str] | None = None,
    orthologs_dir: Path | None = None,
) -> None:
    """
    Raise OrthologApprovalError if the active curated overlay contains POU5F1B
    (or other forbidden Ensembl IDs) as source or target.
    """
    forbidden = {
        normalize_gene_id(x) for x in (forbidden_ids or [POU5F1B_ENSEMBL]) if x
    }
    parsed = parse_species_config(species)
    pair = conversion_pair(parsed["model_organism"], parsed["model"])
    if pair is None:
        return
    overlay_raw = _curated_overlay_from_species(parsed)
    overlay_path = resolve_curated_overlay_path(pair, curated_overlay=overlay_raw)
    if overlay_path is None:
        return
    rows = _read_ortholog_rows(overlay_path)
    for row in rows:
        src = normalize_gene_id(row.source) if is_ensembl_id(row.source) else row.source
        tgt = normalize_gene_id(row.target) if is_ensembl_id(row.target) else row.target
        if src in forbidden or tgt in forbidden:
            raise OrthologApprovalError(
                f"Curated overlay must not include paralog {POU5F1B_ENSEMBL} "
                f"(found in {overlay_path}: {row.source} → {row.target})."
            )


def validate_approval_record(
    approval: Mapping[str, Any],
    fingerprints: Mapping[str, Any],
    *,
    overlay_sha256: str | None = None,
) -> str:
    """
    Fail-closed validation of an approval record against the current run.

    Returns the normalized decision string.
    Raises OrthologApprovalError on pending / mismatch / missing fields.
    """
    status = str(approval.get("status") or "").strip().lower()
    if status == "pending":
        raise OrthologApprovalError(
            "status: pending cannot be used as --ortholog-approval-record; "
            "create a NEW file with status: approved."
        )
    if status != "approved":
        raise OrthologApprovalError(
            f"approval status must be 'approved' (got {status!r})."
        )

    blocked = approval.get("blocked_run")
    if not isinstance(blocked, Mapping):
        raise OrthologApprovalError("approved record missing blocked_run mapping.")

    request_id = str(approval.get("request_id") or "").strip()
    expected_id = mint_request_id(blocked)
    if not request_id or request_id != expected_id:
        raise OrthologApprovalError(
            "request_id does not match sha256 of blocked_run "
            f"(got {request_id!r}, expected {expected_id!r})."
        )

    for key in (
        "direction",
        "input_feature_sha256",
        "manifest_sha256",
        "mapping_source_sha256",
    ):
        got = blocked.get(key)
        want = fingerprints.get(key)
        if got != want:
            raise OrthologApprovalError(
                f"blocked_run.{key} mismatch with current run "
                f"(approval={got!r}, current={want!r})."
            )

    decision = normalize_decision(
        approval.get("decision") or approval.get("choice")
    )
    if not decision:
        raise OrthologApprovalError(
            "approved record requires decision "
            "(stop|curated_bridge|sensitivity_best_of_n|mark_non_critical or A|B|C|D)."
        )
    if decision not in (
        "stop",
        "curated_bridge",
        "sensitivity_best_of_n",
        "mark_non_critical",
    ):
        raise OrthologApprovalError(f"unknown approval decision: {decision!r}")

    if decision == "curated_bridge":
        co = approval.get("curated_overlay") or {}
        if not isinstance(co, Mapping):
            raise OrthologApprovalError(
                "curated_bridge decision requires curated_overlay.path and sha256."
            )
        approved_hash = str(co.get("sha256") or "").strip()
        current_hash = (overlay_sha256 or fingerprints.get("overlay_sha256") or "")
        current_hash = str(current_hash).strip()
        if not approved_hash or not current_hash or approved_hash != current_hash:
            raise OrthologApprovalError(
                "curated_bridge overlay sha256 mismatch "
                f"(approval={approved_hash!r}, current={current_hash!r})."
            )
    return decision


def load_approval_request(path: str | Path) -> dict[str, Any]:
    """Load a Block-time ``ortholog_approval_request.yaml`` (must be pending)."""
    p = Path(str(path)).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"ortholog approval request not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Invalid approval request: {p}")
    return dict(data)


def assert_request_matches_disk(request_path: str | Path) -> dict[str, Any]:
    """
    Fail-closed check that the on-disk request is pending and internally consistent.

    Does not unlock a Block by itself — only validates the pending request file.
    """
    req = load_approval_request(request_path)
    status = str(req.get("status") or "").strip().lower()
    if status != "pending":
        raise OrthologApprovalError(
            f"approval request must have status: pending (got {status!r})."
        )
    blocked = req.get("blocked_run")
    if not isinstance(blocked, Mapping):
        raise OrthologApprovalError("approval request missing blocked_run.")
    request_id = str(req.get("request_id") or "").strip()
    expected = mint_request_id(blocked)
    if request_id != expected:
        raise OrthologApprovalError(
            f"request_id does not match blocked_run "
            f"(got {request_id!r}, expected {expected!r})."
        )
    return req


def build_approved_record(
    request: Mapping[str, Any],
    *,
    decision: str,
    approved_by: str,
    reason: str,
    curated_overlay_path: str | Path | None = None,
    approved_bridge: Mapping[str, Any] | None = None,
    explicitly_excluded: Sequence[str] | None = None,
    ui_session_id: str | None = None,
) -> dict[str, Any]:
    """
    Build an immutable approval / rejection record from a pending request.

    Copies ``request_id`` and ``blocked_run`` verbatim from the request.
    For ``curated_bridge``, pins overlay path + sha256 from the live overlay file.
    ``reject`` yields ``status: rejected`` (not accepted by the gate for re-run).
    """
    status_req = str(request.get("status") or "").strip().lower()
    if status_req != "pending":
        raise OrthologApprovalError(
            "build_approved_record requires a pending approval request "
            f"(got status={request.get('status')!r})."
        )
    blocked = request.get("blocked_run")
    if not isinstance(blocked, Mapping):
        raise OrthologApprovalError("request missing blocked_run.")
    request_id = str(request.get("request_id") or "").strip()
    if not request_id or request_id != mint_request_id(blocked):
        raise OrthologApprovalError(
            "request_id / blocked_run integrity check failed before recording approval."
        )
    who = str(approved_by or "").strip()
    why = str(reason or "").strip()
    if not who:
        raise OrthologApprovalError("approved_by is required.")
    if not why:
        raise OrthologApprovalError("reason is required.")

    norm = normalize_decision(decision) or str(decision).strip().lower()
    if norm == "reject":
        record: dict[str, Any] = {
            "status": "rejected",
            "request_id": request_id,
            "approved_by": who,
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "decision": "reject",
            "reason": why,
            "blocked_run": dict(blocked),
            "blocked_genes": list(request.get("blocked_genes") or []),
        }
        if ui_session_id:
            record["ui_session_id"] = str(ui_session_id)
        return record

    if norm not in (
        "stop",
        "curated_bridge",
        "sensitivity_best_of_n",
        "mark_non_critical",
    ):
        raise OrthologApprovalError(f"unsupported decision for approval record: {decision!r}")

    record = {
        "status": "approved",
        "request_id": request_id,
        "approved_by": who,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "decision": norm,
        "reason": why,
        "blocked_run": dict(blocked),
        "blocked_genes": list(request.get("blocked_genes") or []),
        "explicitly_excluded": [
            str(x).strip() for x in (explicitly_excluded or [POU5F1B_ENSEMBL]) if str(x).strip()
        ],
    }
    if ui_session_id:
        record["ui_session_id"] = str(ui_session_id)

    if norm == "curated_bridge":
        if curated_overlay_path is None:
            raise OrthologApprovalError(
                "curated_bridge requires curated_overlay_path (directory or TSV)."
            )
        # Resolve pair-specific overlay file for hashing (human_to_mouse by default
        # from request direction when possible).
        direction = str(blocked.get("direction") or "")
        overlay_arg: str | Path = curated_overlay_path
        try:
            pair = ConversionPair(direction) if direction else None
        except ValueError:
            pair = None
        resolved = (
            resolve_curated_overlay_path(pair, curated_overlay=overlay_arg)
            if pair is not None
            else None
        )
        if resolved is None:
            cand = Path(str(curated_overlay_path)).expanduser()
            if cand.is_file():
                resolved = cand
            elif cand.is_dir() and direction:
                guess = cand / f"curated_bridge_{direction}.tsv"
                if guess.is_file():
                    resolved = guess
        if resolved is None or not resolved.is_file():
            raise OrthologApprovalError(
                f"could not resolve curated overlay TSV from {curated_overlay_path!r} "
                f"(direction={direction!r})."
            )
        digest = _sha256_file(resolved)
        if not digest:
            raise OrthologApprovalError(f"failed to hash overlay: {resolved}")
        # Policy: refuse overlays that include POU5F1B
        rows = _read_ortholog_rows(resolved)
        forbidden = normalize_gene_id(POU5F1B_ENSEMBL)
        for row in rows:
            src = normalize_gene_id(row.source) if is_ensembl_id(row.source) else row.source
            tgt = normalize_gene_id(row.target) if is_ensembl_id(row.target) else row.target
            if src == forbidden or tgt == forbidden:
                raise OrthologApprovalError(
                    f"Overlay includes forbidden paralog {POU5F1B_ENSEMBL}: {resolved}"
                )
        record["curated_overlay"] = {
            "path": str(Path(str(curated_overlay_path)).expanduser().resolve()),
            "resolved_tsv": str(resolved.resolve()),
            "sha256": digest,
        }
        bridge = dict(approved_bridge or {})
        if not bridge:
            # Default from first blocked gene + first candidate
            genes = list(request.get("blocked_genes") or [])
            primary = request.get("blocked_gene") if isinstance(request.get("blocked_gene"), Mapping) else None
            src_row = primary or (genes[0] if genes else {})
            if isinstance(src_row, Mapping):
                bridge = {
                    "source_id": src_row.get("source_id") or "",
                    "source_symbol": src_row.get("source_symbol") or "",
                    "target_id": src_row.get("candidate_target_id")
                    or (
                        (src_row.get("candidate_target_ids") or [""])[0]
                        if src_row.get("candidate_target_ids")
                        else ""
                    ),
                }
        if not bridge.get("source_id") or not bridge.get("target_id"):
            raise OrthologApprovalError(
                "curated_bridge requires approved_bridge.source_id and target_id "
                "(or blocked_gene candidates on the request)."
            )
        record["approved_bridge"] = {
            "source_id": str(bridge["source_id"]),
            "target_id": str(bridge["target_id"]),
            "source_symbol": str(bridge.get("source_symbol") or ""),
            "target_symbol": str(bridge.get("target_symbol") or ""),
        }
    return record


def write_approval_record(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """
    Write an immutable approval_record.yaml.

    By default refuses to overwrite an existing file; writes
    ``approval_record_<utc>.yaml`` beside it instead when ``overwrite`` is False.
    """
    out = Path(str(path)).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and not overwrite:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = out.with_name(f"{out.stem}_{stamp}{out.suffix}")
    out.write_text(
        yaml.safe_dump(dict(record), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return out


def load_ortholog_audit_config(
    source: str | Path | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Load ``ortholog_audit`` from a mapping, a YAML file containing that key,
    or a YAML file that *is* the audit block.
    """
    if source is None:
        return None
    if isinstance(source, Mapping):
        if "ortholog_audit" in source:
            raw = source.get("ortholog_audit")
            return dict(raw) if isinstance(raw, Mapping) else None
        if "critical_sets" in source or "behavior_on_critical_loss" in source:
            return dict(source)
        return None

    path = Path(str(source)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"ortholog_audit config not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Invalid ortholog_audit YAML: {path}")
    if "ortholog_audit" in data:
        block = data["ortholog_audit"]
        return dict(block) if isinstance(block, Mapping) else None
    if "critical_sets" in data or "behavior_on_critical_loss" in data:
        return dict(data)
    return None


def resolve_ortholog_audit_config(
    tokenizer_cfg: Mapping[str, Any] | None = None,
    *,
    cli_path: str | Path | None = None,
) -> dict[str, Any] | None:
    if cli_path:
        return load_ortholog_audit_config(cli_path)
    cfg = tokenizer_cfg or {}
    if cfg.get("ortholog_audit"):
        return load_ortholog_audit_config(cfg["ortholog_audit"])
    inline = cfg.get("ortholog_audit_inline")
    if isinstance(inline, Mapping):
        return dict(inline)
    return None


def resolve_enable_ortholog_loss_gate(
    species: Mapping[str, str] | None,
    tokenizer_cfg: Mapping[str, Any] | None = None,
    *,
    audit_cfg: Mapping[str, Any] | None = None,
    cli_flag: bool | None = None,
) -> bool:
    """
    Enable when ``ortholog_audit`` is present (including same-species → not_applicable).
    Explicit CLI / YAML ``ortholog_loss_gate`` overrides.
    """
    del species  # reserved for future; enable is audit-driven
    if cli_flag is not None:
        return bool(cli_flag)
    cfg = tokenizer_cfg or {}
    if cfg.get("ortholog_loss_gate") is not None:
        return bool(cfg.get("ortholog_loss_gate"))
    return bool(audit_cfg)


def _flatten_critical_sets(
    audit_cfg: Mapping[str, Any],
    *,
    non_critical_overrides: Sequence[str] | None = None,
) -> list[tuple[str, str]]:
    """Return list of (set_name, gene_symbol)."""
    skip = {str(x).strip() for x in (non_critical_overrides or []) if str(x).strip()}
    skip_upper = {x.upper() for x in skip}
    out: list[tuple[str, str]] = []
    sets = audit_cfg.get("critical_sets") or {}
    if not isinstance(sets, Mapping):
        return out
    for set_name, genes in sets.items():
        if not genes:
            continue
        for g in genes:
            gene = str(g).strip()
            if not gene:
                continue
            if gene in skip or gene.upper() in skip_upper:
                continue
            out.append((str(set_name), gene))
    return out


def _build_input_index(
    gene_ids: Sequence[str],
    symbol_table: Mapping[str, str] | None,
) -> set[str]:
    """Normalized set of raw input feature IDs (and reverse-mapped symbols)."""
    features: set[str] = set()
    for g in gene_ids:
        raw = str(g).strip()
        if not raw:
            continue
        features.add(raw)
        if is_ensembl_id(raw) or is_fly_id(raw):
            features.add(normalize_gene_id(raw))

    if symbol_table:
        for sym, ens in symbol_table.items():
            ens_n = normalize_gene_id(ens) if is_ensembl_id(ens) else ens
            if ens_n in features or ens in features:
                features.add(str(sym))
                features.add(str(sym).upper())
    return features


def _critical_matches_input(
    gene: str,
    features: set[str],
    symbol_table: Mapping[str, str] | None,
    aliases: Mapping[str, Sequence[str]] | None,
) -> tuple[bool, str]:
    """
    Return (present, source_feature_id).

    ``source_feature_id`` is the concrete ID/symbol found in the raw input
    (prefer Ensembl / alias hit over the critical-set display name alone).
    """
    alias_list: list[str] = []
    if aliases:
        if gene in aliases:
            alias_list.extend(str(a) for a in aliases[gene] or [])
        for canon, alts in aliases.items():
            if str(canon).upper() == gene.upper():
                alias_list.extend(str(a) for a in alts or [])

    for a in alias_list:
        a = str(a).strip()
        if not a:
            continue
        if a in features:
            return True, a
        if is_ensembl_id(a) and normalize_gene_id(a) in features:
            return True, normalize_gene_id(a)

    ens = None
    if symbol_table:
        ens = _symbol_lookup_ci(symbol_table, gene)
        if ens:
            ens_n = normalize_gene_id(ens) if is_ensembl_id(ens) else ens
            if ens in features or ens_n in features:
                return True, ens_n if ens_n in features else ens

    for c in (gene, gene.upper(), gene.lower(), gene.capitalize()):
        if c in features:
            return True, c
        if is_ensembl_id(c) and normalize_gene_id(c) in features:
            return True, normalize_gene_id(c)

    return False, ""


def _mapping_status_for_critical(
    gene: str,
    source_feature: str,
    inspections_by_id: Mapping[str, GeneMappingInspection],
    symbol_table: Mapping[str, str] | None,
) -> GeneMappingInspection | None:
    keys = [source_feature, gene]
    if is_ensembl_id(source_feature):
        keys.append(normalize_gene_id(source_feature))
    if symbol_table:
        ens = _symbol_lookup_ci(symbol_table, gene)
        if ens:
            keys.append(normalize_gene_id(ens))
            keys.append(ens)
    for k in keys:
        if k in inspections_by_id:
            return inspections_by_id[k]
        if is_ensembl_id(k):
            nk = normalize_gene_id(k)
            if nk in inspections_by_id:
                return inspections_by_id[nk]
    for insp in inspections_by_id.values():
        if insp.source_id in keys or insp.input_id in keys:
            return insp
        if gene.upper() == insp.input_id.upper():
            return insp
    return None


def _drop_label(status: str) -> str:
    if status.startswith("dropped_"):
        return status
    if status in _DROP_STATUSES:
        return f"dropped_{status}" if not status.startswith("dropped_") else status
    if status in ("ortholog_one2many", "ortholog_many2many", "no_ortholog"):
        return f"dropped_{status}"
    return status


def build_mapping_provenance(
    species: Mapping[str, str] | None,
    *,
    orthologs_dir: Path | None = None,
) -> dict[str, Any]:
    parsed = parse_species_config(species)
    pair = conversion_pair(parsed["model_organism"], parsed["model"])
    policy = parse_ortholog_policy(parsed)
    base = orthologs_dir or ORTHOLOGS_DIR
    overlay_raw = _curated_overlay_from_species(parsed)
    overlay_path = (
        resolve_curated_overlay_path(pair, curated_overlay=overlay_raw) if pair else None
    )
    main_path = (base / _PAIR_TO_FILE[pair]) if pair else None

    prov: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_organism": parsed["model_organism"],
        "model": parsed["model"],
        "conversion_pair": pair.value if pair else None,
        "ortholog_policy": policy.value,
        "ortholog_curated_overlay": overlay_raw,
        "ortholog_table_path": str(main_path) if main_path else None,
        "ortholog_table_sha256": _sha256_file(main_path) if main_path else None,
        "overlay_path": str(overlay_path) if overlay_path else None,
        "overlay_sha256": _sha256_file(overlay_path) if overlay_path else None,
    }

    search_dirs: list[Path] = []
    if overlay_raw:
        p = Path(str(overlay_raw)).expanduser()
        if p.is_dir():
            search_dirs.append(p)
        elif p.is_file():
            search_dirs.append(p.parent)
    for d in search_dirs:
        for name in ("provenance.json", "ensembl_freeze.json"):
            fp = d / name
            if fp.is_file():
                try:
                    prov[name.replace(".json", "")] = json.loads(
                        fp.read_text(encoding="utf-8")
                    )
                except Exception:
                    prov[name.replace(".json", "")] = {
                        "path": str(fp),
                        "error": "unreadable",
                    }
    return prov


def evaluate_ortholog_loss_gate(
    gene_ids: Sequence[str],
    species: Mapping[str, str] | None,
    audit_cfg: Mapping[str, Any],
    *,
    orthologs_dir: Path | None = None,
    approval: Mapping[str, Any] | None = None,
    symbol_aliases: Mapping[str, Sequence[str]] | None = None,
    audit_path: str | Path | None = None,
) -> GateResult:
    parsed = parse_species_config(species)
    pair = conversion_pair(parsed["model_organism"], parsed["model"])
    policy = parse_ortholog_policy(parsed)

    fingerprints = build_run_fingerprints(
        gene_ids,
        parsed,
        audit_path=audit_path,
        audit_cfg=audit_cfg,
        orthologs_dir=orthologs_dir,
    )

    # Same-species / no conversion → not_applicable (still recordable).
    if pair is None or not should_convert(parsed):
        return GateResult(
            verdict="not_applicable",
            conversion_pair=pair.value if pair else None,
            ortholog_policy=policy.value,
            mapped_pct=100.0,
            input_genes=len([g for g in gene_ids if str(g).strip()]),
            mapped=len([g for g in gene_ids if str(g).strip()]),
            unmapped=0,
            summary={"reason": "no cross-species conversion"},
            provenance=build_mapping_provenance(parsed, orthologs_dir=orthologs_dir),
            fingerprints=fingerprints,
            message="Ortholog loss gate: not_applicable (same-species / no conversion).",
            continue_tokenize=True,
        )

    # Fail closed on forbidden paralogs in overlay (e.g. POU5F1B).
    validate_overlay_excludes_paralog(parsed, orthologs_dir=orthologs_dir)

    approval = dict(approval or {})
    decision: str | None = None
    if approval:
        decision = validate_approval_record(
            approval,
            fingerprints,
            overlay_sha256=fingerprints.get("overlay_sha256"),
        )

    non_critical: list[str] = list(approval.get("non_critical_overrides") or [])
    if decision == "mark_non_critical":
        extra = (
            approval.get("genes")
            or approval.get("non_critical_genes")
            or approval.get("mark_non_critical_genes")
            or []
        )
        non_critical.extend(str(x) for x in extra)

    symbol_table = load_input_symbol_table(parsed)
    aliases = symbol_aliases or audit_cfg.get("symbol_aliases") or {}
    if not isinstance(aliases, Mapping):
        aliases = {}

    features = _build_input_index(gene_ids, symbol_table)
    inspections = inspect_gene_mappings(
        gene_ids,
        parsed,
        orthologs_dir=orthologs_dir,
        input_symbol_table=symbol_table or None,
    )
    inspections_by_id: dict[str, GeneMappingInspection] = {}
    for insp in inspections:
        inspections_by_id[insp.input_id] = insp
        inspections_by_id[insp.source_id] = insp
        if is_ensembl_id(insp.input_id):
            inspections_by_id[normalize_gene_id(insp.input_id)] = insp

    mapped = sum(1 for i in inspections if i.status in _MAPPED_STATUSES and i.selected_target)
    input_n = len(inspections)
    unmapped = input_n - mapped
    mapped_pct = round(100.0 * mapped / input_n, 2) if input_n else 0.0

    type_counts: dict[str, int] = {}
    for insp in inspections:
        type_counts[insp.status] = type_counts.get(insp.status, 0) + 1

    critical_defs = _flatten_critical_sets(audit_cfg, non_critical_overrides=non_critical)
    critical_rows: list[CriticalGeneAuditRow] = []
    for set_name, gene in critical_defs:
        present, source_feature = _critical_matches_input(
            gene, features, symbol_table, aliases
        )
        if not present:
            critical_rows.append(
                CriticalGeneAuditRow(
                    gene=gene,
                    critical_set=set_name,
                    input_presence="absent_from_input",
                    mapping_status="absent_from_input",
                    block_eligible=False,
                    reaches_token=False,
                )
            )
            continue

        insp = _mapping_status_for_critical(
            gene, source_feature, inspections_by_id, symbol_table
        )
        if insp is None:
            status = "symbol_unresolved"
            selected = ""
            candidates: list[str] = []
            types: list[str] = []
            reaches = False
        else:
            status = insp.status
            selected = insp.selected_target or ""
            candidates = list(insp.candidates)
            types = list(insp.orthology_types)
            reaches = bool(insp.selected_target) and insp.status in _MAPPED_STATUSES

        if reaches:
            mapping_status = (
                "mapped_curated_bridge" if status == "curated_bridge" else "mapped"
            )
            block_eligible = False
        else:
            mapping_status = _drop_label(status)
            block_eligible = True

        critical_rows.append(
            CriticalGeneAuditRow(
                gene=gene,
                critical_set=set_name,
                input_presence="present_in_input",
                mapping_status=mapping_status,
                source_feature=source_feature,
                selected_target=selected,
                candidates=candidates,
                orthology_types=types,
                block_eligible=block_eligible,
                reaches_token=reaches,
            )
        )

    pass_min = float(audit_cfg.get("pass_mapped_pct_min", DEFAULT_PASS_MAPPED_PCT_MIN))
    warn_min = float(audit_cfg.get("warn_mapped_pct_min", DEFAULT_WARN_MAPPED_PCT_MIN))
    block_min = float(audit_cfg.get("block_mapped_pct_min", DEFAULT_BLOCK_MAPPED_PCT_MIN))
    warn_on = {
        str(x).strip().lower()
        for x in (audit_cfg.get("warn_on") or [])
        if str(x).strip()
    }

    provenance = build_mapping_provenance(parsed, orthologs_dir=orthologs_dir)
    provenance_missing = False
    if pair is not None:
        if not provenance.get("ortholog_policy") or not provenance.get("ortholog_table_path"):
            provenance_missing = True
        if provenance.get("ortholog_table_sha256") is None:
            provenance_missing = True

    present_dropped = [r for r in critical_rows if r.block_eligible]
    absent = [r for r in critical_rows if r.input_presence == "absent_from_input"]

    continue_tokenize = True
    verdict = "pass"
    messages: list[str] = []

    if decision == "stop":
        verdict = "stopped"
        continue_tokenize = False
        messages.append("Approval decision=stop: tokenize skipped.")
    elif decision == "sensitivity_best_of_n":
        verdict = "stopped"
        continue_tokenize = False
        messages.append(
            "Approval decision=sensitivity_best_of_n: not auto-run in v1; "
            "re-tokenize manually with ortholog_policy=best_of_n."
        )
    elif decision == "curated_bridge":
        overlay_ok = bool(_curated_overlay_from_species(parsed))
        if not overlay_ok:
            messages.append(
                "Approval curated_bridge recorded, but species.ortholog_curated_overlay "
                "is not set. Activate via --ortholog-curated-overlay and re-run; "
                "the gate never auto-writes curated TSV rows."
            )
        elif present_dropped:
            messages.append(
                "Approval curated_bridge: overlay active but critical gene(s) still "
                "unmapped: " + ", ".join(sorted({r.gene for r in present_dropped}))
            )
        else:
            messages.append("Approval curated_bridge: overlay accepted.")
    elif decision == "mark_non_critical":
        messages.append(
            "Approval mark_non_critical: overrides applied: "
            + (", ".join(sorted(set(non_critical))) or "(none)")
        )

    if verdict != "stopped":
        if present_dropped or mapped_pct < block_min or provenance_missing:
            verdict = "block"
            continue_tokenize = False
            if present_dropped:
                messages.append(
                    "Critical gene(s) present in input but dropped by mapping policy: "
                    + ", ".join(sorted({r.gene for r in present_dropped}))
                )
            if mapped_pct < block_min:
                messages.append(
                    f"mapped_pct {mapped_pct}% below block_mapped_pct_min {block_min}%"
                )
            if provenance_missing:
                messages.append("mapping provenance incomplete (policy/table hash).")
        else:
            warn_reasons: list[str] = []
            if absent:
                warn_reasons.append(
                    "critical gene(s) absent from raw input: "
                    + ", ".join(sorted({r.gene for r in absent}))
                )
            if mapped_pct < warn_min:
                warn_reasons.append(
                    f"mapped_pct {mapped_pct}% below warn_mapped_pct_min {warn_min}%"
                )
            if warn_on:
                for insp in inspections:
                    st = insp.status.lower()
                    if st in warn_on or st.replace("dropped_", "") in warn_on:
                        if insp.status not in _MAPPED_STATUSES:
                            warn_reasons.append(
                                f"non-critical/policy drop status seen: {insp.status}"
                            )
                            break
            if mapped_pct < pass_min:
                warn_reasons.append(
                    f"mapped_pct {mapped_pct}% below pass_mapped_pct_min {pass_min}%"
                )
            if warn_reasons:
                verdict = "warn"
                messages.extend(warn_reasons)
            else:
                verdict = "pass"
                if not messages or decision not in ("curated_bridge", "mark_non_critical"):
                    messages.append("Ortholog loss gate: pass.")

    summary = {
        "status_counts": type_counts,
        "mapped_pct": mapped_pct,
        "critical_present_in_input": sum(
            1 for r in critical_rows if r.input_presence == "present_in_input"
        ),
        "critical_absent_from_input": len(absent),
        "critical_present_dropped": len(present_dropped),
        "critical_present_mapped": sum(
            1
            for r in critical_rows
            if r.input_presence == "present_in_input" and r.reaches_token
        ),
        "approval_decision": decision,
        "thresholds": {
            "pass_mapped_pct_min": pass_min,
            "warn_mapped_pct_min": warn_min,
            "block_mapped_pct_min": block_min,
        },
    }

    approval_request: dict[str, Any] = {}
    if verdict == "block":
        approval_request = build_approval_request(
            fingerprints,
            present_dropped,
            absent=absent,
            policy=policy.value,
        )

    return GateResult(
        verdict=verdict,
        conversion_pair=pair.value if pair else None,
        ortholog_policy=policy.value,
        mapped_pct=mapped_pct,
        input_genes=input_n,
        mapped=mapped,
        unmapped=unmapped,
        critical_rows=critical_rows,
        gene_inspections=inspections,
        summary=summary,
        provenance=provenance,
        approval_request=approval_request,
        fingerprints=fingerprints,
        message="\n".join(messages),
        continue_tokenize=continue_tokenize and verdict in ("pass", "warn", "not_applicable"),
    )


def format_approval_ticket(result: GateResult) -> str:
    req = result.approval_request or {}
    blocked_run = req.get("blocked_run") or result.fingerprints or {}
    lines = [
        "=" * 72,
        "ORTHOLOG_MAPPING_APPROVAL_REQUIRED",
        "=" * 72,
        f"request_id: {req.get('request_id')}",
        f"Direction: {blocked_run.get('direction')}",
        f"Policy: {req.get('policy') or result.ortholog_policy}",
        f"input_feature_sha256: {blocked_run.get('input_feature_sha256')}",
        f"manifest_sha256: {blocked_run.get('manifest_sha256')}",
        f"mapping_source_sha256: {blocked_run.get('mapping_source_sha256')}",
        "",
        "Critical gene lost (present in input, dropped by mapping policy):",
    ]
    lost = req.get("blocked_genes") or []
    if not lost and req.get("blocked_gene"):
        lost = [req["blocked_gene"]]
    if not lost:
        lines.append("  (none)")
    for row in lost:
        lines.append(f"  {row.get('source_symbol')} ({row.get('source_id')})")
        lines.append(f"    Conversion status: {row.get('mapping_status')}")
        cands = row.get("candidate_target_ids") or []
        if row.get("candidate_target_id") and not cands:
            cands = [row["candidate_target_id"]]
        if cands:
            lines.append(f"    Candidate target(s): {', '.join(cands)}")
    lines.append("")
    lines.append("Critical gene not in raw input (not Block):")
    absent = req.get("critical_gene_not_in_raw_input") or []
    if not absent:
        lines.append("  (none)")
    for row in absent:
        lines.append(f"  {row.get('gene')} [{row.get('critical_set')}]")
    lines.extend(
        [
            "",
            "Fail-closed approval:",
            "  1. Keep ortholog_approval_request.yaml as status: pending (do NOT pass it",
            "     to --ortholog-approval-record).",
            "  2. Create a NEW YAML with status: approved, the same request_id and",
            "     blocked_run hashes, plus decision / approved_by.",
            "  Decisions: stop | curated_bridge | sensitivity_best_of_n | mark_non_critical",
            "  (aliases A|B|C|D).",
            "=" * 72,
        ]
    )
    return "\n".join(lines)


def write_gate_artifacts(result: GateResult, output_dir: Path | str) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    detail_path = out / "conversion_gene_detail.tsv"
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "input_id",
                "source_id",
                "status",
                "selected_target",
                "candidates",
                "orthology_types",
                "via_curated_overlay",
                "via_platform_curated",
            ]
        )
        for insp in result.gene_inspections:
            w.writerow(
                [
                    insp.input_id,
                    insp.source_id,
                    insp.status,
                    insp.selected_target or "",
                    ",".join(insp.candidates),
                    ",".join(insp.orthology_types),
                    "1" if insp.via_curated_overlay else "0",
                    "1" if insp.via_platform_curated else "0",
                ]
            )
    paths["conversion_gene_detail"] = str(detail_path)

    crit_path = out / "critical_gene_audit.tsv"
    with open(crit_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "gene",
                "critical_set",
                "input_presence",
                "mapping_status",
                "source_feature",
                "selected_target",
                "candidates",
                "orthology_types",
                "block_eligible",
                "reaches_token",
            ]
        )
        for r in result.critical_rows:
            w.writerow(
                [
                    r.gene,
                    r.critical_set,
                    r.input_presence,
                    r.mapping_status,
                    r.source_feature,
                    r.selected_target,
                    ",".join(r.candidates),
                    ",".join(r.orthology_types),
                    "1" if r.block_eligible else "0",
                    "1" if r.reaches_token else "0",
                ]
            )
    paths["critical_gene_audit"] = str(crit_path)

    summary_path = out / "ortholog_loss_summary.json"
    summary_path.write_text(
        json.dumps(result.to_summary_dict(), indent=2) + "\n", encoding="utf-8"
    )
    paths["ortholog_loss_summary"] = str(summary_path)

    prov_path = out / "mapping_provenance.json"
    prov_path.write_text(json.dumps(result.provenance, indent=2) + "\n", encoding="utf-8")
    paths["mapping_provenance"] = str(prov_path)

    if result.verdict == "block" and result.approval_request:
        # Refresh critical_gene_audit_sha256 from the written TSV and remint request_id.
        audit_sha = _sha256_file(crit_path) or ""
        req = dict(result.approval_request)
        blocked = dict(req.get("blocked_run") or {})
        blocked["critical_gene_audit_sha256"] = audit_sha
        req["blocked_run"] = blocked
        req["request_id"] = mint_request_id(blocked)
        req["status"] = "pending"
        result.approval_request = req
        if result.fingerprints is not None:
            fps = dict(result.fingerprints)
            fps["critical_gene_audit_sha256"] = audit_sha
            result.fingerprints = fps

        approval_path = out / "ortholog_approval_request.yaml"
        approval_path.write_text(
            yaml.safe_dump(req, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        paths["ortholog_approval_request"] = str(approval_path)

        # Rewrite summary so fingerprints / request_id stay consistent.
        summary_path.write_text(
            json.dumps(result.to_summary_dict(), indent=2) + "\n", encoding="utf-8"
        )

    result.artifact_paths = paths
    return paths


def load_approval_record(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(str(path)).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"ortholog approval record not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Invalid approval record: {p}")
    return dict(data)


def collect_genes_for_gate(
    loom_directory: Path | str,
    *,
    file_format: str = "loom",
) -> list[str]:
    loom_dir = Path(loom_directory)
    genes: set[str] = set()
    for loom_path in sorted(loom_dir.glob(f"*.{file_format}")):
        genes.update(collect_loom_gene_ids(loom_path))
    return sorted(genes)


def run_ortholog_loss_gate(
    loom_directory: Path | str,
    output_dir: Path | str,
    species: Mapping[str, str] | None,
    audit_cfg: Mapping[str, Any],
    *,
    file_format: str = "loom",
    approval_record: str | Path | None = None,
    gene_ids: Sequence[str] | None = None,
    audit_path: str | Path | None = None,
) -> GateResult:
    try:
        genes = (
            list(gene_ids)
            if gene_ids is not None
            else collect_genes_for_gate(loom_directory, file_format=file_format)
        )
        approval = load_approval_record(approval_record)
        result = evaluate_ortholog_loss_gate(
            genes,
            species,
            audit_cfg,
            approval=approval,
            audit_path=audit_path,
        )
        paths = write_gate_artifacts(result, output_dir)
        print(f"  ortholog_loss_gate:   {result.verdict}")
        print(f"  ortholog loss summary: {paths.get('ortholog_loss_summary')}")
        if result.message:
            print(result.message)
        if result.verdict == "block":
            print(format_approval_ticket(result))
            raise OrthologLossGateError(
                result.message or "ortholog_loss_gate blocked", result=result
            )
        if result.verdict == "stopped":
            print("  ortholog_loss_gate: tokenize skipped (approval stop).")
        return result
    except OrthologApprovalError as exc:
        print(f"  ortholog_loss_gate: approval rejected: {exc}")
        raise SystemExit(2) from exc
