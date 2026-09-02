"""ortholog_loss_gate: fail-closed approval + Pass/Warn/Block."""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "core", ROOT / "contracts", ROOT / "webui"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
sys.path.insert(0, str(ROOT))


def _load_modules():
    geneformer_pkg = types.ModuleType("geneformer")
    backends_pkg = types.ModuleType("geneformer.backends")
    sys.modules["geneformer"] = geneformer_pkg
    sys.modules["geneformer.backends"] = backends_pkg

    def _load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    registry = _load(
        "geneformer.backends.registry",
        ROOT / "core" / "geneformer" / "backends" / "registry.py",
    )
    backends_pkg.registry = registry
    for name in (
        "BackendSpec",
        "HumanVariant",
        "ModelId",
        "MouseVariant",
        "Organism",
        "OrthologPolicy",
        "get_backend",
        "native_organism",
        "needs_gene_conversion",
        "parse_species_config",
        "resolve_pretrained_path",
    ):
        setattr(backends_pkg, name, getattr(registry, name))

    gc = _load(
        "geneformer.gene_converter",
        ROOT / "core" / "geneformer" / "gene_converter.py",
    )
    geneformer_pkg.gene_converter = gc

    conv = types.ModuleType("geneformer.conversion_report")

    def collect_loom_gene_ids(_path):
        return []

    conv.collect_loom_gene_ids = collect_loom_gene_ids
    sys.modules["geneformer.conversion_report"] = conv

    sc = _load(
        "geneformer.species_context",
        ROOT / "core" / "geneformer" / "species_context.py",
    )
    geneformer_pkg.species_context = sc

    gate = _load(
        "geneformer.ortholog_loss_gate",
        ROOT / "core" / "geneformer" / "ortholog_loss_gate.py",
    )
    geneformer_pkg.ortholog_loss_gate = gate
    gate.load_input_symbol_table = lambda species: {}
    return gc, gate


gc, gate = _load_modules()

POU5F1 = "ENSG00000204531"
POU5F1B = "ENSG00000212993"
SOX2 = "ENSG00000181449"
KLF4 = "ENSG00000136826"
MYC = "ENSG00000136997"
POU5F1_MOUSE = "ENSMUSG00000024406"
ORTHOLOGS = ROOT / "core" / "geneformer" / "dicts" / "orthologs"
PROJECT_OVERLAY = ROOT / "analysis" / "ortholog_policy" / "v1"

AUDIT = {
    "policy": "ensembl_one2one",
    "behavior_on_critical_loss": "require_user_approval",
    "critical_sets": {"oskm": ["POU5F1", "SOX2", "KLF4", "MYC"]},
    "symbol_aliases": {
        "POU5F1": [POU5F1],
        "SOX2": [SOX2],
        "KLF4": [KLF4],
        "MYC": [MYC],
    },
    "warn_on": ["ortholog_one2many", "ortholog_many2many", "no_ortholog"],
    "pass_mapped_pct_min": 0,
    "warn_mapped_pct_min": 0,
    "block_mapped_pct_min": 0,
}

SPECIES = {
    "model_organism": "human",
    "model": "mouse_geneformer",
    "ortholog_policy": "one2one",
}


def _build_approved_from_request(
    request: dict,
    *,
    decision: str = "curated_bridge",
    overlay_path: Path | None = None,
    overlay_sha256: str | None = None,
    blocked_run_overrides: dict | None = None,
) -> dict:
    blocked = dict(request.get("blocked_run") or {})
    if blocked_run_overrides:
        blocked.update(blocked_run_overrides)
    approved: dict = {
        "status": "approved",
        "request_id": gate.mint_request_id(blocked),
        "approved_by": "test",
        "approved_at": "2026-08-11T00:00:00+00:00",
        "decision": decision,
        "blocked_run": blocked,
    }
    if decision == "curated_bridge":
        path = overlay_path or PROJECT_OVERLAY
        sha = overlay_sha256
        if sha is None:
            # Hash the resolved bridge TSV (same as gate fingerprints).
            tsv = path / "curated_bridge_human_to_mouse.tsv"
            sha = gate._sha256_file(tsv)
        approved["curated_overlay"] = {"path": str(path), "sha256": sha}
        approved["approved_bridge"] = {
            "source_id": POU5F1,
            "target_id": POU5F1_MOUSE,
        }
        approved["explicitly_excluded"] = [POU5F1B]
    return approved


@unittest.skipUnless(
    (ORTHOLOGS / "human_to_mouse.tsv").is_file(),
    "production ortholog TSV missing",
)
class TestOrthologLossGate(unittest.TestCase):
    def setUp(self):
        gc._TABLE_CACHE.clear()
        self._prev_env = os.environ.pop("GENEFORMER_ORTHOLOG_CURATED_OVERLAY", None)

    def tearDown(self):
        gc._TABLE_CACHE.clear()
        if self._prev_env is None:
            os.environ.pop("GENEFORMER_ORTHOLOG_CURATED_OVERLAY", None)
        else:
            os.environ["GENEFORMER_ORTHOLOG_CURATED_OVERLAY"] = self._prev_env

    def test_01_critical_one2one_pass(self):
        """SOX2, KLF4, MYC only as critical (all in input) → Pass."""
        audit = {
            **AUDIT,
            "critical_sets": {"oskm": ["SOX2", "KLF4", "MYC"]},
        }
        genes = [SOX2, KLF4, MYC]
        result = gate.evaluate_ortholog_loss_gate(genes, SPECIES, audit)
        self.assertEqual(result.verdict, "pass")
        self.assertTrue(result.continue_tokenize)
        self.assertFalse(any(r.block_eligible for r in result.critical_rows))

    def test_02_critical_absent_from_input_warn(self):
        genes = [SOX2, KLF4, MYC]  # no POU5F1
        result = gate.evaluate_ortholog_loss_gate(genes, SPECIES, AUDIT)
        self.assertEqual(result.verdict, "warn")
        self.assertTrue(result.continue_tokenize)
        pou = [r for r in result.critical_rows if r.gene == "POU5F1"]
        self.assertTrue(pou)
        self.assertEqual(pou[0].input_presence, "absent_from_input")
        self.assertFalse(pou[0].block_eligible)

    def test_03_non_critical_pou5f1_one2many_warn(self):
        audit = {
            "critical_sets": {},
            "warn_on": ["ortholog_one2many"],
            "pass_mapped_pct_min": 0,
            "warn_mapped_pct_min": 0,
            "block_mapped_pct_min": 0,
        }
        result = gate.evaluate_ortholog_loss_gate([POU5F1, SOX2], SPECIES, audit)
        self.assertEqual(result.verdict, "warn")
        self.assertTrue(result.continue_tokenize)

    def test_04_pou5f1_critical_present_block_writes_pending_request(self):
        genes = [POU5F1, SOX2, KLF4, MYC]
        result = gate.evaluate_ortholog_loss_gate(genes, SPECIES, AUDIT)
        self.assertEqual(result.verdict, "block")
        pou = [r for r in result.critical_rows if r.gene == "POU5F1"]
        self.assertTrue(pou)
        self.assertEqual(pou[0].input_presence, "present_in_input")
        self.assertTrue(pou[0].block_eligible)
        self.assertIn("one2many", pou[0].mapping_status)
        self.assertEqual(result.approval_request.get("status"), "pending")
        self.assertTrue(result.approval_request.get("request_id", "").startswith("sha256:"))

        with tempfile.TemporaryDirectory() as tmp:
            paths = gate.write_gate_artifacts(result, tmp)
            req_path = Path(paths["ortholog_approval_request"])
            self.assertTrue(req_path.is_file())
            self.assertEqual(req_path.name, "ortholog_approval_request.yaml")
            self.assertFalse((Path(tmp) / "approval_record.yaml").exists())
            loaded = yaml.safe_load(req_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["status"], "pending")
            self.assertEqual(
                loaded["request_id"],
                gate.mint_request_id(loaded["blocked_run"]),
            )
            self.assertTrue(loaded["blocked_run"]["critical_gene_audit_sha256"])

    def test_05_pending_request_as_approval_rejected(self):
        genes = [POU5F1, SOX2, KLF4, MYC]
        blocked = gate.evaluate_ortholog_loss_gate(genes, SPECIES, AUDIT)
        pending = dict(blocked.approval_request)
        self.assertEqual(pending.get("status"), "pending")
        with self.assertRaises(gate.OrthologApprovalError) as ctx:
            gate.evaluate_ortholog_loss_gate(
                genes, SPECIES, AUDIT, approval=pending
            )
        self.assertIn("pending", str(ctx.exception).lower())

    @unittest.skipUnless(
        (ROOT / "analysis" / "ortholog_policy" / "v1").is_dir(),
        "project ortholog overlay missing",
    )
    def test_06_approved_curated_bridge_matching_hashes_pass(self):
        genes = [POU5F1, SOX2, KLF4, MYC]
        # First evaluate without overlay to get fingerprints / request.
        blocked = gate.evaluate_ortholog_loss_gate(genes, SPECIES, AUDIT)
        self.assertEqual(blocked.verdict, "block")
        with tempfile.TemporaryDirectory() as tmp:
            gate.write_gate_artifacts(blocked, tmp)
            request = dict(blocked.approval_request)

        approved = _build_approved_from_request(request)
        species = dict(SPECIES)
        species["ortholog_curated_overlay"] = str(PROJECT_OVERLAY)
        result = gate.evaluate_ortholog_loss_gate(
            genes, species, AUDIT, approval=approved
        )
        self.assertIn(result.verdict, ("pass", "warn"))
        self.assertTrue(result.continue_tokenize)
        pou = [r for r in result.critical_rows if r.gene == "POU5F1"][0]
        self.assertTrue(pou.reaches_token)
        self.assertFalse(pou.block_eligible)

    def test_07_approved_curated_bridge_wrong_overlay_hash_reject(self):
        genes = [POU5F1, SOX2, KLF4, MYC]
        blocked = gate.evaluate_ortholog_loss_gate(genes, SPECIES, AUDIT)
        with tempfile.TemporaryDirectory() as tmp:
            gate.write_gate_artifacts(blocked, tmp)
            request = dict(blocked.approval_request)
        approved = _build_approved_from_request(
            request, overlay_sha256="0" * 64
        )
        species = dict(SPECIES)
        species["ortholog_curated_overlay"] = str(PROJECT_OVERLAY)
        with self.assertRaises(gate.OrthologApprovalError) as ctx:
            gate.evaluate_ortholog_loss_gate(
                genes, species, AUDIT, approval=approved
            )
        self.assertIn("overlay", str(ctx.exception).lower())

    def test_08_approved_wrong_direction_reject(self):
        genes = [POU5F1, SOX2, KLF4, MYC]
        blocked = gate.evaluate_ortholog_loss_gate(genes, SPECIES, AUDIT)
        with tempfile.TemporaryDirectory() as tmp:
            gate.write_gate_artifacts(blocked, tmp)
            request = dict(blocked.approval_request)
        approved = _build_approved_from_request(
            request,
            blocked_run_overrides={"direction": "mouse_to_human"},
        )
        species = dict(SPECIES)
        species["ortholog_curated_overlay"] = str(PROJECT_OVERLAY)
        with self.assertRaises(gate.OrthologApprovalError) as ctx:
            gate.evaluate_ortholog_loss_gate(
                genes, species, AUDIT, approval=approved
            )
        self.assertIn("direction", str(ctx.exception).lower())

    def test_09_overlay_containing_pou5f1b_reject(self):
        genes = [POU5F1, SOX2]
        with tempfile.TemporaryDirectory() as tmp:
            overlay_dir = Path(tmp)
            tsv = overlay_dir / "curated_bridge_human_to_mouse.tsv"
            tsv.write_text(
                "source_id\ttarget_id\n"
                f"{POU5F1}\t{POU5F1_MOUSE}\n"
                f"{POU5F1B}\t{POU5F1_MOUSE}\n",
                encoding="utf-8",
            )
            species = dict(SPECIES)
            species["ortholog_curated_overlay"] = str(overlay_dir)
            with self.assertRaises(gate.OrthologApprovalError) as ctx:
                gate.evaluate_ortholog_loss_gate(genes, species, AUDIT)
            self.assertIn(POU5F1B, str(ctx.exception))

    def test_10_same_species_not_applicable(self):
        species = {
            "model_organism": "mouse",
            "model": "mouse_geneformer",
            "ortholog_policy": "one2one",
        }
        result = gate.evaluate_ortholog_loss_gate(
            ["ENSMUSG00000000001", "ENSMUSG00000000003"],
            species,
            AUDIT,
        )
        self.assertEqual(result.verdict, "not_applicable")
        self.assertTrue(result.continue_tokenize)

    def test_resolve_enable_with_audit_even_same_species(self):
        species = {
            "model_organism": "mouse",
            "model": "mouse_geneformer",
        }
        self.assertTrue(
            gate.resolve_enable_ortholog_loss_gate(
                species, audit_cfg=AUDIT
            )
        )
        self.assertFalse(
            gate.resolve_enable_ortholog_loss_gate(
                SPECIES, audit_cfg=None
            )
        )


if __name__ == "__main__":
    unittest.main()
