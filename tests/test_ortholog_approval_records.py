"""Fail-closed approval_record writer + WebUI artifact discovery."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "core", ROOT / "webui"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)


def _load_gate():
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
    conv.collect_loom_gene_ids = lambda _p: []
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
    return gate


gate = _load_gate()

# Load webui helper after gate is in sys.modules
ui_spec = importlib.util.spec_from_file_location(
    "ortholog_approval_ui",
    ROOT / "webui" / "ortholog_approval_ui.py",
)
ui = importlib.util.module_from_spec(ui_spec)
sys.modules["ortholog_approval_ui"] = ui
ui_spec.loader.exec_module(ui)

ORTHOLOGS = ROOT / "core" / "geneformer" / "dicts" / "orthologs"
BBRC_OVERLAY = ROOT / "analysis" / "bbrc_oskm" / "ortholog_policy" / "v1"
POU5F1 = "ENSG00000204531"
POU5F1_MOUSE = "ENSMUSG00000024406"


def _pending_request():
    fps = {
        "direction": "human_to_mouse",
        "input_feature_sha256": "feat",
        "manifest_sha256": "man",
        "converter_git_commit": "abc",
        "mapping_source_sha256": "map",
        "critical_gene_audit_sha256": "aud",
    }
    row = gate.CriticalGeneAuditRow(
        gene="POU5F1",
        critical_set="oskm",
        input_presence="present_in_input",
        mapping_status="dropped_ortholog_one2many",
        source_feature=POU5F1,
        candidates=[POU5F1_MOUSE],
        block_eligible=True,
    )
    return gate.build_approval_request(fps, [row], policy="one2one"), fps


class TestApprovalRecordWriter(unittest.TestCase):
    def test_pending_cannot_be_validated_as_approval(self):
        req, fps = _pending_request()
        with self.assertRaises(gate.OrthologApprovalError):
            gate.validate_approval_record(req, fps)

    def test_build_stop_record_copies_blocked_run(self):
        req, _fps = _pending_request()
        rec = gate.build_approved_record(
            req, decision="stop", approved_by="u", reason="keep one2one"
        )
        self.assertEqual(rec["status"], "approved")
        self.assertEqual(rec["decision"], "stop")
        self.assertEqual(rec["request_id"], req["request_id"])
        self.assertEqual(rec["blocked_run"], req["blocked_run"])

    def test_reject_status_not_accepted_by_gate(self):
        req, fps = _pending_request()
        rec = gate.build_approved_record(
            req, decision="reject", approved_by="u", reason="no"
        )
        self.assertEqual(rec["status"], "rejected")
        with self.assertRaises(gate.OrthologApprovalError):
            gate.validate_approval_record(rec, fps)

    def test_write_refuses_overwrite_by_default(self):
        req, _ = _pending_request()
        rec = gate.build_approved_record(
            req, decision="stop", approved_by="u", reason="x"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approval_record.yaml"
            p1 = gate.write_approval_record(path, rec, overwrite=False)
            p2 = gate.write_approval_record(path, rec, overwrite=False)
            self.assertTrue(p1.is_file())
            self.assertTrue(p2.is_file())
            self.assertNotEqual(p1, p2)

    @unittest.skipUnless(BBRC_OVERLAY.is_dir(), "BBRC overlay missing")
    def test_curated_bridge_pins_overlay_hash(self):
        req, fps = _pending_request()
        rec = gate.build_approved_record(
            req,
            decision="curated_bridge",
            approved_by="u",
            reason="OSKM landmark",
            curated_overlay_path=BBRC_OVERLAY,
        )
        self.assertEqual(rec["status"], "approved")
        self.assertIn("sha256", rec["curated_overlay"])
        self.assertEqual(rec["approved_bridge"]["source_id"], POU5F1)
        self.assertEqual(rec["approved_bridge"]["target_id"], POU5F1_MOUSE)
        # Gate accepts when fingerprints + overlay hash match
        fps2 = dict(fps)
        fps2["overlay_sha256"] = rec["curated_overlay"]["sha256"]
        decision = gate.validate_approval_record(
            rec, fps2, overlay_sha256=rec["curated_overlay"]["sha256"]
        )
        self.assertEqual(decision, "curated_bridge")


class TestWebuiArtifactFind(unittest.TestCase):
    def test_find_pending_request(self):
        req, _ = _pending_request()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tokenized_dataset").mkdir()
            rp = root / "tokenized_dataset" / "ortholog_approval_request.yaml"
            rp.write_text(yaml.safe_dump(req), encoding="utf-8")
            (root / "tokenized_dataset" / "ortholog_loss_summary.json").write_text(
                '{"verdict":"block"}\n', encoding="utf-8"
            )
            arts = ui.find_ortholog_block_artifacts([root])
            self.assertIsNotNone(arts)
            self.assertEqual(arts.request_path, rp)
            written, rec = ui.create_approval_record_from_ui(
                arts, action="stop", approved_by="ui", reason="stop"
            )
            self.assertTrue(written.is_file())
            self.assertEqual(rec["decision"], "stop")


if __name__ == "__main__":
    unittest.main()
