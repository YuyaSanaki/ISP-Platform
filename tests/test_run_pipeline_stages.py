"""run_pipeline stage plumbing: child log relay, ortholog stop status, E2E ISP UMAP choice."""
from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "core", ROOT / "contracts", ROOT / "webui"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


def _load_run_pipeline():
    for name in ("pipeline_lib", "data_input_layout", "run_pipeline_log", "run_provenance"):
        sys.modules.setdefault(name, types.ModuleType(name))
    pl = sys.modules["pipeline_lib"]
    for attr in (
        "ROOT",
        "build_finetune_config",
        "build_isp_config",
        "build_tokenize_config",
        "format_pipeline_banner",
        "load_yaml",
        "remove_tokenized_outputs",
        "resolve_finetune_model_dir",
        "resolve_pipeline_paths",
        "resolve_tokenized_dataset",
        "study_name_from_input_dir",
        "write_yaml",
    ):
        if not hasattr(pl, attr):
            setattr(pl, attr, ROOT / "core" if attr == "ROOT" else (lambda *a, **k: None))
    dil = sys.modules["data_input_layout"]
    if not hasattr(dil, "unique_states_from_samples"):
        dil.unique_states_from_samples = lambda study_root: []
    rpl = sys.modules["run_pipeline_log"]
    if not hasattr(rpl, "install_rotating_stdio_tee"):
        rpl.install_rotating_stdio_tee = lambda *a, **k: None
    rp = sys.modules["run_provenance"]
    for attr in ("update_service_provenance", "write_service_provenance"):
        if not hasattr(rp, attr):
            setattr(rp, attr, lambda *a, **k: None)

    spec = importlib.util.spec_from_file_location(
        "run_pipeline_stages_under_test", ROOT / "core" / "run_pipeline.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class _Capture(io.StringIO):
    """sys.stdout stand-in that only sees Python-level writes, like the log tee."""


class TestSubprocessRelay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rp = _load_run_pipeline()

    def _run(self, code: str) -> tuple[str, BaseException | None]:
        cap = _Capture()
        err = None
        with mock.patch.object(sys, "stdout", cap):
            try:
                self.rp._run_subprocess([sys.executable, "-c", code], dict(), "Child")
            except subprocess.CalledProcessError as exc:
                err = exc
        return cap.getvalue(), err

    def test_child_stdout_and_stderr_reach_python_stdout(self):
        out, err = self._run(
            "import sys; print('to-stdout'); print('to-stderr', file=sys.stderr); "
            "sys.stdout.write('日本語\\n')"
        )
        self.assertIsNone(err)
        self.assertIn("to-stdout", out)
        self.assertIn("to-stderr", out)
        self.assertIn("日本語", out)
        self.assertIn(">>> Child finished OK.", out)

    def test_failure_output_is_relayed_before_raising(self):
        out, err = self._run("import sys; print('traceback-ish', file=sys.stderr); sys.exit(3)")
        self.assertIsInstance(err, subprocess.CalledProcessError)
        self.assertEqual(err.returncode, 3)
        self.assertIn("traceback-ish", out)
        self.assertNotIn("finished OK", out)


class _PipelineHarness:
    """Drive run_pipeline.main() with stage subprocesses replaced by callbacks."""

    def __init__(self, rp, tmp: Path, *, isp_perturbation: dict, on_stage=None):
        self.rp = rp
        self.run_dir = tmp / "run"
        self.run_dir.mkdir()
        self.output_root = tmp / "output"
        self.config = tmp / "pipeline.yaml"
        self.config.write_text("data: {input_dir: /data/study}\n", encoding="utf-8")
        self.resolved = {
            "study_name": "study",
            "input_dir": str(tmp / "study"),
            "tokenized_dir": str(self.run_dir / "tokenized_dataset"),
            "dataset_path": str(self.run_dir / "tokenized_dataset" / "study.dataset"),
            "finetune_run_dir": str(self.run_dir / "finetune"),
            "finetune_model_dir": str(self.run_dir / "finetune" / "all_run1"),
            "pipeline_run_dir": str(self.run_dir),
        }
        self.isp_cfg = {"perturbation": dict(isp_perturbation)}
        self.on_stage = on_stage or (lambda label, cmd: None)
        self.commands: list[tuple[str, list[str]]] = []
        self.provenance = mock.Mock()
        self.events: list[tuple[str, object]] = []

    def _write_yaml(self, path, data):
        self.events.append(("write_yaml", (Path(path).name, data)))

    def _write_provenance(self, *a, **k):
        self.events.append(("write_service_provenance", k))

    def _existing(self, key: str) -> Path:
        path = Path(self.resolved[key])
        if not path.is_dir():
            raise FileNotFoundError(f"not found: {path}")
        return path

    def _fake_run(self, cmd, env, label):
        self.commands.append((label, list(cmd)))
        self.on_stage(label, cmd)

    def main(self, *argv: str) -> None:
        rp = self.rp
        patches = [
            mock.patch.object(sys, "argv", ["run_pipeline.py", "--config", str(self.config), *argv]),
            mock.patch.object(
                rp,
                "load_yaml",
                lambda p: {
                    "data": {"input_dir": "/data/study"},
                    "paths": {"output_root": str(self.output_root)},
                },
            ),
            mock.patch.object(rp, "resolve_pipeline_paths", lambda pipeline, run_dir: dict(self.resolved)),
            mock.patch.object(rp, "remove_tokenized_outputs", lambda *a, **k: []),
            mock.patch.object(rp, "resolve_tokenized_dataset", lambda *a, **k: self._existing("dataset_path")),
            mock.patch.object(
                rp, "resolve_finetune_model_dir", lambda *a, **k: self._existing("finetune_model_dir")
            ),
            mock.patch.object(rp, "build_tokenize_config", lambda *a, **k: {}),
            mock.patch.object(rp, "build_finetune_config", lambda *a, **k: {}),
            mock.patch.object(rp, "build_isp_config", lambda *a, **k: self.isp_cfg),
            mock.patch.object(rp, "write_yaml", self._write_yaml),
            mock.patch.object(rp, "format_pipeline_banner", lambda *a, **k: ""),
            mock.patch.object(rp, "install_rotating_stdio_tee", lambda *a, **k: None),
            mock.patch.object(rp, "write_service_provenance", self._write_provenance),
            mock.patch.object(rp, "update_service_provenance", self.provenance),
            mock.patch.object(rp, "_validate_perturbation_states", lambda *a, **k: None),
            mock.patch.object(rp, "_num_classes_from_label_dict", lambda *a, **k: 2),
            mock.patch.object(rp, "_run_subprocess", self._fake_run),
            mock.patch.object(sys, "stdout", io.StringIO()),
        ]
        for p in patches:
            p.start()
        try:
            rp.main()
        finally:
            for p in reversed(patches):
                p.stop()

    @property
    def final_status(self) -> str:
        return self.provenance.call_args.args[1]["run_status"]

    def labels(self) -> list[str]:
        return [label for label, _ in self.commands]


class TestOrthologApprovalStop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rp = _load_run_pipeline()

    @staticmethod
    def _write_tokenize_meta(tokenized_dir: Path, run_status: str, message: str | None = None):
        tokenized_dir.mkdir(parents=True, exist_ok=True)
        meta = {"run_status": run_status}
        if message is not None:
            meta["ortholog_loss_gate"] = {"verdict": "stopped", "message": message}
        (tokenized_dir / "tokenize_run_metadata.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")

    def test_stop_message_reads_tokenize_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok = Path(tmp) / "tokenized_dataset"
            self.assertIsNone(self.rp._ortholog_approval_stop_message(tok))
            self._write_tokenize_meta(tok, "completed")
            self.assertIsNone(self.rp._ortholog_approval_stop_message(tok))
            self._write_tokenize_meta(tok, "stopped_ortholog_approval", "Approval decision=stop: tokenize skipped.")
            self.assertEqual(
                self.rp._ortholog_approval_stop_message(tok),
                "Approval decision=stop: tokenize skipped.",
            )

    def test_stop_decision_ends_run_as_stopped_not_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            def on_stage(label, cmd):
                if label == "Tokenize":
                    self._write_tokenize_meta(
                        Path(h.resolved["tokenized_dir"]),
                        "stopped_ortholog_approval",
                        "Approval decision=stop: tokenize skipped.",
                    )

            h = _PipelineHarness(self.rp, Path(tmp), isp_perturbation={}, on_stage=on_stage)
            h.main()
            self.assertEqual(h.labels(), ["Tokenize"])
            self.assertEqual(h.final_status, "stopped_ortholog_approval")

    def test_missing_dataset_after_normal_tokenize_still_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = _PipelineHarness(self.rp, Path(tmp), isp_perturbation={})
            with self.assertRaises(FileNotFoundError):
                h.main()
            self.assertEqual(h.final_status, "failed")


class TestE2EIspUmapChoice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rp = _load_run_pipeline()

    def _harness(self, tmp: Path, perturbation: dict, on_stage=None) -> _PipelineHarness:
        h = _PipelineHarness(self.rp, tmp, isp_perturbation=perturbation, on_stage=on_stage)
        Path(h.resolved["dataset_path"]).mkdir(parents=True)
        Path(h.resolved["finetune_model_dir"]).mkdir(parents=True)
        return h

    def _umap_cmds(self, h: _PipelineHarness) -> list[list[str]]:
        return [cmd for _, cmd in h.commands if any(str(c).endswith("run_isp_umap.py") for c in cmd)]

    def test_gene_list_runs_group_umap_without_top1_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = self._harness(Path(tmp), {"type": "delete", "genes_to_perturb": ["GATA4", "NKX2-5"]})
            h.main("--run-dir", str(Path(tmp) / "run"), "--skip-tokenize", "--skip-finetune")
            self.assertEqual(h.labels(), ["ISP", "ISP UMAP (gene set)"])
            (cmd,) = self._umap_cmds(h)
            self.assertNotIn("--gene", cmd)
            self.assertEqual(cmd[cmd.index("--run-dir") + 1], str(h.run_dir))
            self.assertEqual(h.final_status, "completed")

    def test_gene_list_with_unsupported_type_skips_umap(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = self._harness(Path(tmp), {"type": "inhibit", "genes_to_perturb": ["GATA4"]})
            h.main("--run-dir", str(Path(tmp) / "run"), "--skip-tokenize", "--skip-finetune")
            self.assertEqual(h.labels(), ["ISP"])
            self.assertEqual(h.final_status, "completed")

    def test_genome_wide_keeps_top1_umap(self):
        with tempfile.TemporaryDirectory() as tmp:
            def on_stage(label, cmd):
                if label == "ISP":
                    stats = h.run_dir / "ispstats_results"
                    stats.mkdir()
                    (stats / "significant_genes.csv").write_text(
                        "Gene_name,Ensembl_ID,Shift_to_goal_end\n"
                        "GATA4,ENSG00000136574,0.02\n"
                        "NKX2-5,ENSG00000183072,0.05\n",
                        encoding="utf-8",
                    )

            h = self._harness(Path(tmp), {"type": "delete", "genes_to_perturb": []}, on_stage=on_stage)
            h.main("--run-dir", str(Path(tmp) / "run"), "--skip-tokenize", "--skip-finetune")
            self.assertEqual(h.labels(), ["ISP", "ISP UMAP (TOP1 significant)"])
            (cmd,) = self._umap_cmds(h)
            self.assertEqual(cmd[cmd.index("--gene") + 1], "ENSG00000183072")


class TestRunDirAndConfigRecord(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rp = _load_run_pipeline()

    def test_skip_tokenize_without_run_dir_stops_before_any_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = _PipelineHarness(self.rp, Path(tmp), isp_perturbation={})
            with mock.patch.object(sys, "stderr", io.StringIO()) as err:
                with self.assertRaises(SystemExit) as ctx:
                    h.main("--skip-tokenize")
            self.assertEqual(ctx.exception.code, 2)
            self.assertIn("--skip-tokenize needs --run-dir", err.getvalue())
            self.assertEqual(h.commands, [])
            self.assertEqual(h.events, [])

    def test_config_used_records_cli_overrides_after_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = _PipelineHarness(self.rp, Path(tmp), isp_perturbation={})
            Path(h.resolved["dataset_path"]).mkdir(parents=True)
            audit = Path(tmp) / "audit.yaml"
            h.main(
                "--run-dir", str(h.run_dir),
                "--skip-tokenize", "--skip-finetune", "--skip-isp",
                "--ortholog-audit", str(audit),
            )
            names = [
                ev[1][0] if ev[0] == "write_yaml" else ev[0] for ev in h.events
            ]
            last_used = max(i for i, n in enumerate(names) if n == "pipeline_config_used.yaml")
            self.assertGreater(last_used, names.index("write_service_provenance"))
            used = h.events[last_used][1][1]
            tok = used["stages"]["tokenize"]["tokenizer"]
            self.assertEqual(tok["ortholog_audit"], str(audit.resolve()))
            self.assertTrue(tok["ortholog_loss_gate"])
            (prov,) = [ev[1] for ev in h.events if ev[0] == "write_service_provenance"]
            self.assertEqual(
                prov["extra_meta"]["cli_overrides"], {"--ortholog-audit": str(audit)}
            )


if __name__ == "__main__":
    unittest.main()
