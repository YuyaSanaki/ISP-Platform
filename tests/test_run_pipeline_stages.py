"""run_pipeline stage plumbing: child log relay, ortholog stop status, E2E ISP UMAP choice."""
from __future__ import annotations

import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import threading
import time
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

    def __init__(self, rp, tmp: Path, *, isp_perturbation: dict, on_stage=None, pipeline=None):
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
        self.pipeline = pipeline or {}
        self.on_stage = on_stage or (lambda label, cmd: None)
        self.envs: list[dict] = []
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

    def _load_yaml(self, _path):
        body = {
            "data": {"input_dir": "/data/study"},
            "paths": {"output_root": str(self.output_root)},
        }
        for key, value in self.pipeline.items():
            if key in body and isinstance(body[key], dict) and isinstance(value, dict):
                merged = dict(body[key])
                merged.update(value)
                body[key] = merged
            else:
                body[key] = value
        return body

    def _fake_run(self, cmd, env, label):
        self.commands.append((label, list(cmd)))
        self.envs.append(dict(env))
        self.on_stage(label, cmd)

    def main(self, *argv: str) -> None:
        rp = self.rp
        patches = [
            mock.patch.object(sys, "argv", ["run_pipeline.py", "--config", str(self.config), *argv]),
            mock.patch.object(rp, "load_yaml", self._load_yaml),
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

    def _harness(self, tmp: Path, perturbation: dict, on_stage=None, pipeline=None) -> _PipelineHarness:
        h = _PipelineHarness(
            self.rp, tmp, isp_perturbation=perturbation, on_stage=on_stage, pipeline=pipeline
        )
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


class TestReportRunDir(unittest.TestCase):
    """The Web UI reads the run folder from this file instead of guessing by mtime."""

    @classmethod
    def setUpClass(cls):
        cls.rp = _load_run_pipeline()

    def test_writes_run_dir_when_env_names_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "pipeline_run_dir.txt"
            run_dir = Path(tmp) / "pipeline_000000_000000Z"
            run_dir.mkdir()
            with mock.patch.dict("os.environ", {self.rp.RUN_DIR_FILE_ENV: str(marker)}):
                self.rp._report_run_dir(run_dir)
            self.assertEqual(marker.read_text(encoding="utf-8").strip(), str(run_dir.resolve()))

    def test_no_env_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict("os.environ", {}, clear=True):
                self.rp._report_run_dir(Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [])


class TestParallelIspStage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rp = _load_run_pipeline()

    def test_cuda_ids_come_from_the_environment(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=False):
            self.assertEqual(self.rp.cuda_device_ids(), ["0", "1"])
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "none"}, clear=False):
            self.assertEqual(self.rp.cuda_device_ids(), [])

    def test_plan_stays_one_accelerate_launch_until_parallel_is_on(self):
        cfg = Path("/tmp/isp.yaml")
        run = Path("/tmp/run")
        with mock.patch.dict(os.environ, {"ISP_NUM_GPUS": "1"}, clear=False):
            plan = self.rp.isp_stage_plan({}, {"perturbation": {}}, cfg, run)
        self.assertEqual(len(plan), 1)
        label, cmd, overlay = plan[0]
        self.assertEqual(label, "ISP")
        self.assertEqual(cmd[0], "accelerate")
        self.assertEqual(overlay, {})
        self.assertIn("--no-output-time-subdir", cmd)
        off = self.rp.isp_stage_plan(
            {"stages": {"isp": {"parallel": {"enabled": False, "chunk_count": 8}}}},
            {"perturbation": {"genes_to_perturb": ["GATA4"]}},
            cfg,
            run,
        )
        self.assertEqual(off[0][0], "ISP")
        self.assertEqual(off[0][1][0], "accelerate")

    def test_plan_splits_genome_wide_and_refuses_a_gene_list(self):
        pipeline = {"stages": {"isp": {"parallel": {"enabled": True, "chunk_count": 3}}}}
        cfg = Path("/tmp/isp.yaml")
        run = Path("/tmp/run")
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=False):
            plan = self.rp.isp_stage_plan(pipeline, {"perturbation": {"genes_to_perturb": []}}, cfg, run)
        self.assertEqual(
            [item[0] for item in plan],
            ["ISP chunk 1/3", "ISP chunk 2/3", "ISP chunk 3/3", "ISP merge"],
        )
        for index, (_label, cmd, overlay) in enumerate(plan[:3]):
            self.assertNotIn("accelerate", cmd)
            self.assertEqual(cmd[cmd.index("--chunk-id") + 1], str(index))
            self.assertEqual(cmd[cmd.index("--chunk-count") + 1], "3")
            self.assertIn("--no-output-time-subdir", cmd)
            self.assertIn("--no-output-date-subdir", cmd)
            self.assertEqual(overlay["CUDA_VISIBLE_DEVICES"], "0" if index % 2 == 0 else "1")
        merge = plan[-1][1]
        self.assertTrue(str(merge[1]).endswith("run_isp_merge.py"))
        self.assertEqual(merge[merge.index("--run-dir") + 1], str(run))
        self.assertEqual(merge[merge.index("--chunk-count") + 1], "3")
        with self.assertRaises(ValueError):
            self.rp.isp_stage_plan(
                pipeline,
                {"perturbation": {"genes_to_perturb": ["GATA4"]}},
                cfg,
                run,
            )
        with self.assertRaises(ValueError):
            self.rp.parallel_chunk_count(
                {"stages": {"isp": {"parallel": {"enabled": True, "chunk_count": 1}}}}
            )

    def test_pipeline_launches_chunks_then_merge_in_the_run_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            def on_stage(label, cmd):
                if label == "ISP merge":
                    stats = h.run_dir / "ispstats_results"
                    stats.mkdir()
                    (stats / "significant_genes.csv").write_text(
                        "Gene_name,Ensembl_ID,Shift_to_goal_end,N_Detections\n"
                        "NKX2-5,ENSG00000183072,0.05,40\n",
                        encoding="utf-8",
                    )

            h = _PipelineHarness(
                self.rp,
                Path(tmp),
                isp_perturbation={"type": "delete", "genes_to_perturb": []},
                on_stage=on_stage,
                pipeline={"stages": {"isp": {"parallel": {"enabled": True, "chunk_count": 2}}}},
            )
            Path(h.resolved["dataset_path"]).mkdir(parents=True)
            Path(h.resolved["finetune_model_dir"]).mkdir(parents=True)
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False):
                h.main("--run-dir", str(h.run_dir), "--skip-tokenize", "--skip-finetune")
            self.assertEqual(
                h.labels(),
                ["ISP chunk 1/2", "ISP chunk 2/2", "ISP merge", "ISP UMAP (TOP1 significant)"],
            )
            chunk_cmds = [cmd for label, cmd in h.commands if label.startswith("ISP chunk")]
            self.assertTrue(all("accelerate" not in cmd for cmd in chunk_cmds))
            self.assertEqual(
                [env.get("CUDA_VISIBLE_DEVICES") for env in h.envs[:2]],
                ["0", "0"],
            )
            merge = next(cmd for label, cmd in h.commands if label == "ISP merge")
            self.assertEqual(merge[merge.index("--run-dir") + 1], str(h.run_dir))
            self.assertEqual(h.final_status, "completed")

    def test_gene_list_parallel_stops_before_isp(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = _PipelineHarness(
                self.rp,
                Path(tmp),
                isp_perturbation={"genes_to_perturb": ["GATA4"]},
                pipeline={"stages": {"isp": {"parallel": {"enabled": True, "chunk_count": 2}}}},
            )
            Path(h.resolved["dataset_path"]).mkdir(parents=True)
            Path(h.resolved["finetune_model_dir"]).mkdir(parents=True)
            with self.assertRaises(ValueError):
                h.main("--run-dir", str(h.run_dir), "--skip-tokenize", "--skip-finetune")
            self.assertEqual(h.commands, [])
            self.assertEqual(h.final_status, "failed")

    def test_two_gpus_run_together_and_a_failure_skips_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            started = [root / "a", root / "b"]
            release = root / "release"
            wait_code = (
                "import os, pathlib, time\n"
                "pathlib.Path(os.environ['STARTED']).write_text('up')\n"
                "release = pathlib.Path(os.environ['RELEASE'])\n"
                "while not release.exists():\n"
                "    time.sleep(0.02)\n"
                "print('done')\n"
            )
            plan = [
                (
                    "ISP chunk 1/2",
                    [sys.executable, "-c", wait_code],
                    {"CUDA_VISIBLE_DEVICES": "0", "STARTED": str(started[0]), "RELEASE": str(release)},
                ),
                (
                    "ISP chunk 2/2",
                    [sys.executable, "-c", wait_code],
                    {"CUDA_VISIBLE_DEVICES": "1", "STARTED": str(started[1]), "RELEASE": str(release)},
                ),
                ("ISP merge", [sys.executable, "-c", "print('merged')"], {}),
            ]
            flag = {"ok": False}

            def releaser():
                deadline = time.time() + 3
                while time.time() < deadline:
                    if all(path.is_file() for path in started):
                        flag["ok"] = True
                        release.write_text("go")
                        return
                    time.sleep(0.02)
                release.write_text("go")

            thread = threading.Thread(target=releaser, daemon=True)
            thread.start()
            cap = io.StringIO()
            with mock.patch.object(sys, "stdout", cap):
                self.rp._run_isp_plan(plan, os.environ.copy())
            thread.join(timeout=5)
            self.assertTrue(flag["ok"])
            self.assertIn("merged", cap.getvalue())

        with tempfile.TemporaryDirectory() as tmp:
            plan = [
                (
                    "ISP chunk 1/2",
                    [sys.executable, "-c", "import sys; print('boom'); sys.exit(2)"],
                    {"CUDA_VISIBLE_DEVICES": "4"},
                ),
                (
                    "ISP chunk 2/2",
                    [sys.executable, "-c", "import os, time; time.sleep(30); print(os.environ['MARKER'])"],
                    {"CUDA_VISIBLE_DEVICES": "5", "MARKER": "RAN_CHUNK_2"},
                ),
                ("ISP merge", [sys.executable, "-c", "print('merged')"], {}),
            ]
            cap = io.StringIO()
            started_at = time.time()
            with mock.patch.object(sys, "stdout", cap):
                with self.assertRaises(subprocess.CalledProcessError) as ctx:
                    self.rp._run_isp_plan(plan, os.environ.copy())
            self.assertEqual(ctx.exception.returncode, 2)
            self.assertLess(time.time() - started_at, 10)
            text = cap.getvalue()
            self.assertIn("boom", text)
            self.assertNotIn("merged", text)
            self.assertNotIn("RAN_CHUNK_2", text)


if __name__ == "__main__":
    unittest.main()
