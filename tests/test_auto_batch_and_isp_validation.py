"""Unit tests for GPU auto batch sizing helpers and pipeline ISP pre-flight validation."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "core", ROOT / "contracts", ROOT / "webui"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_auto_batch_size():
    # Avoid importing geneformer/__init__.py (heavy deps). Load as a free-standing module.
    installed_torch_stub = False
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        cuda = types.SimpleNamespace(
            is_available=lambda: False,
            OutOfMemoryError=RuntimeError,
            empty_cache=lambda: None,
            reset_peak_memory_stats=lambda: None,
            memory_allocated=lambda: 0,
            mem_get_info=lambda: (0, 0),
            synchronize=lambda: None,
            max_memory_allocated=lambda: 0,
        )
        torch_stub.cuda = cuda
        sys.modules["torch"] = torch_stub
        installed_torch_stub = True
    try:
        # Do not register under geneformer.* here — callers that need the package path do it.
        return _load_module(
            "auto_batch_size_standalone", ROOT / "core" / "geneformer" / "auto_batch_size.py"
        )
    finally:
        # Drop the stub so later tests can import real torch (module keeps its own ref).
        if installed_torch_stub:
            sys.modules.pop("torch", None)


def _load_run_pipeline_validators():
    """Import only the validation helpers from run_pipeline without executing main."""
    # Minimal stubs so run_pipeline import side-effects stay light.
    for name in (
        "yaml",
        "pipeline_lib",
        "data_input_layout",
        "run_pipeline_log",
        "run_provenance",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))

    dil = sys.modules["data_input_layout"]
    if not hasattr(dil, "unique_states_from_samples"):
        dil.unique_states_from_samples = lambda study_root: []

    pl = sys.modules["pipeline_lib"]
    for attr in (
        "ROOT",
        "build_finetune_config",
        "build_isp_config",
        "build_tokenize_config",
        "format_pipeline_banner",
        "load_yaml",
        "resolve_pipeline_paths",
        "study_name_from_input_dir",
        "write_yaml",
    ):
        if not hasattr(pl, attr):
            setattr(pl, attr, ROOT if attr == "ROOT" else (lambda *a, **k: None))

    rpl = sys.modules["run_pipeline_log"]
    if not hasattr(rpl, "install_rotating_stdio_tee"):
        rpl.install_rotating_stdio_tee = lambda *a, **k: None
    rp = sys.modules["run_provenance"]
    for attr in ("update_service_provenance", "write_service_provenance"):
        if not hasattr(rp, attr):
            setattr(rp, attr, lambda *a, **k: None)

    return _load_module("run_pipeline_under_test", ROOT / "core" / "run_pipeline.py")


class TestAutoBatchSize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.abs = _load_auto_batch_size()

    def test_is_auto(self):
        self.assertTrue(self.abs.is_auto("auto"))
        self.assertTrue(self.abs.is_auto(" AUTO "))
        self.assertFalse(self.abs.is_auto(100))
        self.assertFalse(self.abs.is_auto("100"))

    def test_coerce_batch_size(self):
        self.assertEqual(self.abs.coerce_batch_size("auto", default=50), "auto")
        self.assertEqual(self.abs.coerce_batch_size(32, default=50), 32)
        self.assertEqual(self.abs.coerce_batch_size("64", default=50), 64)
        self.assertEqual(self.abs.coerce_batch_size("nope", default=50), 50)

    def test_resolve_batch_size_passthrough(self):
        self.assertEqual(
            self.abs.resolve_batch_size(
                42,
                default=10,
                probe=lambda _: None,
                cache_fields={"task": "unit"},
            ),
            42,
        )

    def test_resolve_auto_without_cuda_uses_default(self):
        # Force the no-CUDA branch instead of relying on torch being un-imported.
        with mock.patch.object(self.abs.torch.cuda, "is_available", return_value=False):
            chosen = self.abs.resolve_batch_size(
                "auto",
                default=77,
                probe=lambda _: None,
                cache_fields={"task": "unit_no_cuda"},
            )
        self.assertEqual(chosen, 77)

    def test_calibrate_stops_on_oom(self):
        def probe(bs: int) -> None:
            if bs >= 64:
                raise RuntimeError("CUDA out of memory")

        snap = self.abs.GpuMemorySnapshot(
            free=8 * 2**30,
            total=16 * 2**30,
            allocated=0,
            util_pct=0.0,
            free_accounting_reliable=True,
        )
        with mock.patch.object(
            self.abs,
            "_measure",
            side_effect=lambda probe_fn, candidate, repeats: (
                None if candidate >= 64 else (float(candidate), 1000)
            ),
        ), mock.patch.object(
            self.abs, "read_gpu_memory_snapshot", return_value=snap
        ):
            chosen = self.abs.calibrate_batch_size(
                probe,
                candidates=(16, 32, 64, 128),
                min_gain=0.05,
                label="unit",
            )
        self.assertEqual(chosen, 32)

    def test_calibrate_require_idle_raises_when_gpu_busy(self):
        snap = self.abs.GpuMemorySnapshot(
            free=100 * 2**30,
            total=122 * 2**30,
            allocated=0,
            util_pct=80.0,
            free_accounting_reliable=False,
        )
        with mock.patch.object(
            self.abs, "read_gpu_memory_snapshot", return_value=snap
        ), mock.patch.object(
            self.abs, "gpu_is_busy_for_calibration", return_value=(True, "util high")
        ):
            with self.assertRaises(self.abs.BusyGPUError):
                self.abs.calibrate_batch_size(
                    lambda _: None,
                    candidates=(6, 8, 16),
                    require_idle=True,
                    label="unit",
                )

    def test_require_idle_allows_gb10_underreported_free(self):
        """GB10: torch free ~58/122 at 0% util must not abort; budget uses total."""
        calls = {"n": 0}

        def measure(probe_fn, candidate, repeats):
            calls["n"] += 1
            # Peak scales with batch; with total-based budget this should not stop early.
            return (float(candidate) * 10, int(candidate * 0.05 * 2**30))

        snap = self.abs.GpuMemorySnapshot(
            free=int(58 * 2**30),
            total=int(122 * 2**30),
            allocated=0,
            util_pct=0.0,
            free_accounting_reliable=False,
        )
        with mock.patch.object(self.abs, "_measure", side_effect=measure), mock.patch.object(
            self.abs, "read_gpu_memory_snapshot", return_value=snap
        ):
            chosen = self.abs.calibrate_batch_size(
                lambda _: None,
                candidates=(6, 8, 16, 32),
                require_idle=True,
                min_gain=0.05,
                memory_fraction=0.7,
                label="unit",
            )
        self.assertEqual(chosen, 32)
        self.assertGreaterEqual(calls["n"], 4)
        # Budget must be from total (usable), not under-reported free.
        budget = self.abs.calibration_memory_budget(snap, memory_fraction=0.7)
        self.assertGreater(budget, int(0.7 * 58 * 2**30))
        self.assertAlmostEqual(budget / 2**30, 0.7 * 122, places=0)

    def test_gpu_is_busy_uses_util_not_soft_memory(self):
        idle = self.abs.GpuMemorySnapshot(
            free=int(58 * 2**30),
            total=int(122 * 2**30),
            allocated=0,
            util_pct=0.0,
            free_accounting_reliable=False,
        )
        busy_snap = self.abs.GpuMemorySnapshot(
            free=int(100 * 2**30),
            total=int(122 * 2**30),
            allocated=0,
            util_pct=55.0,
            free_accounting_reliable=False,
        )
        with mock.patch.object(self.abs.torch.cuda, "is_available", return_value=True):
            busy, _ = self.abs.gpu_is_busy_for_calibration(idle)
            self.assertFalse(busy)
            busy, reason = self.abs.gpu_is_busy_for_calibration(busy_snap)
            self.assertTrue(busy)
            self.assertIn("utilization", reason.lower())

    def test_unreliable_free_never_soft_contended(self):
        snap = self.abs.GpuMemorySnapshot(
            free=int(10 * 2**30),
            total=int(122 * 2**30),
            allocated=0,
            util_pct=0.0,
            free_accounting_reliable=False,
        )
        self.assertFalse(self.abs.gpu_memory_contended(snap))

    def test_train_candidates_start_at_six(self):
        self.assertEqual(self.abs.DEFAULT_TRAIN_CANDIDATES[0], 6)
        self.assertNotIn(2, self.abs.DEFAULT_TRAIN_CANDIDATES)

    def test_long_seq_train_candidates_and_fraction(self):
        self.assertEqual(self.abs.train_candidates_for_seq_len(2048)[0], 6)
        self.assertEqual(self.abs.train_candidates_for_seq_len(4096)[0], 2)
        self.assertEqual(
            self.abs.train_memory_fraction_for_seq_len(2048),
            self.abs.DEFAULT_TRAIN_MEMORY_FRACTION,
        )
        self.assertEqual(
            self.abs.train_memory_fraction_for_seq_len(4096),
            self.abs.LONG_SEQ_TRAIN_MEMORY_FRACTION,
        )
        self.assertLess(
            self.abs.train_memory_fraction_for_seq_len(4096),
            self.abs.train_memory_fraction_for_seq_len(2048),
        )
    def test_pipeline_lib_preserves_auto(self):
        # Isolate package stubs so we do not pollute other tests' imports.
        saved = {
            k: sys.modules.get(k)
            for k in (
                "geneformer",
                "geneformer.auto_batch_size",
                "geneformer.backends",
                "geneformer.species_context",
                "data_input_layout",
                "pipeline_lib_under_test",
                "yaml",
            )
        }
        try:
            geneformer_pkg = types.ModuleType("geneformer")
            geneformer_pkg.__path__ = [str(ROOT / "core" / "geneformer")]
            sys.modules["geneformer"] = geneformer_pkg
            abs_mod = _load_auto_batch_size()
            geneformer_pkg.auto_batch_size = abs_mod
            sys.modules["geneformer.auto_batch_size"] = abs_mod

            backends = types.ModuleType("geneformer.backends")
            backends.get_backend = lambda *a, **k: types.SimpleNamespace(max_input_size=2048)
            backends.parse_species_config = lambda *a, **k: {
                "model_organism": "mouse",
                "model": "mouse_geneformer",
            }
            backends.resolve_pretrained_path = lambda *a, **k: "/app/models/mouse-Geneformer/"
            sys.modules["geneformer.backends"] = backends
            geneformer_pkg.backends = backends

            sc = types.ModuleType("geneformer.species_context")
            sc.default_isp_forward_batch_size = lambda mis: 25 if int(mis) > 2048 else 100
            sys.modules["geneformer.species_context"] = sc
            geneformer_pkg.species_context = sc

            dil = types.ModuleType("data_input_layout")
            dil.resolve_single_cell_input_dir = lambda p: str(p)
            sys.modules["data_input_layout"] = dil

            # Prefer the real PyYAML if already importable; only stub as a last resort.
            try:
                import yaml as _real_yaml  # noqa: F401
            except ImportError:
                yaml_mod = types.ModuleType("yaml")
                yaml_mod.safe_load = lambda *a, **k: {}
                yaml_mod.dump = lambda *a, **k: None
                sys.modules["yaml"] = yaml_mod

            pl = _load_module("pipeline_lib_under_test", ROOT / "core" / "pipeline_lib.py")
            self.assertEqual(pl._batch_size_value("auto"), "auto")
            self.assertEqual(pl._batch_size_value(50), 50)

            template = {"runtime": {}, "paths": {}, "perturbation": {}, "model": {}, "species": {}}
            pipeline = {
                "runtime": {"forward_batch_size": "auto", "nproc": 2},
                "perturbation": {},
                "species": {"model_organism": "mouse", "model": "mouse_geneformer"},
            }
            resolved = {
                "dataset_path": "/tmp/ds",
                "pipeline_run_dir": "/tmp/run",
                "species": pipeline["species"],
                "max_input_size": 2048,
            }
            cfg = pl.build_isp_config(
                pipeline,
                resolved,
                template,
                finetune_model_dir="/tmp/model",
                num_classes=2,
            )
            self.assertEqual(cfg["runtime"]["forward_batch_size"], "auto")
        finally:
            for key, mod in saved.items():
                if mod is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = mod
            # Drop any other geneformer.* stubs created during this test.
            for key in list(sys.modules):
                if key.startswith("geneformer.") and key not in saved:
                    # Keep real packages for later tests if present on disk imports.
                    if saved.get("geneformer") is None:
                        sys.modules.pop(key, None)
            if saved.get("geneformer") is None:
                sys.modules.pop("geneformer", None)


class TestValidatePerturbationStates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rp = _load_run_pipeline_validators()

    def test_rejects_identical_start_end(self):
        with self.assertRaises(ValueError) as ctx:
            self.rp._validate_perturbation_states(
                {
                    "perturbation": {
                        "state_key": "disease",
                        "start_state": "AD",
                        "end_state": "AD",
                    }
                },
                "/tmp/study",
            )
        self.assertIn("both `AD`", str(ctx.exception))

    def test_rejects_missing_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp)
            with mock.patch.object(
                self.rp,
                "unique_states_from_samples",
                return_value=["AD", "WT"],
            ):
                # Patch the imported name used inside the function.
                import data_input_layout as dil

                with mock.patch.object(
                    dil, "unique_states_from_samples", return_value=["AD", "WT"]
                ):
                    with self.assertRaises(ValueError) as ctx:
                        self.rp._validate_perturbation_states(
                            {
                                "perturbation": {
                                    "state_key": "disease",
                                    "start_state": "AD",
                                    "end_state": "Ctrl",
                                }
                            },
                            str(study),
                        )
        self.assertIn("Ctrl", str(ctx.exception))

    def test_skips_non_path_derived_state_key(self):
        # Should not raise even if start==end when state_key is custom.
        self.rp._validate_perturbation_states(
            {
                "perturbation": {
                    "state_key": "custom_label",
                    "start_state": "X",
                    "end_state": "X",
                }
            },
            "/tmp/study",
        )

    def test_accepts_valid_pair(self):
        import data_input_layout as dil

        with mock.patch.object(dil, "unique_states_from_samples", return_value=["AD", "WT"]):
            self.rp._validate_perturbation_states(
                {
                    "perturbation": {
                        "state_key": "disease",
                        "start_state": "AD",
                        "end_state": "WT",
                    }
                },
                "/tmp/study",
            )


def _load_streamlit_app_helpers():
    """Import streamlit_app/app.py with lightweight stubs (no live Streamlit server)."""
    # Ensure a real PyYAML is present even if an earlier test stubbed `yaml`.
    import importlib

    if "yaml" not in sys.modules or not hasattr(sys.modules["yaml"], "safe_load"):
        sys.modules.pop("yaml", None)
    elif getattr(sys.modules["yaml"], "dump", None) is None or sys.modules[
        "yaml"
    ].safe_load.__module__ == "tests.test_auto_batch_and_isp_validation":
        sys.modules.pop("yaml", None)
    # Drop broken stubs that return {} / None.
    yaml_mod = sys.modules.get("yaml")
    if yaml_mod is not None and getattr(yaml_mod, "__file__", None) is None:
        sys.modules.pop("yaml", None)
    importlib.import_module("yaml")

    saved = {
        k: sys.modules.get(k)
        for k in (
            "streamlit",
            "streamlit_upload",
            "streamlit_remote_data",
            "streamlit_app_under_test",
        )
    }

    class _Session(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as e:
                raise AttributeError(key) from e

        def __setattr__(self, key, value):
            self[key] = value

    st = types.ModuleType("streamlit")
    st.session_state = _Session()

    def _passthrough_decorator(*_a, **_k):
        def wrap(fn):
            return fn

        return wrap

    for name in (
        "set_page_config",
        "title",
        "markdown",
        "caption",
        "code",
        "button",
        "radio",
        "number_input",
        "download_button",
        "error",
        "warning",
        "info",
        "success",
        "rerun",
        "stop",
        "columns",
        "expander",
        "tabs",
        "write",
        "text_area",
        "selectbox",
        "file_uploader",
        "checkbox",
        "slider",
        "progress",
        "empty",
        "sidebar",
    ):
        setattr(st, name, lambda *a, **k: None)
    # Decorators used at import time (@st.fragment / @st.cache_*).
    st.fragment = _passthrough_decorator
    st.cache_data = _passthrough_decorator
    st.cache_resource = _passthrough_decorator
    sys.modules["streamlit"] = st

    su = types.ModuleType("streamlit_upload")
    for attr in (
        "import_study_zip",
        "normalize_study_name",
        "resolve_study_tokenize_dir",
        "study_folder",
        "summarize_existing_study",
    ):
        setattr(su, attr, lambda *a, **k: None)
    sys.modules["streamlit_upload"] = su

    srd = types.ModuleType("streamlit_remote_data")

    class RemoteDataError(Exception):
        pass

    srd.RemoteDataError = RemoteDataError
    srd.import_study_from_url = lambda *a, **k: None
    sys.modules["streamlit_remote_data"] = srd

    mod = _load_module("streamlit_app_under_test", ROOT / "webui" / "streamlit_app" / "app.py")
    return mod, st, saved


def _restore_modules(saved: dict) -> None:
    for key, mod in saved.items():
        if mod is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = mod


class TestTailLogSeek(unittest.TestCase):
    def test_seek_based_tail_source_and_behavior(self):
        app_path = ROOT / "webui" / "streamlit_app" / "app.py"
        src = app_path.read_text(encoding="utf-8")
        self.assertIn("f.seek(-max_bytes, os.SEEK_END)", src)
        self.assertNotIn(
            "path.read_bytes()", src.split("def _tail_log")[1].split("def ")[0]
        )

        app, _st, saved = _load_streamlit_app_helpers()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "big.log"
                # 200 KiB of filler + a unique marker so a full-file read would be huge.
                p.write_bytes(b"A" * 200_000 + b"TAIL_MARKER_ONLY")
                out = app._tail_log(p, max_bytes=64)
                self.assertIn("TAIL_MARKER_ONLY", out)
                self.assertIn("showing tail", out)
                self.assertLess(len(out.encode("utf-8")), 200)
                self.assertNotIn("A" * 1000, out)
        finally:
            _restore_modules(saved)


class TestBatchSizeYamlControls(unittest.TestCase):
    def test_auto_and_manual_patch_forward_batch_size(self):
        app, st, saved = _load_streamlit_app_helpers()
        try:
            st.session_state.clear()
            st.session_state["yaml_editor"] = (
                "runtime:\n  forward_batch_size: 50\n"
                "data:\n  input_dir: /app/data/x\n"
            )

            # Manual mode writes an int.
            st.session_state["pipeline_batch_mode"] = app.BATCH_MODE_MANUAL
            st.session_state["pipeline_batch_size"] = 64
            app._apply_batch_size_to_yaml()
            patched = __import__("yaml").safe_load(st.session_state["yaml_editor"])
            self.assertEqual(patched["runtime"]["forward_batch_size"], 64)

            # Auto mode writes the string "auto".
            st.session_state["pipeline_batch_mode"] = app.BATCH_MODE_AUTO
            app._apply_batch_size_to_yaml()
            patched = __import__("yaml").safe_load(st.session_state["yaml_editor"])
            self.assertEqual(patched["runtime"]["forward_batch_size"], "auto")

            # Sync from YAML: "auto" → Auto radio; int → Manual.
            st.session_state.clear()
            st.session_state["yaml_editor"] = "runtime:\n  forward_batch_size: auto\n"
            app._sync_batch_size_controls()
            self.assertEqual(st.session_state["pipeline_batch_mode"], app.BATCH_MODE_AUTO)

            st.session_state.clear()
            st.session_state["yaml_editor"] = "runtime:\n  forward_batch_size: 32\n"
            app._sync_batch_size_controls()
            self.assertEqual(st.session_state["pipeline_batch_mode"], app.BATCH_MODE_MANUAL)
            self.assertEqual(st.session_state["pipeline_batch_size"], 32)
        finally:
            _restore_modules(saved)

    def test_ft_train_batch_size_patch_and_calibrated_apply(self):
        app, st, saved = _load_streamlit_app_helpers()
        try:
            st.session_state.clear()
            st.session_state["yaml_editor"] = (
                "runtime:\n  forward_batch_size: auto\n"
                "data:\n  input_dir: /app/data/x\n"
            )
            st.session_state["pipeline_ft_train_batch_size"] = 32
            app._apply_ft_train_batch_to_yaml()
            patched = __import__("yaml").safe_load(st.session_state["yaml_editor"])
            self.assertEqual(patched["runtime"]["train_batch_size"], 32)
            self.assertEqual(patched["runtime"]["forward_batch_size"], "auto")

            st.session_state["calibrated_ft_batch_size"] = 48
            app._apply_calibrated_ft_batch_to_pipeline()
            self.assertEqual(st.session_state["pipeline_ft_train_batch_size"], 48)
            patched = __import__("yaml").safe_load(st.session_state["yaml_editor"])
            self.assertEqual(patched["runtime"]["train_batch_size"], 48)
        finally:
            _restore_modules(saved)

    def test_parse_ft_batch_marker_and_ingest_json(self):
        app, st, saved = _load_streamlit_app_helpers()
        try:
            self.assertEqual(
                app._parse_ft_batch_from_log("hello\nRECOMMENDED_TRAIN_BATCH_SIZE=64\n"),
                64,
            )
            self.assertIsNone(app._parse_ft_batch_from_log("no marker here\n"))

            with tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp)
                (run_dir / app.FT_BATCH_RESULT_FILENAME).write_text(
                    '{"recommended_train_batch_size": 16}\n',
                    encoding="utf-8",
                )
                st.session_state.clear()
                st.session_state["run_type_sel"] = app.RUN_TYPE_FT_BATCH
                st.session_state["last_run_dir"] = str(run_dir)
                st.session_state["last_exit_code"] = 0
                app._ingest_ft_batch_calibrate_result()
                self.assertEqual(st.session_state["calibrated_ft_batch_size"], 16)
                self.assertEqual(st.session_state["pipeline_ft_train_batch_size"], 16)
        finally:
            _restore_modules(saved)

    def test_ft_calibrate_command_wiring(self):
        app, _st, saved = _load_streamlit_app_helpers()
        try:
            self.assertIn(app.RUN_TYPE_FT_BATCH, app.RUN_FILES)
            cfg = Path("/tmp/fake_ft_calibrate.yaml")
            cmd, env = app._build_command_and_env(app.RUN_TYPE_FT_BATCH, cfg)
            self.assertEqual(cmd[0], "python3")
            self.assertTrue(cmd[1].endswith("run_ft_batch_calibrate.py"))
            self.assertEqual(env.get("FT_BATCH_CALIBRATE_CONFIG"), str(cfg))
        finally:
            _restore_modules(saved)

    def test_sequential_isp_command_wiring(self):
        app, _st, saved = _load_streamlit_app_helpers()
        try:
            self.assertIn(app.RUN_TYPE_SEQUENTIAL_ISP, app.RUN_FILES)
            cfg = Path("/tmp/fake_sequential_isp.yaml")
            cmd, env = app._build_command_and_env(app.RUN_TYPE_SEQUENTIAL_ISP, cfg)
            self.assertEqual(cmd[0], "python3")
            self.assertTrue(cmd[1].endswith("run_sequential_isp.py"))
            self.assertEqual(env.get("SEQUENTIAL_ISP_CONFIG"), str(cfg))
        finally:
            _restore_modules(saved)

    def test_build_sequential_isp_yaml_from_pipeline_run(self):
        app, st, saved = _load_streamlit_app_helpers()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp) / "pipeline_test"
                stage = run_dir / "stage_configs"
                stage.mkdir(parents=True)
                (stage / "isp.yaml").write_text(
                    "paths:\n  dataset: /app/data/x.dataset\n"
                    "  geneformer_model: /app/models/ft\n"
                    "perturbation:\n  state_key: disease\n  start_state: AD\n  end_state: WT\n"
                    "model:\n  type: CellClassifier\n  num_classes: 2\n"
                    "isp:\n  max_ncells: 500\n"
                    "runtime:\n  nproc: 4\n"
                    "species:\n  model_organism: mouse\n",
                    encoding="utf-8",
                )
                st.session_state.clear()
                st.session_state["seq_isp_n_steps"] = 2
                st.session_state["seq_isp_step_0_type"] = "overexpress"
                st.session_state["seq_isp_step_0_genes"] = "Pou5f1\nSox2"
                st.session_state["seq_isp_step_0_name"] = "oskm"
                st.session_state["seq_isp_step_1_type"] = "delete"
                st.session_state["seq_isp_step_1_genes"] = "Igfbp2"
                st.session_state["seq_isp_step_1_name"] = "kd"
                st.session_state["seq_isp_batch_mode"] = app.BATCH_MODE_AUTO
                st.session_state["seq_isp_max_ncells"] = 200
                yaml_text, err = app._build_sequential_isp_yaml_from_pipeline_run(run_dir)
                self.assertIsNone(err, err)
                cfg = __import__("yaml").safe_load(yaml_text)
                self.assertEqual(cfg["paths"]["dataset"], "/app/data/x.dataset")
                self.assertTrue(str(cfg["paths"]["output_root"]).endswith("sequential_isp"))
                self.assertFalse(cfg["paths"]["output_time_subdir"])
                self.assertEqual(cfg["sequential"]["steps"][0]["type"], "overexpress")
                self.assertEqual(cfg["sequential"]["steps"][0]["genes"], ["Pou5f1", "Sox2"])
                self.assertEqual(cfg["sequential"]["steps"][1]["type"], "delete")
                self.assertEqual(cfg["runtime"]["forward_batch_size"], "auto")
                self.assertEqual(cfg["isp"]["max_ncells"], 200)
        finally:
            _restore_modules(saved)


class TestPipelineRunZip(unittest.TestCase):
    def test_build_pipeline_run_zip_skips_heavy_artifacts(self):
        app, _st, saved = _load_streamlit_app_helpers()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp) / "pipeline_smokezip_000000Z"
                (run_dir / "tokenized_dataset" / "smoke_0.dataset").mkdir(parents=True)
                (run_dir / "tokenized_dataset" / "smoke_0.dataset" / "data.arrow").write_bytes(
                    b"HUGE_TOKENIZED"
                )
                (run_dir / "finetune" / "all_run1" / "checkpoint-32").mkdir(parents=True)
                (
                    run_dir / "finetune" / "all_run1" / "checkpoint-32" / "model.safetensors"
                ).write_bytes(b"HUGE_CKPT")
                (run_dir / "finetune" / "all_run1").mkdir(parents=True, exist_ok=True)
                (run_dir / "finetune" / "all_run1" / "label_dict.json").write_text(
                    '{"AD": 0, "WT": 1}\n', encoding="utf-8"
                )
                (run_dir / "pipeline_run.log").write_text("ok\n", encoding="utf-8")
                (run_dir / "isp_results").mkdir()
                (run_dir / "isp_results" / "stats.csv").write_text("a,b\n1,2\n", encoding="utf-8")
                (run_dir / "loom_files" / "x.loom").parent.mkdir(parents=True, exist_ok=True)
                (run_dir / "loom_files" / "x.loom").write_bytes(b"LOOM")

                payload = app._build_pipeline_run_zip(run_dir)
                self.assertIsNotNone(payload)
                data, name = payload
                self.assertTrue(name.endswith(".zip"))
                self.assertTrue(name.startswith("pipeline_"))

                import io
                import zipfile

                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    names = set(zf.namelist())
                joined = "\n".join(names)
                self.assertIn("pipeline_run.log", joined)
                self.assertIn("label_dict.json", joined)
                self.assertIn("stats.csv", joined)
                self.assertNotIn("tokenized_dataset", joined)
                self.assertNotIn("checkpoint-", joined)
                self.assertNotIn("loom_files", joined)
                self.assertNotIn("HUGE_TOKENIZED", joined)
        finally:
            _restore_modules(saved)


if __name__ == "__main__":
    unittest.main()
