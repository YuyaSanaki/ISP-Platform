"""
In-silico perturbation driver (notebook logic as a script).

Configuration: YAML file (default /app/core/config/isp.yaml). Override path with --config or ISP_CONFIG.
CLI --forward-batch-size / --nproc override the YAML runtime section when passed.
Outputs: with paths.output_root, writes to {output_root}/[{YYYYMMDD}/][run_folder/]isp_results and .../ispstats_results.
Date folder (default on): --output-date, ISP_OUTPUT_DATE, or today; disable with paths.output_date_subdir false, ISP_OUTPUT_DATE_SUBDIR=0, or --no-output-date-subdir.
Time folder (default on): run_<UTC HHMMSS>_<microseconds>Z under the date (or output_root) folder; disable with paths.output_time_subdir false, ISP_OUTPUT_TIME_SUBDIR=0, or --no-output-time-subdir.
Explicit output_subdir (paths.output_subdir, ISP_OUTPUT_SUBDIR, --output-subdir) overrides the time folder with a fixed name.
Also writes {output_root}/{YYYYMMDD}/[run_folder/]isp_run.log (rotating; env ISP_LOG_MAX_BYTES, ISP_LOG_BACKUP_COUNT; ISP_DISABLE_RUN_LOG=1 to skip) mirroring stdout/stderr on the main process only.
Optional ISP_FINGERPRINT_INPUTS=1: SHA-256 content fingerprints of paths.dataset and paths.geneformer_model are printed and stored in isp_run_metadata.yaml (can be slow for large trees).

Speed on DGX / large GPUs:
  - Imports apply TF32-friendly matmul settings (see in_silico_perturber._configure_cuda_performance).
  - Tune forward_batch_size in config until just below OOM.
  - Optional: GENEFORMER_TORCH_COMPILE=1 for torch.compile (PyTorch 2+).
  - Multi-GPU: accelerate launch --num_processes <ngpu> run_isp.py [--config ...]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from accelerate import Accelerator
from geneformer import InSilicoPerturber, InSilicoPerturberStats
from geneformer.auto_batch_size import coerce_batch_size
from geneformer.species_context import (
    backend_from_config,
    default_isp_forward_batch_size,
    convert_perturbation_genes,
    load_input_symbol_table,
    log_species_banner,
    species_from_config,
)

from run_pipeline_log import format_isp_run_banner, install_rotating_stdio_tee


def _deep_get(m: Mapping[str, Any] | None, *keys: str, default: Any = None) -> Any:
    cur: Any = m
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _ensure_dir_suffix(path: str) -> str:
    return path if path.endswith(os.sep) else path + os.sep


def _resolve_output_subdir(
    cli: str | None,
    env: str | None,
    paths_cfg: Mapping[str, Any] | None,
) -> str | None:
    """Single folder segment under {output_root}/{YYYYMMDD}/ for isolating same-day runs."""
    raw = (cli or env or _deep_get(paths_cfg, "output_subdir", default="") or "").strip()
    if not raw:
        return None
    if raw in (".", "..") or "/" in raw or "\\" in raw or "\x00" in raw:
        raise ValueError(
            "output_subdir must be one path segment (no slashes), not '.' or '..'. "
            "Set paths.output_subdir, ISP_OUTPUT_SUBDIR, or --output-subdir (e.g. run1, batch_a)."
        )
    return raw


def _want_date_subdir(
    paths_cfg: Mapping[str, Any] | None,
    env_raw: str | None,
    cli_disable: bool,
) -> bool:
    """Default True: insert {YYYYMMDD} under paths.output_root unless disabled."""
    if cli_disable:
        return False
    s = (env_raw or "").strip().lower()
    if s in ("0", "false", "no", "off"):
        return False
    if s in ("1", "true", "yes", "on"):
        return True
    v = _deep_get(paths_cfg, "output_date_subdir", default=True)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _want_auto_time_subdir(
    paths_cfg: Mapping[str, Any] | None,
    env_raw: str | None,
    cli_disable: bool,
) -> bool:
    """Default True: create run_<UTC time> under the date folder unless disabled."""
    if cli_disable:
        return False
    s = (env_raw or "").strip().lower()
    if s in ("0", "false", "no", "off"):
        return False
    if s in ("1", "true", "yes", "on"):
        return True
    v = _deep_get(paths_cfg, "output_time_subdir", default=True)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _utc_run_folder_name(now_utc: datetime) -> str:
    """One path segment: UTC start time + microseconds to avoid same-second clashes."""
    return f"isp_{now_utc.strftime('%H%M%S')}_{now_utc.microsecond:06d}Z"


def isp_stats_dictionary_kwargs(backend: Any) -> dict[str, Any]:
    """Token/symbol dict paths for InSilicoPerturberStats, matched to the model backend."""
    kwargs: dict[str, Any] = {"token_dictionary_file": backend.token_dictionary}
    if getattr(backend, "gene_symbol_to_ensembl", None) is not None:
        kwargs["gene_name_id_dictionary_file"] = backend.gene_symbol_to_ensembl
    return kwargs


def _broadcast_run_folder_from_main(accelerator: Any, value_on_main: str) -> str:
    """
    Replicate the run folder name from the global main process on all ranks.
    Older Accelerate versions omit broadcast_object_list; then we try torch.distributed.
    """
    n_raw = getattr(accelerator, "num_processes", 1)
    try:
        n = int(n_raw) if n_raw is not None else 1
    except (TypeError, ValueError):
        n = 1
    if n <= 1:
        return value_on_main

    bucket: list[str] = [value_on_main if accelerator.is_main_process else ""]
    acc_bc = getattr(accelerator, "broadcast_object_list", None)
    if callable(acc_bc):
        acc_bc(bucket, from_process=0)
        out = bucket[0]
        if isinstance(out, str) and out:
            return out
        raise RuntimeError("broadcast_object_list did not return a run folder name on all ranks.")

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            obj_list: list[Any]
            if dist.get_rank() == 0:
                obj_list = [value_on_main]
            else:
                obj_list = [None]
            dist.broadcast_object_list(obj_list, src=0)
            got = obj_list[0]
            if isinstance(got, str) and got:
                return got
    except Exception:
        pass

    raise RuntimeError(
        "Multi-process ISP with paths.output_time_subdir needs a way to broadcast the run folder "
        "name across ranks. This Accelerate build has no broadcast_object_list, and "
        "torch.distributed is not initialized. Options: upgrade `accelerate`, or set "
        "paths.output_time_subdir: false / ISP_OUTPUT_TIME_SUBDIR=0 / --no-output-time-subdir, "
        "or set a fixed ISP_OUTPUT_SUBDIR."
    )


def load_isp_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"ISP config not found: {path}. Create config/isp.yaml or pass --config."
        )
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"ISP config must be a mapping at root: {path}")
    return data


def _write_run_provenance(
    run_root: Path,
    config_path: Path,
    forward_batch_size: int | str,
    nproc: int,
    cli_forward_batch_size: int | str | None,
    cli_nproc: int | None,
    input_fingerprints: dict[str, Any] | None = None,
) -> None:
    """Copy the ISP YAML and record metadata next to dated outputs (reproducibility)."""
    run_root.mkdir(parents=True, exist_ok=True)
    dest_cfg = run_root / "isp_config_used.yaml"
    shutil.copy2(config_path, dest_cfg)

    meta: dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_source_path": str(config_path.resolve()),
        "effective_runtime": {
            "forward_batch_size": forward_batch_size,
            "nproc": nproc,
        },
        "cli_overrides": {
            k: v
            for k, v in (
                ("forward_batch_size", cli_forward_batch_size),
                ("nproc", cli_nproc),
            )
            if v is not None
        },
    }
    git_sha = os.environ.get("ISP_GIT_COMMIT", "").strip()
    if not git_sha:
        try:
            rev = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parent,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if rev.returncode == 0:
                git_sha = rev.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    if git_sha:
        meta["git_commit"] = git_sha
    if input_fingerprints:
        meta["input_content_fingerprints"] = input_fingerprints

    with open(run_root / "isp_run_metadata.yaml", "w", encoding="utf-8") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)


def _merge_isp_run_metadata(run_root: Path, updates: dict[str, Any]) -> None:
    """Append keys to isp_run_metadata.yaml if it exists (e.g. finished_at_utc)."""
    path = run_root / "isp_run_metadata.yaml"
    if not path.is_file():
        return
    with open(path, encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    if not isinstance(meta, dict):
        meta = {}
    meta.update(updates)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)


def main() -> None:
    accelerator = Accelerator()
    default_cfg = os.environ.get("ISP_CONFIG", "/app/core/config/isp.yaml")
    p = argparse.ArgumentParser(description="Run Geneformer in-silico perturbation + stats.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path(default_cfg),
        help="YAML config path (default: ISP_CONFIG or /app/core/config/isp.yaml).",
    )
    p.add_argument(
        "--forward-batch-size",
        default=None,
        metavar="N|auto",
        help="Override config runtime.forward_batch_size ('auto' measures this GPU).",
    )
    p.add_argument(
        "--nproc",
        type=int,
        default=None,
        help="Override config runtime.nproc.",
    )
    p.add_argument(
        "--output-date",
        default=None,
        metavar="YYYYMMDD",
        help="Date folder under output_root (default: today, or ISP_OUTPUT_DATE).",
    )
    p.add_argument(
        "--output-subdir",
        default=None,
        metavar="NAME",
        help="Optional subfolder under output_root/DATE for this run (default: paths.output_subdir or ISP_OUTPUT_SUBDIR).",
    )
    p.add_argument(
        "--no-output-time-subdir",
        action="store_true",
        help="Write isp_results directly under output_root/DATE (no default run_<UTC time> folder).",
    )
    p.add_argument(
        "--no-output-date-subdir",
        action="store_true",
        help="Write isp_results directly under paths.output_root (no {YYYYMMDD} folder).",
    )
    p.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip isp_analysis.py figures/tables after stats.",
    )
    p.add_argument(
        "--skip-isp-umap",
        action="store_true",
        help="Skip ISP UMAP trajectories (normally auto-runs for targeted genes).",
    )
    args = p.parse_args()

    cfg = load_isp_config(args.config)

    species = species_from_config(cfg)
    backend = backend_from_config(cfg)
    cfg.setdefault("model", {})
    yaml_mis = cfg["model"].get("max_input_size")
    cfg["model"]["max_input_size"] = int(backend.max_input_size)
    if yaml_mis is not None and int(yaml_mis) != int(backend.max_input_size):
        print(
            f"  NOTE: model.max_input_size {yaml_mis} → {backend.max_input_size} "
            f"(from species backend {species.get('model')})"
        )

    forward_batch_size = args.forward_batch_size
    if forward_batch_size is None:
        rt_fb = _deep_get(cfg, "runtime", "forward_batch_size")
        if rt_fb is not None:
            forward_batch_size = rt_fb
        elif os.environ.get("ISP_FORWARD_BATCH_SIZE"):
            forward_batch_size = os.environ["ISP_FORWARD_BATCH_SIZE"]
        else:
            forward_batch_size = default_isp_forward_batch_size(backend.max_input_size)
            print(
                f"  NOTE: runtime.forward_batch_size default → {forward_batch_size} "
                f"(from backend max_input_size {backend.max_input_size})"
            )
    species_default = default_isp_forward_batch_size(backend.max_input_size)
    forward_batch_size = coerce_batch_size(forward_batch_size, default=species_default)
    nproc = args.nproc
    if nproc is None:
        nproc = int(_deep_get(cfg, "runtime", "nproc", default=os.environ.get("ISP_NPROC", "8")))

    paths = cfg.get("paths") or {}
    dataset_name = paths.get("dataset")
    model_dir = paths.get("geneformer_model")
    output_root = paths.get("output_root")
    isp_out: str | None
    stats_out: str | None
    date_used: str | None = None
    run_root: Path | None = None
    if output_root:
        use_date_subdir = _want_date_subdir(
            paths,
            os.environ.get("ISP_OUTPUT_DATE_SUBDIR"),
            args.no_output_date_subdir,
        )
        date_used = None
        if use_date_subdir:
            date_used = (
                args.output_date or os.environ.get("ISP_OUTPUT_DATE") or datetime.now().strftime("%Y%m%d")
            ).strip()
            if len(date_used) != 8 or not date_used.isdigit():
                raise ValueError(
                    "output date must be YYYYMMDD (8 digits). "
                    "Set --output-date, ISP_OUTPUT_DATE, or use default today."
                )
        explicit_subdir = _resolve_output_subdir(
            args.output_subdir,
            os.environ.get("ISP_OUTPUT_SUBDIR"),
            paths,
        )
        use_auto_time = _want_auto_time_subdir(
            paths,
            os.environ.get("ISP_OUTPUT_TIME_SUBDIR"),
            args.no_output_time_subdir,
        )
        run_folder = ""
        if explicit_subdir:
            run_folder = explicit_subdir
        elif use_auto_time:
            proposed = ""
            if accelerator.is_main_process:
                proposed = _utc_run_folder_name(datetime.now(timezone.utc))
            run_folder = _broadcast_run_folder_from_main(accelerator, proposed)
        run_root = Path(output_root)
        if use_date_subdir and date_used:
            run_root = run_root / date_used
        if run_folder:
            run_root = run_root / run_folder
        isp_out = os.path.join(str(run_root), "isp_results")
        stats_out = os.path.join(str(run_root), "ispstats_results")
    else:
        run_folder = ""
        isp_out = paths.get("isp_results_dir")
        stats_out = paths.get("ispstats_results_dir")
    for key, val in (
        ("paths.dataset", dataset_name),
        ("paths.geneformer_model", model_dir),
        ("paths output (isp or legacy)", isp_out),
        ("paths output (stats or legacy)", stats_out),
    ):
        if not val:
            raise ValueError(
                f"Missing required config key: {key}. "
                "Set paths.output_root (recommended) or paths.isp_results_dir and paths.ispstats_results_dir."
            )

    pert = cfg.get("perturbation") or {}
    select_perturb_type = pert.get("type", "delete")
    start_state = pert.get("start_state")
    end_state = pert.get("end_state")
    if start_state is None or end_state is None:
        raise ValueError("config perturbation.start_state and perturbation.end_state are required.")
    alt_state = list(pert.get("alt_states") or [])
    organ_data = pert.get("organ_data", "experiment")
    genes_to_perturb_list = list(pert.get("genes_to_perturb") or [])
    state_key = pert.get("state_key", "disease")
    perturb_gene_labels: dict[str, str] = {}

    if accelerator.is_main_process:
        log_species_banner(species, prefix="  ")
    if genes_to_perturb_list:
        symbol_table = load_input_symbol_table(species)
        genes_to_perturb_list, perturb_conv = convert_perturbation_genes(
            genes_to_perturb_list,
            species,
            input_symbol_table=symbol_table,
        )
        perturb_gene_labels = {target: source for source, target in perturb_conv.mapped.items()}
        if not genes_to_perturb_list:
            raise ValueError(
                "No perturbation genes remained after species/ortholog conversion. "
                "Check species config and ortholog tables."
            )

    mdl = cfg.get("model") or {}
    use_model_type = mdl.get("type", "Pretrained")
    num_classes = int(mdl.get("num_classes", 0))

    isp_cfg = cfg.get("isp") or {}
    max_ncells = isp_cfg.get("max_ncells")
    emb_layer = int(isp_cfg.get("emb_layer", -1))
    emb_mode = isp_cfg.get("emb_mode", "cell")
    cell_emb_style = isp_cfg.get("cell_emb_style", "mean_pool")
    combos = int(isp_cfg.get("combos", 0))
    anchor_gene = isp_cfg.get("anchor_gene")
    perturb_rank_shift = isp_cfg.get("perturb_rank_shift")
    perturb_rank_direct_shift = isp_cfg.get("perturb_rank_direct_shift")
    filter_data = isp_cfg.get("filter_data")

    st = cfg.get("stats") or {}
    stats_mode = st.get("mode", "goal_state_shift")

    start_state_fn = start_state.replace(" ", "-")
    end_state_fn = end_state.replace(" ", "-")
    output_prefix = "output_in-silico_SE{}_OR{}_ST{}_EN{}".format(
        select_perturb_type, organ_data, start_state_fn, end_state_fn
    )

    isp_dir = _ensure_dir_suffix(str(isp_out))
    os.makedirs(isp_dir, exist_ok=True)

    if run_root is not None:
        log_dir = run_root
    else:
        log_dir = Path(isp_out.rstrip(os.sep)).parent
    log_path = log_dir / "isp_run.log"

    input_fps: dict[str, Any] | None = None
    if accelerator.is_main_process:
        install_rotating_stdio_tee(log_path, env_prefix="ISP")
        
        # Read provenance settings from config with overrides
        prov_cfg = cfg.get("provenance") or {}
        enable_fp = prov_cfg.get("enable_input_fingerprint", False)
        if os.environ.get("ISP_FINGERPRINT_INPUTS"):
            enable_fp = os.environ.get("ISP_FINGERPRINT_INPUTS").lower() in ("1", "true", "yes")
        
        is_fast = prov_cfg.get("fingerprint_fast", True)
        if os.environ.get("ISP_FINGERPRINT_FAST"):
            is_fast = os.environ.get("ISP_FINGERPRINT_FAST").lower() in ("1", "true", "yes")

        if enable_fp:
            from run_input_fingerprint import fingerprint_isp_inputs

            mode_str = "(fast mode)" if is_fast else "(full content mode)"
            print(f"Computing input content fingerprints {mode_str}...", flush=True)
            input_fps = fingerprint_isp_inputs(str(dataset_name), str(model_dir), fast=is_fast)
            for key, block in input_fps.items():
                digest = block.get("content_fingerprint_sha256")
                err = block.get("error")
                print(
                    f"  {key} content_fingerprint_sha256: {digest or f'({err})'}",
                    flush=True,
                )

        if not output_root:
            banner_run_folder = "(legacy paths)"
        elif run_folder:
            banner_run_folder = run_folder
        elif date_used:
            banner_run_folder = "(flat under date)"
        else:
            banner_run_folder = "(flat under output_root)"
        banner = format_isp_run_banner(
            cfg,
            extras={
                "config_path": str(args.config.resolve()),
                "date_used": date_used or "",
                "isp_out": isp_out,
                "stats_out": stats_out,
                "output_prefix": output_prefix,
                "forward_batch_size": forward_batch_size,
                "nproc": nproc,
                "skip_analysis": args.skip_analysis,
                "run_folder": banner_run_folder,
            },
        )
        print(banner, end="")
        print(f"  run log (rotating): {log_path}")

    print("Checking CUDA...", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))

    token_dict_path = backend.token_dictionary

    isp = InSilicoPerturber(
        perturb_type=select_perturb_type,
        perturb_rank_shift=perturb_rank_shift,
        perturb_rank_direct_shift=perturb_rank_direct_shift,
        genes_to_perturb="all" if len(genes_to_perturb_list) == 0 else genes_to_perturb_list,
        combos=combos,
        anchor_gene=anchor_gene,
        model_type=use_model_type,
        num_classes=num_classes,
        emb_mode=emb_mode,
        cell_emb_style=cell_emb_style,
        filter_data=filter_data,
        cell_states_to_model={
            "state_key": state_key,
            "start_state": start_state,
            "goal_state": end_state,
            "alt_states": alt_state,
        },
        max_ncells=max_ncells,
        emb_layer=emb_layer,
        forward_batch_size=forward_batch_size,
        nproc=nproc,
        token_dictionary_file=token_dict_path,
        model_input_size=int(backend.max_input_size),
    )

    if output_root:
        if date_used:
            print(f"Output root: {output_root}  (date {date_used})")
        else:
            print(f"Output root: {output_root}  (no date subfolder)")
        print(f"  → ISP results: {isp_out}")
        print(f"  → ISP stats:   {stats_out}")

    if run_root is not None and accelerator.is_main_process:
        _write_run_provenance(
            run_root,
            args.config.resolve(),
            forward_batch_size,
            nproc,
            args.forward_batch_size,
            args.nproc,
            input_fingerprints=input_fps,
        )
        print(
            f"  → Provenance: {run_root / 'isp_config_used.yaml'} "
            f"+ isp_run_metadata.yaml"
        )

    tracker_root: Path | None = run_root
    main_pipeline_ok = False

    try:
        print("Starting perturbation...")
        isp.perturb_data(model_dir, dataset_name, isp_dir, output_prefix)

        # Wait for all processes to finish perturbing
        accelerator.wait_for_everyone()

        # Only run analysis on the main process to avoid race conditions
        if accelerator.is_main_process:
            print("Perturbation complete. Generating stats...")
            # Match token/symbol dicts to the model backend (human ENSG vs mouse ENSMUSG).
            ispstats = InSilicoPerturberStats(
                mode=stats_mode,
                genes_perturbed=(
                    "all" if len(genes_to_perturb_list) == 0 else genes_to_perturb_list
                ),
                combos=combos,
                anchor_gene=anchor_gene,
                cell_states_to_model={
                    "state_key": state_key,
                    "start_state": start_state,
                    "goal_state": end_state,
                    "alt_states": alt_state,
                },
                **isp_stats_dictionary_kwargs(backend),
            )

            stats_dir = _ensure_dir_suffix(str(stats_out))
            os.makedirs(stats_dir, exist_ok=True)

            ispstats.get_stats(isp_dir, None, stats_dir, output_prefix)
            print("Stats generation complete. Check the parquet file.")

            stats_parent = Path(stats_dir.rstrip(os.sep))
            analysis_cfg = cfg.get("analysis") or {}
            run_figures = analysis_cfg.get("enabled", True) and not args.skip_analysis
            if run_figures:
                from isp_analysis import run_isp_figure_analysis

                parquet_file = os.path.join(stats_dir.rstrip(os.sep), f"{output_prefix}.parquet")
                figures_dir = stats_parent.parent / "figures"
                figures_dir.mkdir(parents=True, exist_ok=True)
                print("Running ISP figure analysis (isp_analysis.py)...")
                isp_max_ncells = (cfg.get("isp") or {}).get("max_ncells")
                try:
                    isp_max_ncells = int(isp_max_ncells) if isp_max_ncells is not None else None
                except (TypeError, ValueError):
                    isp_max_ncells = None
                run_isp_figure_analysis(
                    parquet_file,
                    figures_dir,
                    stats_parent,
                    label_start=start_state,
                    label_end=end_state,
                    n_isp_cells=isp_max_ncells,
                )
                print(f"Figures directory: {figures_dir}")

            if (
                not args.skip_isp_umap
                and genes_to_perturb_list
                and (cfg.get("umap") or {}).get("enabled", True)
            ):
                from run_isp_umap import run_isp_umap_for_targeted_genes

                umap_root = (run_root if run_root is not None else stats_parent.parent) / "isp_umap"
                print(f"Running ISP UMAP for {len(genes_to_perturb_list)} targeted gene(s)...")
                umap_dirs = run_isp_umap_for_targeted_genes(
                    cfg,
                    genes_to_perturb_list,
                    gene_labels=perturb_gene_labels,
                    output_root=umap_root,
                )
                print(f"ISP UMAP outputs: {umap_dirs[0].parent if umap_dirs else umap_root}")

            main_pipeline_ok = True
    finally:
        if accelerator.is_main_process and tracker_root is not None:
            _merge_isp_run_metadata(
                tracker_root,
                {
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "run_status": "completed" if main_pipeline_ok else "failed",
                },
            )


if __name__ == "__main__":
    main()
