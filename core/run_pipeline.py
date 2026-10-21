"""
End-to-end pipeline: Tokenize → Fine-tune → ISP.

Reads config/pipeline.yaml (or PIPELINE_CONFIG), derives chained paths from
data.input_dir, writes per-stage configs under the run directory, and runs
each stage sequentially.

Usage:
  docker compose run --rm pipeline
  python3 run_pipeline.py --config /app/core/config/my_pipeline.yaml
"""
from __future__ import annotations

import argparse
import codecs
import json
import os
import subprocess
import sys
import csv
from datetime import datetime, timezone
from pathlib import Path

import yaml

from pipeline_lib import (
    ROOT,
    build_finetune_config,
    build_isp_config,
    build_tokenize_config,
    format_pipeline_banner,
    load_yaml,
    remove_tokenized_outputs,
    resolve_finetune_model_dir,
    resolve_pipeline_paths,
    resolve_tokenized_dataset,
    study_name_from_input_dir,
    write_yaml,
)
from data_input_layout import unique_states_from_samples
from run_pipeline_log import install_rotating_stdio_tee
from platform_paths import local_path, localize_path
from run_provenance import update_service_provenance, write_service_provenance


# state_key values that tokenize fills from the sample folder name (Time-State-Suffix)
_PATH_DERIVED_STATE_KEYS = frozenset({"disease", "genotype", "state"})


def _utc_pipeline_folder_name(now_utc: datetime) -> str:
    return f"pipeline_{now_utc.strftime('%H%M%S')}_{now_utc.microsecond:06d}Z"


RUN_DIR_FILE_ENV = "ISP_PIPELINE_RUN_DIR_FILE"


def _report_run_dir(run_dir: Path) -> None:
    """Write the run folder to the file named in ISP_PIPELINE_RUN_DIR_FILE (Web UI jobs)."""
    target = os.environ.get(RUN_DIR_FILE_ENV, "").strip()
    if not target:
        return
    try:
        Path(target).write_text(f"{run_dir.resolve()}\n", encoding="utf-8")
    except OSError as exc:
        print(f"WARNING: could not write {RUN_DIR_FILE_ENV}={target}: {exc}", flush=True)


def _validate_perturbation_states(pipeline: dict, study_root: str) -> None:
    """Fail before tokenize/fine-tune when ISP states cannot exist or are not unique."""
    perturbation = pipeline.get("perturbation") or {}
    state_key = str(perturbation.get("state_key") or "").strip()
    if state_key not in _PATH_DERIVED_STATE_KEYS:
        return

    start = str(perturbation.get("start_state") or "").strip()
    end = str(perturbation.get("end_state") or "").strip()
    if start and end and start == end:
        raise ValueError(
            f"ISP start_state and end_state are both `{start}`. "
            "Geneformer requires distinct states for goal-state-shift ISP "
            "(Web UI: pick different ISP start_state / end_state, e.g. AD → WT)."
        )

    available = unique_states_from_samples(Path(study_root))
    if not available:
        return

    wanted = [start, end]
    wanted += list(perturbation.get("alt_states") or [])
    missing = sorted({str(v) for v in wanted if v} - set(available))
    if missing:
        raise ValueError(
            f"ISP perturbation states {', '.join(missing)} are not in the study data. "
            f"Sample folders provide {state_key}: {', '.join(available)}. "
            "Fix perturbation.start_state / end_state (Web UI: ISP state dropdowns)."
        )


def _run_subprocess(cmd: list[str], env: dict[str, str], label: str) -> None:
    """Run a stage and relay its stdout and stderr through sys.stdout.

    The child inherits file descriptors, not the Python-level tee of pipeline_run.log, so
    its output is piped and written here to reach both the console and the log.
    """
    print(f"\n>>> Starting {label}: {' '.join(cmd)}\n", flush=True)
    env = dict(env)
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read1(65536)
            if not chunk:
                break
            sys.stdout.write(decoder.decode(chunk))
            sys.stdout.flush()
        sys.stdout.write(decoder.decode(b"", final=True))
        returncode = proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)
    print(f"\n>>> {label} finished OK.\n", flush=True)


def _ortholog_approval_stop_message(tokenized_dir: Path) -> str | None:
    """Gate message when tokenize ended on an approved stop decision, else None."""
    meta = tokenized_dir / "tokenize_run_metadata.yaml"
    if not meta.is_file():
        return None
    with open(meta, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if data.get("run_status") != "stopped_ortholog_approval":
        return None
    gate = data.get("ortholog_loss_gate") or {}
    return str(gate.get("message") or "The ortholog approval decision stopped tokenize.")


def _num_classes_from_label_dict(model_dir: Path) -> int:
    path = model_dir / "label_dict.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Fine-tune checkpoint missing label_dict.json: {path}. "
            "Check finetune logs; ISP needs num_classes from this file."
        )
    with open(path, encoding="utf-8") as f:
        label_dict = json.load(f)
    if not isinstance(label_dict, dict) or not label_dict:
        raise ValueError(f"Invalid label_dict.json: {path}")
    return len(label_dict)


def _gene_token_from_row(row: dict) -> str:
    """Prefer Ensembl_ID for robust token resolution; fall back to Gene_name."""
    return str(row.get("Ensembl_ID") or row.get("Gene_name") or "").strip()


def pick_e2e_top1_significant_gene(ispstats_dir: Path) -> str | None:
    """Pick TOP1 significant gene (max Shift_to_goal_end among Sig==1) for E2E UMAP.

    Uses ``significant_genes.csv`` when present. Falls back to the ISP stats
    parquet filtered to ``Sig==1``. Does **not** use ``top100_positive_shifters.csv``
    (that table includes non-significant high-shift genes).
    """
    sig_csv = Path(ispstats_dir) / "significant_genes.csv"
    if sig_csv.is_file():
        try:
            with open(sig_csv, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows and "Shift_to_goal_end" in rows[0]:
                best_row = max(
                    rows,
                    key=lambda r: float(r.get("Shift_to_goal_end") or "-inf"),
                )
                gene = _gene_token_from_row(best_row)
                if gene:
                    return gene
        except Exception:
            pass

    try:
        parquets = sorted(Path(ispstats_dir).glob("*.parquet"))
        if not parquets:
            return None
        import pandas as pd  # local import: avoid hard dependency at module import time

        df = pd.read_parquet(parquets[0])
        if df.empty or "Shift_to_goal_end" not in df.columns:
            return None
        if "Sig" in df.columns:
            df = df[df["Sig"] == 1]
        else:
            return None
        if df.empty:
            return None
        row = df.sort_values("Shift_to_goal_end", ascending=False).iloc[0]
        gene = str(row.get("Ensembl_ID") or row.get("Gene_name") or "").strip()
        return gene or None
    except Exception:
        return None


def main() -> None:
    default_cfg = os.environ.get("PIPELINE_CONFIG", str(ROOT / "config" / "pipeline.yaml"))
    p = argparse.ArgumentParser(description="Run Tokenize → Fine-tune → ISP pipeline.")
    p.add_argument(
        "--config",
        type=local_path,
        default=Path(default_cfg),
        help="Pipeline YAML (default: PIPELINE_CONFIG or config/pipeline.yaml).",
    )
    p.add_argument(
        "--skip-tokenize",
        action="store_true",
        help="Skip tokenization (tokenized_dataset/{study}_<n>.dataset must already exist).",
    )
    p.add_argument(
        "--skip-finetune",
        action="store_true",
        help="Skip fine-tuning (use the existing checkpoint under finetune/{group}_run{k}).",
    )
    p.add_argument(
        "--skip-isp",
        action="store_true",
        help="Stop after fine-tuning.",
    )
    p.add_argument(
        "--run-dir",
        type=local_path,
        default=None,
        help="Resume an existing pipeline run directory (required with --skip-tokenize).",
    )
    p.add_argument(
        "--ortholog-curated-overlay",
        type=localize_path,
        default=None,
        help=(
            "Optional project curated ortholog overlay (TSV file or directory with "
            "curated_bridge_{pair}.tsv). Sets species.ortholog_curated_overlay and "
            "GENEFORMER_ORTHOLOG_CURATED_OVERLAY for all stages."
        ),
    )
    p.add_argument(
        "--ortholog-audit",
        type=localize_path,
        default=None,
        help=(
            "Path to analysis_manifest.yaml (or ortholog_audit YAML) for tokenize "
            "ortholog_loss_gate."
        ),
    )
    p.add_argument(
        "--ortholog-approval-record",
        type=localize_path,
        default=None,
        help=(
            "NEW approved YAML (status: approved) for ortholog_loss_gate; "
            "not the pending ortholog_approval_request.yaml. "
            "Decisions: stop|curated_bridge|sensitivity_best_of_n|mark_non_critical "
            "(aliases A|B|C|D)."
        ),
    )
    args = p.parse_args()

    pipeline_path = args.config.expanduser().resolve()
    pipeline = load_yaml(pipeline_path)

    if args.ortholog_curated_overlay:
        overlay = str(Path(args.ortholog_curated_overlay).expanduser().resolve())
        os.environ["GENEFORMER_ORTHOLOG_CURATED_OVERLAY"] = overlay
        pipeline = dict(pipeline)
        species = dict(pipeline.get("species") or {})
        species["ortholog_curated_overlay"] = overlay
        pipeline["species"] = species
    elif (pipeline.get("species") or {}).get("ortholog_curated_overlay"):
        os.environ.setdefault(
            "GENEFORMER_ORTHOLOG_CURATED_OVERLAY",
            str(
                Path(pipeline["species"]["ortholog_curated_overlay"]).expanduser().resolve()
            ),
        )

    if args.ortholog_audit or args.ortholog_approval_record:
        pipeline = dict(pipeline)
        stages = dict(pipeline.get("stages") or {})
        tok_stage = dict(stages.get("tokenize") or {})
        tok_cfg = dict(tok_stage.get("tokenizer") or {})
        if args.ortholog_audit:
            tok_cfg["ortholog_audit"] = str(
                Path(args.ortholog_audit).expanduser().resolve()
            )
            tok_cfg.setdefault("ortholog_loss_gate", True)
        if args.ortholog_approval_record:
            tok_cfg["ortholog_approval_record"] = str(
                Path(args.ortholog_approval_record).expanduser().resolve()
            )
        tok_stage["tokenizer"] = tok_cfg
        stages["tokenize"] = tok_stage
        pipeline["stages"] = stages

    data = pipeline.get("data") or {}
    if not data.get("input_dir"):
        raise ValueError("pipeline data.input_dir is required.")

    if data.get("output_prefix") in (None, "", "null"):
        study = study_name_from_input_dir(str(data["input_dir"]))
        pipeline = dict(pipeline)
        pipeline["data"] = dict(data)
        pipeline["data"]["output_prefix"] = study

    date_used = datetime.now().strftime("%Y%m%d")
    paths_cfg = pipeline.get("paths") or {}
    output_root = Path(str(paths_cfg.get("output_root") or localize_path("/app/output")))
    explicit_run_dir = args.run_dir or paths_cfg.get("run_dir") or paths_cfg.get("pipeline_run_dir")
    if args.skip_tokenize and not explicit_run_dir:
        p.error(
            "--skip-tokenize needs --run-dir (or paths.run_dir): the tokenized dataset is read "
            "from <run>/tokenized_dataset/, and a new run folder has none."
        )
    if explicit_run_dir:
        run_dir = Path(str(explicit_run_dir)).expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Pipeline run directory not found: {run_dir}")
        print(f"Resuming pipeline run: {run_dir}")
    else:
        now_utc = datetime.now(timezone.utc)
        run_dir = output_root / date_used / _utc_pipeline_folder_name(now_utc)
        run_dir.mkdir(parents=True, exist_ok=True)
    _report_run_dir(run_dir)

    resolved = resolve_pipeline_paths(pipeline, run_dir)
    # ISP is optional for some E2E workflows (UMAP-only preview), so avoid
    # blocking on perturbation start/end state validation when it is skipped.
    if not args.skip_isp:
        _validate_perturbation_states(pipeline, resolved["input_dir"])
    stage_dir = run_dir / "stage_configs"
    tokenize_cfg_path = stage_dir / "tokenize.yaml"
    finetune_cfg_path = stage_dir / "finetune.yaml"
    isp_cfg_path = stage_dir / "isp.yaml"

    tokenize_tpl = load_yaml(ROOT / "config" / "tokenize.yaml")
    finetune_tpl = load_yaml(ROOT / "config" / "finetune.yaml")
    isp_tpl = load_yaml(ROOT / "config" / "isp.yaml")

    tokenize_cfg = build_tokenize_config(pipeline, resolved, tokenize_tpl)
    finetune_cfg = build_finetune_config(pipeline, resolved, finetune_tpl)

    write_yaml(tokenize_cfg_path, tokenize_cfg)
    write_yaml(finetune_cfg_path, finetune_cfg)

    log_path = run_dir / "pipeline_run.log"
    install_rotating_stdio_tee(log_path, env_prefix="PIPELINE")

    stage_paths = {
        "tokenize": tokenize_cfg_path,
        "finetune": finetune_cfg_path,
        "isp": isp_cfg_path,
    }
    print(
        format_pipeline_banner(pipeline_path, resolved, stage_paths),
        end="",
    )
    print(f"  pipeline log: {log_path}")

    write_yaml(run_dir / "pipeline_resolved_paths.yaml", resolved)

    cli_overrides = {
        flag: value
        for flag, value in (
            ("--ortholog-curated-overlay", args.ortholog_curated_overlay),
            ("--ortholog-audit", args.ortholog_audit),
            ("--ortholog-approval-record", args.ortholog_approval_record),
        )
        if value
    }
    write_service_provenance(
        run_root=run_dir,
        service="pipeline",
        config_path=pipeline_path,
        extra_meta={
            "service": "pipeline",
            "resolved_paths": resolved,
            "stage_configs": {k: str(v) for k, v in stage_paths.items()},
            "cli_overrides": cli_overrides,
        },
        input_paths={"input_dir": resolved["input_dir"]},
        fast_fingerprint=True,
    )
    # Must follow write_service_provenance, which copies the source YAML to this name.
    write_yaml(run_dir / "pipeline_config_used.yaml", pipeline)

    pipeline_ok = False
    stopped_status: str | None = None
    finetune_model: Path | None = None
    tokenized_dir = Path(resolved["tokenized_dir"])

    try:
        if not args.skip_tokenize:
            for stale in remove_tokenized_outputs(tokenized_dir, resolved["study_name"]):
                print(f"Removed tokenize output from an earlier attempt: {stale}")
            env = os.environ.copy()
            env["TOKENIZE_CONFIG"] = str(tokenize_cfg_path)
            _run_subprocess(
                [sys.executable, str(ROOT / "execute_tokenizer_pipeline.py")],
                env,
                "Tokenize",
            )
            stop_message = _ortholog_approval_stop_message(tokenized_dir)
            if stop_message is not None:
                stopped_status = "stopped_ortholog_approval"
                print("\n" + "=" * 72)
                print("Pipeline stopped by the ortholog approval record (not an error).")
                print("=" * 72)
                print(f"  {stop_message}")
                print("  No dataset was tokenized; fine-tune and ISP were not run.")
                print(f"  Gate artifacts:    {resolved['tokenized_dir']}")
                print("=" * 72)
                return
        else:
            print("Skipping tokenize (--skip-tokenize).")

        resolved["dataset_path"] = str(
            resolve_tokenized_dataset(tokenized_dir, resolved["study_name"])
        )
        print(f"Tokenized dataset for fine-tune / ISP: {resolved['dataset_path']}")
        write_yaml(finetune_cfg_path, build_finetune_config(pipeline, resolved, finetune_tpl))
        write_yaml(run_dir / "pipeline_resolved_paths.yaml", resolved)

        if args.skip_finetune and args.skip_isp:
            pipeline_ok = True
            print("\n" + "=" * 72)
            print("Pipeline complete (tokenize only).")
            print("=" * 72)
            print(f"  Run directory:     {run_dir}")
            print(f"  Tokenized data:    {resolved['dataset_path']}")
            print("=" * 72)
            return

        if not args.skip_finetune:
            env = os.environ.copy()
            env["FINETUNE_CONFIG"] = str(finetune_cfg_path)
            _run_subprocess(
                [sys.executable, str(ROOT / "run_finetune.py"), "--config", str(finetune_cfg_path)],
                env,
                "Fine-tune",
            )
        else:
            print("Skipping fine-tune (--skip-finetune).")

        finetune_model = resolve_finetune_model_dir(Path(resolved["finetune_run_dir"]))
        resolved["finetune_model_dir"] = str(finetune_model)
        print(f"Fine-tuned model for ISP: {finetune_model}")
        write_yaml(run_dir / "pipeline_resolved_paths.yaml", resolved)

        num_classes = _num_classes_from_label_dict(finetune_model)
        isp_cfg = build_isp_config(
            pipeline,
            resolved,
            isp_tpl,
            finetune_model_dir=str(finetune_model),
            num_classes=num_classes,
        )
        write_yaml(isp_cfg_path, isp_cfg)

        if not args.skip_isp:
            nproc = os.environ.get("ISP_NUM_GPUS", "1")
            env = os.environ.copy()
            env["ISP_CONFIG"] = str(isp_cfg_path)
            _run_subprocess(
                [
                    "accelerate",
                    "launch",
                    "--num_processes",
                    str(nproc),
                    str(ROOT / "run_isp.py"),
                    "--config",
                    str(isp_cfg_path),
                    "--no-output-time-subdir",
                    "--no-output-date-subdir",
                    "--skip-isp-umap",
                ],
                env,
                "ISP",
            )
        else:
            print("Skipping ISP (--skip-isp).")

        # ------------------------------------------------------------------
        # E2E standard: one ISP-UMAP per run
        # ------------------------------------------------------------------
        # We intentionally disable auto ISP UMAP in stages.isp.umap (handled via
        # Web UI "ISP UMAP" run type), but for Pipeline (E2E) we still want a
        # quick "top hit" per run. Genome-wide runs project the TOP1 significant
        # gene (largest shift toward goal among Sig == 1); gene-list runs project
        # the listed genes perturbed together, since group stats have no per-gene
        # ranking to pick from.
        if not args.skip_isp:
            run_dir_path = Path(resolved["pipeline_run_dir"])
            ispstats_dir = run_dir_path / "ispstats_results"
            isp_umap_dir = run_dir_path / "isp_umap"
            perturbation = isp_cfg.get("perturbation") or {}
            listed_genes = [str(g) for g in (perturbation.get("genes_to_perturb") or [])]
            perturb_type = str(perturbation.get("type") or "delete").lower()
            umap_done = isp_umap_dir.is_dir() and any(isp_umap_dir.glob("umap_*.png"))
            if not umap_done and listed_genes:
                if perturb_type in ("delete", "overexpress"):
                    print(f"Running E2E ISP UMAP for the perturbed gene set: {', '.join(listed_genes)}")
                    # Without --gene, run_isp_umap.py perturbs the stage config's genes_to_perturb as one group.
                    _run_subprocess(
                        [
                            sys.executable,
                            str(ROOT / "run_isp_umap.py"),
                            "--run-dir",
                            str(run_dir_path),
                            "--pca-components",
                            "50",
                            "--umap-seed",
                            "0",
                        ],
                        os.environ.copy(),
                        "ISP UMAP (gene set)",
                    )
                else:
                    print(
                        f"E2E ISP UMAP: perturbation.type {perturb_type!r} is not supported "
                        "(delete / overexpress only); skipping."
                    )
            elif not umap_done:
                top_gene = pick_e2e_top1_significant_gene(ispstats_dir)
                if top_gene:
                    print(f"Running E2E TOP1 significant ISP UMAP for gene: {top_gene}")
                    env = os.environ.copy()
                    isp_stage = ((pipeline.get("stages") or {}).get("isp") or {})
                    post = isp_stage.get("postprocess") or {}
                    umap_cmd = [
                        sys.executable,
                        str(ROOT / "run_isp_umap.py"),
                        "--run-dir",
                        str(run_dir_path),
                        "--gene",
                        top_gene,
                        "--pca-components",
                        "50",
                        "--umap-seed",
                        "0",
                    ]
                    if post is True or (isinstance(post, dict) and post.get("enabled")):
                        umap_cmd.append("--enable-postprocess")
                        if isinstance(post, dict) and post.get("celltype_prediction") is False:
                            umap_cmd.append("--no-postprocess-celltype")
                        n_clusters = (post or {}).get("n_clusters") if isinstance(post, dict) else None
                        if n_clusters is not None:
                            umap_cmd.extend(["--postprocess-n-clusters", str(n_clusters)])
                    # Fig.2/3/4 manuscript style projection: PCA(50) → UMAP (seed 0).
                    _run_subprocess(
                        umap_cmd,
                        env,
                        "ISP UMAP (TOP1 significant)",
                    )
                else:
                    print(
                        "E2E TOP1 ISP UMAP: no significant gene found; skipping."
                    )

        pipeline_ok = True
        print("\n" + "=" * 72)
        print("Pipeline complete.")
        print("=" * 72)
        print(f"  Run directory:     {run_dir}")
        print(f"  Tokenized data:    {resolved['dataset_path']}")
        print(f"  Fine-tuned model:  {finetune_model}")
        if not args.skip_isp:
            print(f"  ISP outputs:       {resolved['pipeline_run_dir']}/isp_results/")
        ft_umap = ((pipeline.get("stages") or {}).get("finetune") or {}).get("umap") or {}
        if ft_umap.get("enabled", False):
            print(f"  Fine-tune UMAP:    {resolved['finetune_run_dir']}/figures/")
        print("=" * 72)

    finally:
        update_service_provenance(
            run_dir,
            {
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_status": stopped_status or ("completed" if pipeline_ok else "failed"),
                "finetune_model_dir": str(finetune_model) if finetune_model else None,
                "dataset_path": resolved["dataset_path"],
            },
            service="pipeline",
        )

    if not pipeline_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
