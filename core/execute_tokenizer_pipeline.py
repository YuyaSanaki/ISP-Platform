import os
import shutil
import subprocess
import sys
import argparse
import yaml
import scanpy as sc
import anndata as ad
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add current directory to path so geneformer and provenance can be imported
sys.path.append(os.getcwd())
from data_input_layout import (
    diagnose_input_dir,
    discover_sample_dirs,
    parse_sample_folder_name,
    resolve_single_cell_input_dir,
    sample_label,
    tenx_matrix_directory,
)
from geneformer import TranscriptomeTokenizer
from geneformer.conversion_report import resolve_report_conversion, run_conversion_report
from geneformer.gene_converter import normalize_gene_id
from geneformer.ortholog_loss_gate import (
    OrthologApprovalError,
    OrthologLossGateError,
    resolve_enable_ortholog_loss_gate,
    resolve_ortholog_audit_config,
    run_ortholog_loss_gate,
)
from geneformer.species_context import log_species_banner, species_from_config
from loom_write import write_loom_safe
from run_pipeline_log import format_tokenize_run_banner, install_rotating_stdio_tee
from run_provenance import write_service_provenance, update_service_provenance


def process_single_cell_to_loom(input_dir, loom_temp_dir, settings, tokenizer_cfg, species=None) -> int:
    """Convert subdirectories of (barcodes/features/matrix) to .loom files. Returns loom count."""
    os.makedirs(loom_temp_dir, exist_ok=True)
    input_path = Path(input_dir).resolve()
    loom_path = Path(loom_temp_dir).resolve()
    data_root = resolve_single_cell_input_dir(input_path, loom_path)
    if data_root != input_path:
        print(f"Resolved single-cell study root: {data_root} (from {input_path})")

    sample_dirs = discover_sample_dirs(data_root, exclude=loom_path)
    if sample_dirs:
        print(f"Found {len(sample_dirs)} sample(s): {', '.join(p.name for p in sample_dirs)}")

    if not sample_dirs:
        print(f"No 10x sample directories found under {data_root}")
        print(diagnose_input_dir(input_path, loom_path))
        return 0

    from celltype_annotate_expression import apply_pre_isp_celltype_annotation

    converted = 0
    for sample_dir in sample_dirs:
        sample_path = Path(sample_dir)
        folder_name = sample_path.name
        loom_stem = sample_label(sample_path, data_root)
        mtx_path_obj = tenx_matrix_directory(sample_path)
        if mtx_path_obj is None:
            print(f"Skipping {folder_name}: no 10x matrix files found.")
            continue
        mtx_path = str(mtx_path_obj)

        print(f"Converting {folder_name} → {loom_stem}.loom (from {mtx_path})...")
        try:
            # Read mtx and set Ensembl IDs
            adata = sc.read_10x_mtx(mtx_path, var_names="gene_ids", make_unique=True)
            # Strip version numbers from Ensembl IDs (e.g., ENSMUSG00000102693.2 -> ENSMUSG00000102693)
            adata.var["ensembl_id"] = [
                normalize_gene_id(x) for x in adata.var_names.astype(str)
            ]
            adata.obs["n_counts"] = (
                adata.X.sum(axis=1).A1 if hasattr(adata.X, "sum") else adata.X.sum(axis=1)
            )

            if settings.get("extract_metadata_from_path"):
                meta = parse_sample_folder_name(folder_name)
                adata.obs["time"] = meta["time"]
                adata.obs["genotype"] = meta["genotype"]
                adata.obs["disease"] = meta["disease"]
                adata.obs["replicate"] = meta["replicate"]
                adata.obs["sample_id"] = meta["sample_id"]
            elif tokenizer_cfg.get("custom_attr_name_dict"):
                adata.obs["sample_id"] = folder_name

            apply_pre_isp_celltype_annotation(adata, tokenizer_cfg, species=species)

            if tokenizer_cfg.get("custom_attr_name_dict"):
                for attr_name in tokenizer_cfg["custom_attr_name_dict"].keys():
                    if attr_name not in adata.obs.columns:
                        adata.obs[attr_name] = ""

            if "sample_id" not in adata.obs.columns:
                adata.obs["sample_id"] = folder_name
            loom_out = os.path.join(loom_temp_dir, f"{loom_stem}.loom")
            write_loom_safe(adata, loom_out)
            converted += 1
        except Exception as e:
            print(f"Error converting {folder_name}: {e}")

    print(f"Converted {converted} sample(s) to .loom under {loom_temp_dir}")
    return converted


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize scRNA-seq data for Geneformer.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--report-conversion",
        action="store_true",
        default=None,
        help="Write pre-tokenize ortholog conversion summary (JSON/TSV) to output_dir.",
    )
    group.add_argument(
        "--no-report-conversion",
        action="store_false",
        dest="report_conversion",
        help="Skip ortholog conversion report.",
    )
    parser.add_argument(
        "--ortholog-curated-overlay",
        type=str,
        default=None,
        help=(
            "Optional project curated ortholog overlay: a TSV file or a directory "
            "containing curated_bridge_{pair}.tsv. Overrides YAML when set; "
            "also sets GENEFORMER_ORTHOLOG_CURATED_OVERLAY."
        ),
    )
    gate = parser.add_mutually_exclusive_group()
    gate.add_argument(
        "--ortholog-loss-gate",
        action="store_true",
        default=None,
        help="Run ortholog_loss_gate after conversion report (requires ortholog_audit).",
    )
    gate.add_argument(
        "--no-ortholog-loss-gate",
        action="store_false",
        dest="ortholog_loss_gate",
        help="Skip ortholog_loss_gate even when ortholog_audit is configured.",
    )
    parser.add_argument(
        "--ortholog-audit",
        type=str,
        default=None,
        help="Path to analysis_manifest.yaml (or ortholog_audit YAML) for the loss gate.",
    )
    parser.add_argument(
        "--ortholog-approval-record",
        type=str,
        default=None,
        help=(
            "NEW approved YAML (status: approved) matching a prior Block request_id / "
            "blocked_run hashes. Do not pass ortholog_approval_request.yaml "
            "(status: pending). Decisions: stop|curated_bridge|sensitivity_best_of_n|"
            "mark_non_critical (aliases A|B|C|D)."
        ),
    )
    return parser.parse_args()


def main():
    cli = parse_args()
    config_path = Path(os.getenv("TOKENIZE_CONFIG", "/app/core/config/tokenize.yaml")).expanduser()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if cli.ortholog_curated_overlay:
        os.environ["GENEFORMER_ORTHOLOG_CURATED_OVERLAY"] = str(
            Path(cli.ortholog_curated_overlay).expanduser()
        )
        config.setdefault("species", {})
        config["species"]["ortholog_curated_overlay"] = os.environ[
            "GENEFORMER_ORTHOLOG_CURATED_OVERLAY"
        ]
    elif (config.get("species") or {}).get("ortholog_curated_overlay"):
        os.environ.setdefault(
            "GENEFORMER_ORTHOLOG_CURATED_OVERLAY",
            str(Path(config["species"]["ortholog_curated_overlay"]).expanduser()),
        )

    data_cfg = config['data']
    tokenizer_cfg = config['tokenizer']
    tokenizer_nproc = int(tokenizer_cfg.get('nproc', 1))
    single_cell_settings = config.get('single_cell_settings', {}) or {}

    out_dir = Path(data_cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "tokenize_run.log"
    install_rotating_stdio_tee(log_path, env_prefix="TOKENIZE")
    print(
        format_tokenize_run_banner(
            config_path.resolve(),
            data_cfg,
            tokenizer_cfg,
            single_cell_settings,
        ),
        end="",
    )
    print(f"  run log (rotating): {log_path}")

    species = species_from_config(config)
    log_species_banner(species, prefix="  ")

    report_conversion = resolve_report_conversion(
        species,
        tokenizer_cfg,
        cli_flag=cli.report_conversion,
    )
    if report_conversion:
        print("  report_conversion:     enabled")
    else:
        print("  report_conversion:     disabled")

    if cli.ortholog_audit:
        tokenizer_cfg = dict(tokenizer_cfg)
        tokenizer_cfg["ortholog_audit"] = cli.ortholog_audit
        config["tokenizer"] = tokenizer_cfg
    approval_record = cli.ortholog_approval_record or tokenizer_cfg.get(
        "ortholog_approval_record"
    )
    audit_path = cli.ortholog_audit or tokenizer_cfg.get("ortholog_audit")
    audit_cfg = resolve_ortholog_audit_config(
        tokenizer_cfg, cli_path=cli.ortholog_audit
    )
    enable_gate = resolve_enable_ortholog_loss_gate(
        species,
        tokenizer_cfg,
        audit_cfg=audit_cfg,
        cli_flag=cli.ortholog_loss_gate,
    )
    if enable_gate:
        print("  ortholog_loss_gate:    enabled")
    else:
        print("  ortholog_loss_gate:    disabled")

    # Step 1: Handle Conversion if needed
    if data_cfg["input_type"] == "single-cell":
        print("Input type is single-cell. Converting to loom first...")
        study_root = resolve_single_cell_input_dir(
            data_cfg["input_dir"], data_cfg.get("loom_temp_dir")
        )
        if str(study_root) != str(Path(data_cfg["input_dir"]).resolve()):
            print(
                f"Note: data.input_dir was {data_cfg['input_dir']}; "
                f"using study root {study_root} (ExperimentName / Time-Condition-Replicate layout)."
            )
        data_cfg["input_dir"] = str(study_root)
        n_converted = process_single_cell_to_loom(
            data_cfg["input_dir"],
            data_cfg['loom_temp_dir'],
            single_cell_settings,
            tokenizer_cfg,
            species=species,
        )
        loom_dir = Path(data_cfg['loom_temp_dir'])
        n_looms = len(list(loom_dir.glob("*.loom")))
        if n_looms == 0:
            print(diagnose_input_dir(data_cfg['input_dir'], data_cfg['loom_temp_dir']))
            raise SystemExit(
                "No .loom files produced. Fix data.input_dir layout (see message above) and retry."
            )
        tokenizer_input_dir = str(loom_dir) + ("" if str(loom_dir).endswith(os.sep) else os.sep)
        file_format = "loom"
    else:
        print("Input type is loom. Skipping conversion.")
        tokenizer_input_dir = data_cfg['input_dir']
        file_format = "loom"

    # Step 2: Provenance Tracking & Fingerprinting
    input_paths = {
        "input_data": data_cfg['input_dir']
    }
    
    # Read provenance settings from config with sensible defaults
    prov_cfg = config.get('provenance') or {}
    enable_fp = prov_cfg.get('enable_input_fingerprint', False)
    # Environment variable still takes precedence if set
    if os.environ.get("ENABLE_INPUT_FINGERPRINT"):
        enable_fp = os.environ.get("ENABLE_INPUT_FINGERPRINT").lower() in ("1", "true", "yes")
    
    is_fast = prov_cfg.get('fingerprint_fast', True)
    if os.environ.get("FINGERPRINT_FAST"):
        is_fast = os.environ.get("FINGERPRINT_FAST").lower() in ("1", "true", "yes")
    
    # Temporarily set environment variable for run_provenance utility if enabled by config
    if enable_fp:
        os.environ["ENABLE_INPUT_FINGERPRINT"] = "true"

    write_service_provenance(
        run_root=out_dir,
        service="tokenize",
        config_path=config_path,
        extra_meta={
            "service": "tokenize",
            "tokenizer": {"nproc": tokenizer_nproc},
            "data": {
                "input_type": data_cfg.get("input_type"),
                "input_dir": data_cfg.get("input_dir"),
                "loom_temp_dir": data_cfg.get("loom_temp_dir"),
                "output_dir": data_cfg.get("output_dir"),
                "output_prefix": data_cfg.get("output_prefix"),
            },
        },
        input_paths=input_paths,
        fast_fingerprint=is_fast
    )
    print(f"  Provenance saved in: {out_dir}")

    if report_conversion:
        run_conversion_report(
            tokenizer_input_dir,
            out_dir,
            species,
            file_format=file_format,
        )

    gate_result = None
    if enable_gate:
        if not audit_cfg:
            raise SystemExit(
                "ortholog_loss_gate enabled but no ortholog_audit config found. "
                "Pass --ortholog-audit PATH or set tokenizer.ortholog_audit."
            )
        try:
            gate_result = run_ortholog_loss_gate(
                tokenizer_input_dir,
                out_dir,
                species,
                audit_cfg,
                file_format=file_format,
                approval_record=approval_record,
                audit_path=audit_path,
            )
        except OrthologLossGateError as exc:
            update_service_provenance(
                out_dir,
                {
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "run_status": "blocked_ortholog_loss_gate",
                    "ortholog_loss_gate": exc.result.to_summary_dict(),
                },
                service="tokenize",
            )
            raise SystemExit(2) from exc
        except OrthologApprovalError as exc:
            update_service_provenance(
                out_dir,
                {
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "run_status": "rejected_ortholog_approval",
                    "ortholog_approval_error": str(exc),
                },
                service="tokenize",
            )
            raise SystemExit(2) from exc
        if gate_result is not None and not gate_result.continue_tokenize:
            update_service_provenance(
                out_dir,
                {
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "run_status": "stopped_ortholog_approval",
                    "ortholog_loss_gate": gate_result.to_summary_dict(),
                },
                service="tokenize",
            )
            print("Pipeline stopped after ortholog approval (tokenize skipped).")
            return

    # Step 3: Tokenize
    print(f"Initializing Tokenizer with nproc={tokenizer_nproc}...")
    
    tk = TranscriptomeTokenizer(
        custom_attr_name_dict=tokenizer_cfg.get('custom_attr_name_dict'), 
        nproc=tokenizer_nproc,
        max_cells=int(tokenizer_cfg.get('max_cells', 300_000)),
        species_config=species,
    )
    
    print(f"Starting Tokenization of files in {tokenizer_input_dir}...")
    pipeline_ok = False
    try:
        tk.tokenize_data(
            data_directory=tokenizer_input_dir,
            output_directory=data_cfg['output_dir'],
            output_prefix=data_cfg['output_prefix'],
            file_format=file_format
        )
        pipeline_ok = True
        print("Pipeline Finished.")
    finally:
        extra = {
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_status": "completed" if pipeline_ok else "failed",
        }
        if gate_result is not None:
            extra["ortholog_loss_gate"] = gate_result.to_summary_dict()
        update_service_provenance(
            out_dir,
            extra,
            service="tokenize",
        )

if __name__ == "__main__":
    main()
