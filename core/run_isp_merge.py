"""Combine parallel ISP chunks and write statistics once.

Each chunk process (``run_isp.py --chunk-id``) writes pickle files and a done
marker, and does not compute statistics. This script checks that every chunk
finished from the same configuration and the same centroids, then runs the
usual statistics, figures and postprocess steps.

Chunk tables are not concatenated. The null pool and FDR are computed on the
combined per-cell shifts.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import postprocess_switch as pps
from geneformer.in_silico_perturber import check_perturb_type_implemented
from geneformer.species_context import backend_from_config, species_from_config
from isp_parallel import ChunkMergeError, chunk_config_sha256, validate_parallel_run, yaml_chunk_pair
from run_isp import (
    load_isp_config,
    resolve_forward_batch_size_setting,
    resolve_shared_run_dir,
    run_stats_and_downstream,
)


def _chunk_count(cli_count: int | None, cfg: dict) -> int:
    if cli_count is not None:
        count = int(cli_count)
    else:
        _, yaml_count = yaml_chunk_pair(cfg)
        if yaml_count is None or (isinstance(yaml_count, str) and not yaml_count.strip()):
            raise ChunkMergeError(
                "Pass --chunk-count (or set isp.chunk.count). "
                "Counting marker files would hide a chunk that never finished."
            )
        count = int(yaml_count)
    if count < 2:
        raise ChunkMergeError("chunk count must be at least 2.")
    return count


def main() -> None:
    default_cfg = os.environ.get("ISP_CONFIG", "/app/core/config/isp.yaml")
    parser = argparse.ArgumentParser(description="Merge parallel ISP chunks and write stats once.")
    parser.add_argument("--config", type=Path, default=Path(default_cfg))
    parser.add_argument("--chunk-count", type=int, default=None)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Folder that contains isp_results/ and isp_chunks/. "
        "Default: the same output_root/date/output-subdir the chunks used.",
    )
    parser.add_argument("--output-date", default=None, metavar="YYYYMMDD")
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--no-output-date-subdir", action="store_true")
    parser.add_argument("--forward-batch-size", default=None, metavar="N|auto")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--skip-isp-umap", action="store_true")
    pps.add_postprocess_args(parser)
    args = parser.parse_args()

    cfg = pps.apply_postprocess_args(load_isp_config(args.config), args)
    species_from_config(cfg)
    backend = backend_from_config(cfg)
    forward_batch_size, _source = resolve_forward_batch_size_setting(
        cfg, args.forward_batch_size, backend
    )
    count = _chunk_count(args.chunk_count, cfg)
    if args.run_dir is not None:
        run_root = args.run_dir
    else:
        run_root = resolve_shared_run_dir(
            cfg,
            output_date=args.output_date,
            output_subdir=args.output_subdir,
            no_output_date_subdir=args.no_output_date_subdir,
        )
    isp_dir = run_root / "isp_results"
    stats_out = run_root / "ispstats_results"
    markers = validate_parallel_run(isp_dir, run_root / "isp_chunks", count)
    expected = chunk_config_sha256(cfg, forward_batch_size)
    if any(marker.get("config_sha256") != expected for marker in markers):
        raise ChunkMergeError(
            "This config does not match the chunk markers. "
            "Use the same YAML and the same forward batch size the chunks used."
        )

    pert = cfg.get("perturbation") or {}
    select_perturb_type = pert.get("type", "delete")
    check_perturb_type_implemented(select_perturb_type)
    start_state = pert.get("start_state")
    end_state = pert.get("end_state")
    if start_state is None or end_state is None:
        raise ChunkMergeError("config perturbation.start_state and perturbation.end_state are required.")
    output_prefix = markers[0]["output_prefix"]
    isp_cfg = cfg.get("isp") or {}
    print(f"Merging {count} chunks from {run_root}", flush=True)
    run_stats_and_downstream(
        cfg,
        stats_mode=(cfg.get("stats") or {}).get("mode", "goal_state_shift"),
        genes_to_perturb_list=list(pert.get("genes_to_perturb") or []),
        combos=int(isp_cfg.get("combos", 0) or 0),
        anchor_gene=isp_cfg.get("anchor_gene"),
        state_key=pert.get("state_key", "state"),
        start_state=start_state,
        end_state=end_state,
        alt_state=list(pert.get("alt_states") or []),
        backend=backend,
        isp_dir=str(isp_dir) + os.sep,
        stats_out=str(stats_out),
        output_prefix=output_prefix,
        run_root=run_root,
        perturb_gene_labels={},
        skip_analysis=args.skip_analysis,
        skip_isp_umap=args.skip_isp_umap,
    )


if __name__ == "__main__":
    try:
        main()
    except ChunkMergeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
