#!/usr/bin/env python3
"""Re-apply pre-ISP cell-type labels onto an existing ISP UMAP run CSV.

Expects ``cell_type`` / ``celltype_annotator`` already on ``per_cell_isp_shift.csv``
(from a dataset tokenized with ``tokenizer.celltype_annotation``). Brain token
marker scoring is not available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

_CORE = Path(__file__).resolve().parent
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

from isp_umap_celltype import annotate_dataframe_with_cell_types  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        required=False,
        help="Optional ISP UMAP YAML (reads postprocess.prefer_metadata_celltype)",
    )
    parser.add_argument(
        "--refresh-overlays",
        action="store_true",
        help="Also regenerate joint overlays + L2-by-group + trajectory plots",
    )
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else Path.cwd() / args.run_dir
    prefer = "auto"
    if args.config is not None:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        post = (cfg or {}).get("postprocess") or {}
        prefer = post.get("prefer_metadata_celltype", "auto")

    per_cell_path = run_dir / "per_cell_isp_shift.csv"
    if not per_cell_path.exists():
        raise FileNotFoundError(per_cell_path)
    df = pd.read_csv(per_cell_path)
    annotated = annotate_dataframe_with_cell_types(df, prefer_metadata=prefer)
    annotated.to_csv(per_cell_path, index=False)
    print(f"Updated {per_cell_path} (prefer_metadata={prefer})")

    cluster_csv = run_dir / "cluster_coexpr_analysis" / "per_cell_cluster_l2_celltype.csv"
    if cluster_csv.exists():
        # Keep cluster columns; refresh cell-type fields from annotated table.
        cl = pd.read_csv(cluster_csv)
        for col in (
            "pred_cell_type",
            "pred_score",
            "coarse_type",
            "celltype_source",
            "celltype_plot",
            "cell_type",
            "celltype_annotator",
            "tissue",
            "celltype_score",
        ):
            if col in annotated.columns:
                cl[col] = annotated[col].values if len(cl) == len(annotated) else cl.get(col)
        if len(cl) == len(annotated):
            for col in annotated.columns:
                if col not in cl.columns and col.startswith(("pred_", "coarse_", "celltype")):
                    cl[col] = annotated[col].values
            cl.to_csv(cluster_csv, index=False)
            print(f"Updated {cluster_csv}")

    if args.refresh_overlays:
        from run_isp_umap import run_downstream_plots

        cfg = {"postprocess": {"enabled": True, "prefer_metadata_celltype": prefer}}
        if args.config is not None:
            cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        run_downstream_plots(run_dir, "Igfbp2", cfg)


if __name__ == "__main__":
    main()
