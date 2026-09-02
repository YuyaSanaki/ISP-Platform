#!/usr/bin/env python3
"""Build tiny mouse + human + fly smoke datasets (10x MTX + loom) for matrix testing.

Mouse: subsample data/1w AD/WT 10x matrices.
Human: remap mouse Ensembl IDs → human orthologs (synthetic cross-species smoke).
Fly: remap mouse Ensembl IDs → Drosophila FBgn via inverted drosophila_to_mouse.
Also materializes loom copies so tokenizer can skip 10x→loom when desired.
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import sys
from pathlib import Path

import numpy as np
import scanpy as sc

ROOT = Path(__file__).resolve().parent.parent


def _load_ortholog_map(tsv: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with tsv.open() as fh:
        header = fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[0] or not parts[1]:
                continue
            mapping[parts[0].split(".")[0]] = parts[1].split(".")[0]
    return mapping


def _write_10x_mtx(adata, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # barcodes
    with gzip.open(out_dir / "barcodes.tsv.gz", "wt") as fh:
        for bc in adata.obs_names.astype(str):
            fh.write(f"{bc}\n")
    # features: ensembl_id \t gene_symbol \t Gene Expression
    symbols = (
        adata.var["gene_symbols"].astype(str)
        if "gene_symbols" in adata.var.columns
        else adata.var_names.astype(str)
    )
    ens = adata.var["ensembl_id"].astype(str)
    with gzip.open(out_dir / "features.tsv.gz", "wt") as fh:
        for eid, sym in zip(ens, symbols):
            fh.write(f"{eid}\t{sym}\tGene Expression\n")
    # matrix Market
    x = adata.X
    if hasattr(x, "tocoo"):
        coo = x.tocoo()
    else:
        from scipy import sparse

        coo = sparse.coo_matrix(x)
    n_genes, n_cells = adata.n_vars, adata.n_obs
    # 10x MTX is genes × cells
    with gzip.open(out_dir / "matrix.mtx.gz", "wt") as fh:
        fh.write("%%MatrixMarket matrix coordinate real general\n")
        fh.write(f"{n_genes} {n_cells} {coo.nnz}\n")
        # scanpy/anndata X is cells × genes; Market for 10x is genes × cells
        for r, c, v in zip(coo.col, coo.row, coo.data):
            fh.write(f"{int(r) + 1} {int(c) + 1} {float(v)}\n")


def _write_loom_safe(adata, path: Path) -> None:
    sys.path.insert(0, str(ROOT / "core"))
    from loom_write import write_loom_safe

    write_loom_safe(adata, path)


def _subsample_sample(
    mtx_dir: Path,
    *,
    n_cells: int,
    seed: int,
    disease: str,
    sample_id: str,
) -> "sc.AnnData":
    adata = sc.read_10x_mtx(str(mtx_dir), var_names="gene_ids", make_unique=True)
    rng = np.random.default_rng(seed)
    n = min(n_cells, adata.n_obs)
    idx = np.sort(rng.choice(adata.n_obs, size=n, replace=False))
    adata = adata[idx].copy()
    adata.var["ensembl_id"] = adata.var_names.astype(str).str.split(".").str[0]
    if "gene_symbols" not in adata.var.columns:
        # read_10x_mtx with var_names=gene_ids keeps symbols in gene_symbols when present
        adata.var["gene_symbols"] = adata.var_names.astype(str)
    adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
    adata.obs["disease"] = disease
    adata.obs["sample_id"] = sample_id
    adata.obs["time"] = "1w"
    adata.obs["genotype"] = disease
    adata.obs["replicate"] = "1st"
    return adata


def _remap_genes(adata, id_map: dict[str, str]) -> "sc.AnnData":
    """Map ensembl_id via id_map; keep first occurrence of each target ID."""
    src = adata.var["ensembl_id"].astype(str).to_numpy()
    mapped = np.array([id_map.get(g, "") for g in src], dtype=object)
    keep = mapped != ""
    seen: set[str] = set()
    uniq = []
    for i, tid in enumerate(mapped):
        if not keep[i]:
            uniq.append(False)
            continue
        if tid in seen:
            uniq.append(False)
            continue
        seen.add(tid)
        uniq.append(True)
    mask = np.asarray(uniq, dtype=bool)
    out = adata[:, mask].copy()
    out.var["ensembl_id"] = mapped[mask]
    out.var_names = out.var["ensembl_id"].astype(str)
    out.var_names_make_unique()
    return out


def _invert_ortholog_map(tsv: Path) -> dict[str, str]:
    """Invert source→target TSV to target→source (first wins)."""
    forward = _load_ortholog_map(tsv)
    inverted: dict[str, str] = {}
    for src, tgt in forward.items():
        inverted.setdefault(tgt, src)
    return inverted


def build_smoke(*, n_cells: int, seed: int, force: bool) -> None:
    mouse_root = ROOT / "data" / "smoke_mouse"
    human_root = ROOT / "data" / "smoke_human"
    fly_root = ROOT / "data" / "smoke_fly"
    ortholog = ROOT / "core" / "geneformer" / "dicts" / "orthologs" / "mouse_to_human.tsv"
    fly_ortho = ROOT / "core" / "geneformer" / "dicts" / "orthologs" / "drosophila_to_mouse.tsv"
    mouse_to_human = _load_ortholog_map(ortholog)
    mouse_to_fly = _invert_ortholog_map(fly_ortho)
    print(f"Loaded {len(mouse_to_human)} mouse→human ortholog pairs")
    print(f"Loaded {len(mouse_to_fly)} mouse→fly ortholog pairs (inverted)")

    samples = [
        ("1w-AD-1st", "AD", ROOT / "data" / "1w" / "1w-AD-1st" / "filtered_feature_bc_matrix"),
        ("1w-WT-1st", "WT", ROOT / "data" / "1w" / "1w-WT-1st" / "filtered_feature_bc_matrix"),
    ]

    for i, (name, disease, mtx) in enumerate(samples):
        if not mtx.is_dir():
            raise FileNotFoundError(mtx)
        print(f"Subsampling {name} ({n_cells} cells)...")
        adata = _subsample_sample(
            mtx, n_cells=n_cells, seed=seed + i, disease=disease, sample_id=name
        )

        mouse_mtx = mouse_root / name / "filtered_feature_bc_matrix"
        mouse_loom = mouse_root / "loom" / f"{name}.loom"
        if force or not (mouse_mtx / "matrix.mtx.gz").is_file():
            _write_10x_mtx(adata, mouse_mtx)
        if force or not mouse_loom.is_file():
            _write_loom_safe(adata, mouse_loom)
        print(f"  mouse: {adata.n_obs} cells × {adata.n_vars} genes → {mouse_mtx}")

        hadata = _remap_genes(adata, mouse_to_human)
        # Keep AD/WT sample names for extract_metadata_from_path
        human_mtx = human_root / name / "filtered_feature_bc_matrix"
        human_loom = human_root / "loom" / f"{name}.loom"
        if force or not (human_mtx / "matrix.mtx.gz").is_file():
            _write_10x_mtx(hadata, human_mtx)
        if force or not human_loom.is_file():
            _write_loom_safe(hadata, human_loom)
        print(f"  human: {hadata.n_obs} cells × {hadata.n_vars} genes → {human_mtx}")

        fadata = _remap_genes(adata, mouse_to_fly)
        fly_mtx = fly_root / name / "filtered_feature_bc_matrix"
        fly_loom = fly_root / "loom" / f"{name}.loom"
        if force or not (fly_mtx / "matrix.mtx.gz").is_file():
            _write_10x_mtx(fadata, fly_mtx)
        if force or not fly_loom.is_file():
            _write_loom_safe(fadata, fly_loom)
        print(f"  fly:   {fadata.n_obs} cells × {fadata.n_vars} genes → {fly_mtx}")

    print("Done.")
    print(f"  {mouse_root}")
    print(f"  {human_root}")
    print(f"  {fly_root}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-cells", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    build_smoke(n_cells=args.n_cells, seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()
