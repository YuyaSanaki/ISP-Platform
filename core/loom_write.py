"""Write AnnData to Geneformer-compatible .loom (anndata + loompy fallback)."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


def _loompy_create(adata, loom_out: str) -> None:
    import loompy

    mat = adata.X.T
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    row_attrs = {"ensembl_id": np.asarray(adata.var["ensembl_id"].astype(str))}
    col_attrs = {k: np.asarray(adata.obs[k]) for k in adata.obs.columns}
    if "cell_id" not in col_attrs:
        col_attrs["cell_id"] = np.asarray(adata.obs_names.astype(str))
    loompy.create(loom_out, mat, row_attrs=row_attrs, col_attrs=col_attrs)


def _prepare_var_for_loom(adata) -> None:
    """Avoid anndata.write_loom failures on None gene_symbols (newer anndata)."""
    if "gene_symbols" in adata.var.columns:
        adata.var["gene_symbols"] = (
            adata.var["gene_symbols"].fillna("").astype(str)
        )


def write_loom_safe(adata, path: Union[str, Path]) -> None:
    """Write .loom for Geneformer tokenization; fall back to loompy.create if needed."""
    loom_out = str(path)
    Path(loom_out).parent.mkdir(parents=True, exist_ok=True)
    _prepare_var_for_loom(adata)
    try:
        adata.write_loom(loom_out)
        return
    except Exception as loom_exc:
        print(
            f"  write_loom failed ({type(loom_exc).__name__}: {loom_exc}); "
            "falling back to loompy.create"
        )
    _loompy_create(adata, loom_out)
