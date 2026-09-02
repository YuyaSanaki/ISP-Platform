import scanpy as sc
import anndata as ad
import pandas as pd
import os
import glob
import sys
from pathlib import Path

# Allow `python core/convert_to_loom.py` from repo root and container PYTHONPATH.
_CORE = Path(__file__).resolve().parent
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

from geneformer.gene_converter import normalize_gene_id


def ensembl_ids_without_version(var_names) -> list[str]:
    """Match execute_tokenizer_pipeline: ENSMUSG....1 → ENSMUSG...."""
    return [normalize_gene_id(x) for x in var_names.astype(str)]


def process_sample(sample_dir, output_dir):
    sample_name = os.path.basename(sample_dir.rstrip('/'))
    print(f"Processing {sample_name}...")
    
    # 10x files are in filtered_feature_bc_matrix/
    mtx_path = os.path.join(sample_dir, "filtered_feature_bc_matrix")
    if not os.path.exists(mtx_path):
        # Fallback: check if the files are directly in sample_dir
        if not os.path.exists(os.path.join(sample_dir, "matrix.mtx.gz")):
            print(f"Skipping {sample_name}, mtx path not found.")
            return
        mtx_path = sample_dir

    try:
        # Read 10x data
        # Geneformer needs Ensembl IDs (var_names='gene_ids' uses the first column of features.tsv)
        adata = sc.read_10x_mtx(mtx_path, var_names='gene_ids', make_unique=True)
        
        # Explicitly set ensembl_id (row attribute) and n_counts (column attribute) for Geneformer.
        # PIPseq features are often versioned (ENSMUSG....1); token/median dicts are not.
        adata.var['ensembl_id'] = ensembl_ids_without_version(adata.var_names)
        
        # Calculate total counts per cell (n_counts)
        # Using .A1 to ensure it's a 1D array
        if hasattr(adata.X, "sum"):
            adata.obs['n_counts'] = adata.X.sum(axis=1).A1
        else:
            # In case X is not sparse or has different API
            adata.obs['n_counts'] = adata.X.sum(axis=1)

        # Extract metadata from sample name (e.g., 1w-AD-1st)
        parts = sample_name.split('-')
        if len(parts) >= 3:
            adata.obs['time'] = parts[0]
            adata.obs['genotype'] = parts[1]
            adata.obs['replicate'] = parts[2]
            # Map genotype to disease (common key in ISP)
            adata.obs['disease'] = parts[1]
        else:
            adata.obs['disease'] = sample_name
            
        adata.obs['sample_id'] = sample_name
        
        # Geneformer's tokenizer expects raw counts.
        # Ensure the data is not log-normalized.
        
        # Export to loom
        loom_path = os.path.join(output_dir, f"{sample_name}.loom")
        
        # loompy needs the data to be in float32 or int32 for many operations
        # and write_loom handles the conversion
        adata.write_loom(loom_path)
        print(f"Saved to {loom_path}")
    except Exception as e:
        print(f"Error processing {sample_name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    data_dir = "data/AD"
    output_dir = os.path.join(data_dir, "loom_files")
    os.makedirs(output_dir, exist_ok=True)
    
    # List all subdirectories in data/AD
    # Filter for directories that don't start with "." and are not "loom_files"
    sample_dirs = sorted([os.path.join(data_dir, d) for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d)) 
                  and not d.startswith(".") 
                  and d != "loom_files"])
    
    if not sample_dirs:
        print(f"No sample directories found in {data_dir}")
        return
        
    for sample_dir in sample_dirs:
        process_sample(sample_dir, output_dir)

if __name__ == "__main__":
    main()
