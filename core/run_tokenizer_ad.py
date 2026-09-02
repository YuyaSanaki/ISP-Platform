import os
import sys
# Add current directory to path so geneformer can be imported
sys.path.append(os.getcwd())

from geneformer import TranscriptomeTokenizer

def main():
    # Input loom files directory (created by convert_to_loom.py)
    loom_dir = "data/AD/loom_files/"
    # Output tokenized dataset directory
    output_dir = "data/AD/tokenized_dataset/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Custom attributes to pass through from .loom to the tokenized dataset
    # Format: { loom_attribute_name: dataset_column_name }
    custom_attr_dict = {
        "time": "time",
        "genotype": "genotype",
        "replicate": "replicate",
        "disease": "disease",
        "sample_id": "sample_id"
    }
    
    print("Initializing Tokenizer...")
    # nproc should be adjusted based on available CPU cores
    tk = TranscriptomeTokenizer(custom_attr_name_dict=custom_attr_dict, nproc=16)
    
    print(f"Starting Tokenization of files in {loom_dir}...")
    # This will search for all *.loom files in loom_dir
    tk.tokenize_data(
        data_directory=loom_dir,
        output_directory=output_dir,
        output_prefix="AD_data"
    )
    print(f"Tokenization finished. Output saved to {output_dir}")

if __name__ == "__main__":
    main()
