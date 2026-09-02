# Mouse Geneformer dictionaries

Place mouse token and median dictionaries here (not committed — large binaries).

Expected files (from [Mouse-Genecorpus-20M](https://huggingface.co/datasets/MPRG/Mouse-Genecorpus-20M)):

- `MLM-re_token_dictionary_v1.pkl`
- `mouse_gene_median_dictionary.pkl`
- `MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl`

Both mouse variants (`base` and `12l_e20`) share these dictionaries (`max_input_size: 2048`).

Download helpers: `bash scripts/download_mouse_geneformer.sh` (token dict + weights).

Legacy flat layout `core/geneformer/dicts/*.pkl` is still supported.
