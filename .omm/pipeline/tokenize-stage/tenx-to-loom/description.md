### English

`process_single_cell_to_loom`, per sample: `sc.read_10x_mtx(var_names="gene_ids")`, strip Ensembl version, set `n_counts`,
parse folder name `Time-State-Suffix` into `time`, `disease`, `genotype` (both from the 2nd field), `replicate`,
`sample_id`, call the cell-type annotator, then `write_loom_safe` (falls back to `loompy.create`).
`input_type: loom` skips conversion and uses the folder's loom files directly.

### 日本語

`process_single_cell_to_loom` はサンプルごとに、`sc.read_10x_mtx(var_names="gene_ids")` で読み、Ensembl のバージョン番号を外し、`n_counts` を設定します。
フォルダ名 `Time-State-Suffix` を `time`、`disease`、`genotype`（どちらも2番目の欄）、`replicate`、`sample_id` に分解し、細胞種の付与を呼んでから
`write_loom_safe`（失敗時は `loompy.create`）で書き出します。`input_type: loom` の場合は変換せず、そのフォルダの loom をそのまま使います。
