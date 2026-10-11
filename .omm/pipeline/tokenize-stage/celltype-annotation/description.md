### English

`apply_pre_isp_celltype_annotation` scores marker panels (`isp_expression_v2.json` by default) on the normalised count
matrix with `sc.tl.score_genes` (core + support − anti), applies myeloid / microglia gates, and writes `cell_type` and
`celltype_*` columns. These labels are later reused by ISP UMAP; if a dataset was tokenized without them, UMAP shows `Unknown`.

### 日本語

`apply_pre_isp_celltype_annotation` は、正規化したカウント行列に対しマーカーパネル（既定は `isp_expression_v2.json`）を `sc.tl.score_genes`
で採点し（core + support − anti）、骨髄系・ミクログリアの条件を当てて、`cell_type` と `celltype_*` 列を書きます。
このラベルは後で ISP UMAP が再利用します。ラベルなしでトークン化したデータセットでは、UMAP 上は `Unknown` になります。
