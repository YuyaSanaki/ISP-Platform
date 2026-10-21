### English

By default UMAP still reads the last layer in fp32 (earlier figures unchanged), while ISP reads `emb_layer` in bf16. `umap.readout: isp` reads the ISP layer in bf16 instead. Every run writes `umap_readout.json` (layer, dtype, distances, cell caps, the ISP stage's layer and `max_ncells`, and whether layer and dtype match ISP), and `per_cell_isp_shift.csv` adds `cos_shift_toward_<end>` beside the L2 metrics. The cell sets still differ: UMAP samples `max_cells_per_state` per state (stratified by `sample_key`), ISP takes the first `max_ncells` start cells.

### 日本語

既定では UMAP は今も最終層を fp32 で読み（これまでの図は変わりません）、ISP は `emb_layer` を bf16 で読みます。`umap.readout: isp` にすると ISP と同じ層を bf16 で読みます。各実行は `umap_readout.json`（層、dtype、距離、細胞数の上限、ISP 段の層と `max_ncells`、層と dtype が ISP と一致するか）を書き、`per_cell_isp_shift.csv` には L2 の指標に加えて `cos_shift_toward_<end>` が入ります。細胞の集合は今も異なります。UMAP は状態ごとに `max_cells_per_state` 個を（`sample_key` で層別して）抽出し、ISP は開始状態の先頭 `max_ncells` 個を使います。
