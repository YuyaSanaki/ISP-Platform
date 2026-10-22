### English

Same-species: `max_cells` remains a shard-split threshold; shards are merged into `{study}_merged.dataset`. Cross-species: each file is still truncated to `max_cells` by default (prior results unchanged). The run warns and records truncations in provenance. Set `tokenizer.cross_species_truncate_to_max_cells: false` to keep all filter-pass cells in each file.

### 日本語

同種: `max_cells` は今も分割のしきい値で、シャードは `{study}_merged.dataset` に結合されます。異種: 既定では各ファイルを今も `max_cells` で切り詰めます（これまでの結果は変わりません）。実行時に警告し、切り詰めは provenance に記録します。各ファイルの filter-pass 細胞をすべて残すには `tokenizer.cross_species_truncate_to_max_cells: false` を設定します。
