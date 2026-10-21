### English

`runtime.select_start_cells` (filter by state, sort by `length` descending, take the first N — deterministic, longest
first), `centroid_dataset`, dual-forward batch sizing, gene token resolution, `apply_typed_step`, and the two scorers:
cell-mean cosine shift (used for all conditions) and group-aligned shift (Phase 0 only). Halves the batch on GPU OOM (CUDA or Apple MPS, `compute_device.is_oom`).

### 日本語

`runtime.select_start_cells`（状態で絞り、`length` の降順に並べ、先頭 N 個を取る。乱数を使わず長い細胞から順に選ぶ）、`centroid_dataset`、
2回 forward 用のバッチサイズ決定、遺伝子トークンの解決、`apply_typed_step`、そして2つの採点関数を持ちます。
細胞平均のコサイン shift（全条件で使用）と、グループで位置を揃えた shift（Phase 0 だけ）です。GPU のメモリ不足（CUDA または Apple MPS、`compute_device.is_oom`）ではバッチを半分にして再試行します。
