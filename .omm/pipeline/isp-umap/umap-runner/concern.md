### English

UMAP uses the last layer in fp32 with L2 distances and its own `max_cells_per_state`; ISP uses `emb_layer` in bf16 with cosine shift and `max_ncells`. The two readouts are not calibrated to each other (layers match only when `emb_layer=0`).

### 日本語

UMAP は最終層・fp32・L2 距離・独自の `max_cells_per_state` を使い、ISP は `emb_layer`・bf16・コサイン shift・`max_ncells` を使います。2つの読み出しは互いに較正されていません（層が一致するのは `emb_layer=0` のときだけ）。
