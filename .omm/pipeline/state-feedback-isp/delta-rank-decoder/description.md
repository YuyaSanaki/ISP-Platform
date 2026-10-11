### English

`DeltaRankDecoder`: `Δrank = max_shift · tanh(w·Δh)` (one linear layer, no bias, zero-initialised, so Δh = 0 ⇒ no move); `priority = base_rank + Δrank`. Loss: Huber + pairwise ordering + smoothness. One decoder per `max_shift_grid` value (0.05–0.5); the best held-out Spearman wins. Saved to `decoder/delta_rank_decoder.pt`, `decoder_metrics.json`.

### 日本語

`DeltaRankDecoder`: `Δrank = max_shift · tanh(w·Δh)`（線形層1つ、バイアスなし、重みは 0 で初期化するので Δh = 0 なら動かない）、`priority = base_rank + Δrank`。損失は Huber + 2遺伝子間の順序 + 滑らかさ。`max_shift_grid`（0.05〜0.5）の値ごとに学習し、検証用遺伝子での Spearman が最良のものを採用します。保存先は `decoder/delta_rank_decoder.pt` と `decoder_metrics.json`。
