### English

`observed_delta_rank`: per-gene pseudobulk mean position in start vs observed-state cells, normalised by mean length; teacher = observed − start; genes detected in fewer than `min_detection_count` cells are dropped. `split_tokens`: deterministic hash-based 80:20 gene split (genes, not cells, to avoid leakage).

### 日本語

`observed_delta_rank`: start の細胞と観測状態の細胞で、遺伝子ごとの平均位置（pseudobulk）を平均長で正規化し、教師値 = 観測 − start とします。`min_detection_count` 未満の細胞でしか検出されない遺伝子は除きます。`split_tokens`: ハッシュによる決定的な 80:20 の分割を、細胞ではなく遺伝子単位で行います（答えの漏れを防ぐため）。
