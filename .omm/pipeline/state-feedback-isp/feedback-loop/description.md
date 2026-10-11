### English

`run_condition`: for each step apply the edit, score, then rerank against the reference (`ctrl_reference: start` by default), pin genes overexpressed so far at the front, write back, score again. N steps ⇒ N feedback events. Rerank strategies in `feedback.py`: `rerank_norm`, `rerank_delta_mlm`, `rerank_linear_deltarank`, `rerank_oracle`; all keep the gene set and only permute (`rerank.apply_priority_order`, optional `hysteresis`).

### 日本語

`run_condition`: ステップごとに編集を適用して採点し、基準（既定は `ctrl_reference: start`）に対して並べ替え、これまで OE した遺伝子を先頭に固定して書き戻し、もう一度採点します。N ステップなら N 回のフィードバックです。`feedback.py` の並べ替え方法は `rerank_norm`、`rerank_delta_mlm`、`rerank_linear_deltarank`、`rerank_oracle` で、どれも遺伝子の集合を保ち、順序だけを入れ替えます（`rerank.apply_priority_order`、任意で `hysteresis`）。
