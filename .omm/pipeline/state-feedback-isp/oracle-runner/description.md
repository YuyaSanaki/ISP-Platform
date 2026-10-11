### English

`run_state_feedback_oracle.py`: Phase 0 go/no-go for OSKM orders (`oracle.orders`, e.g. K-M-S-O). OE only, no decoder, one oracle permute mid-chain. Writes `phase0_summary.csv`, `phase0_go_nogo.csv`.

### 日本語

`run_state_feedback_oracle.py`: OSKM の順序（`oracle.orders`、例 K-M-S-O）について Phase 0 の続行可否を判定します。OE だけでデコーダーは使わず、連鎖の途中で1回だけ oracle による並べ替えを行います。出力は `phase0_summary.csv`、`phase0_go_nogo.csv`。
