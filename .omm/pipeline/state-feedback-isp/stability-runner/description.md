### English

`run_state_feedback_stability.py`: runs the configured chain plus `n_random_chains` matched random chains per seed, with
the placebo contrast (decoder reads Δh − mean placebo Δh), retrains the decoder per seed, tracks event-to-event metrics,
adds idle reranks, and reports the S1–S4 stability verdict and the **specific gain** (configured gain − matched placebo
gain, cell × chain bootstrap) in `specificity.csv`. KO configs: `state_feedback_ko_*.yaml`.

### 日本語

`run_state_feedback_stability.py`: 設定した連鎖に加え、シードごとに `n_random_chains` 本の対応づけたランダム連鎖を、プラセボ差し引き付きで走らせます
（デコーダーは Δh − プラセボの平均 Δh を読む）。シードごとにデコーダーを学習し直し、フィードバック間の指標を追い、摂動なしの並べ替えも加えて、
S1〜S4 の安定性判定と**特異的な利得**（設定した連鎖の利得 − 対応づけたプラセボの利得。細胞 × 連鎖のブートストラップ）を `specificity.csv` に出します。KO 用の設定は `state_feedback_ko_*.yaml`。
