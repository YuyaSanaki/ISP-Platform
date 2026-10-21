### English

`run_state_feedback_stability.py`: the paper's method and the Web UI default mode (**Placebo-corrected (paper method)**).
Runs the configured chain plus `n_random_chains` matched comparison placebo chains per draw seed, with the placebo
contrast (decoder reads Δh − mean Δh of `n_estimation` correction placebos), retrains the decoder per seed, tracks
event-to-event metrics, adds idle reranks, and reports the S1–S4 stability verdict and the **specific gain** τ
(configured gain − matched placebo gain, cell × chain bootstrap) in `specificity.csv`. The placebo genes are recorded in
`estimation_placebos.json` (correction) and `random_chains.json` (comparison). Output from the Web UI:
`{pipeline_run}/state_feedback_isp/state_feedback_stability_<time>/`. Defaults (`state_feedback_isp.yaml`): 10 correction
placebos (draw seed 1), 30 comparison chains (draw seed 0), decoder seeds 0–2, 5 idle reranks. KO configs:
`state_feedback_ko_*.yaml`. With `postprocess.enabled` (Web UI **Cluster / cell-type / enrichment analysis**) it also
writes per-seed gene positions (`rank_enrichment_positions.csv.gz`) and runs rank enrichment into `rank_enrichment/`.

### 日本語

`run_state_feedback_stability.py`: 論文の手法で、Web UI の既定モード（**Placebo-corrected (paper method)**）です。
設定した連鎖に加え、抽選シードごとに `n_random_chains` 本の対応づけた比較用プラセボ連鎖を、プラセボ差し引き付きで走らせます
（デコーダーは Δh − `n_estimation` 個の補正用プラセボの平均 Δh を読む）。シードごとにデコーダーを学習し直し、フィードバック間の指標を追い、摂動なしの並べ替えも加えて、
S1〜S4 の安定性判定と**特異的な利得** τ（設定した連鎖の利得 − 対応づけたプラセボの利得。細胞 × 連鎖のブートストラップ）を `specificity.csv` に出します。
プラセボ遺伝子は `estimation_placebos.json`（補正用）と `random_chains.json`（比較用）に記録されます。Web UI からの出力先は
`{pipeline_run}/state_feedback_isp/state_feedback_stability_<time>/`。既定値（`state_feedback_isp.yaml`）は補正用プラセボ 10 個（抽選シード 1）、
比較用の連鎖 30 本（抽選シード 0）、デコーダーのシード 0〜2、摂動なしの並べ替え 5 回。KO 用の設定は `state_feedback_ko_*.yaml`。
`postprocess.enabled`（Web UI の **Cluster / cell-type / enrichment analysis**）がオンなら、シードごとの遺伝子の位置（`rank_enrichment_positions.csv.gz`）も書き、
`rank_enrichment/` に rank enrichment を出します。
