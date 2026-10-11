### English

`run_state_feedback_isp.main`: reads `paths.dataset`, `paths.geneformer_model`, `perturbation.*`, `state_feedback.steps`
(legacy `ordered_rank_edit:` / `sequential:` still read). Trains (or loads `--decoder-checkpoint`) the Δrank decoder, runs
direction fidelity, optional perturbation specificity, then the endpoint conditions (skipped with `--eval-only`).
Output: `{pipeline_run}/state_feedback_isp/state_feedback_isp_<time>/` with `decoder/`, `direction_fidelity/`,
`<condition>/stepNN_*/per_cell_shifts.csv`, `phase12_summary.csv`, `phase12_gate.csv`, `run_manifest.json`.

### 日本語

`run_state_feedback_isp.main`: `paths.dataset`、`paths.geneformer_model`、`perturbation.*`、`state_feedback.steps` を読みます（旧形式の `ordered_rank_edit:` / `sequential:` も読める）。
Δrank デコーダーを学習（または `--decoder-checkpoint` で読み込み）し、direction fidelity、任意で摂動特異性、最後に終点の各条件を実行します（`--eval-only` なら終点条件は省略）。
出力は `{pipeline_run}/state_feedback_isp/state_feedback_isp_<time>/` で、`decoder/`、`direction_fidelity/`、`<condition>/stepNN_*/per_cell_shifts.csv`、
`phase12_summary.csv`、`phase12_gate.csv`、`run_manifest.json` を含みます。
