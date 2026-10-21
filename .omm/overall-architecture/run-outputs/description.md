### English

Everything a run writes, under `output/{YYYYMMDD}/`:
`pipeline_{HHMMSS}_{µs}Z/` → `stage_configs/`, `loom_files/`, `tokenized_dataset/{study}_{n}.dataset` (one shard per
`max_cells`; `{study}_merged.dataset` when there are several; plus conversion and gate reports), `finetune/{group}_run{k}/`,
`isp_results/*_raw.pickle`, `ispstats_results/`, `figures/`, `isp_umap/`, `rank_enrichment/` (when postprocess is on:
`summary.json`, `enrichment/`, `figures/`, `closeness/` for State-feedback ISP), `state_feedback_isp/`, `pipeline_run.log`
(pipeline and stage output), `pipeline_config_used.yaml` (config as run), `pipeline_*_metadata.yaml`. Web UI run folders
live in `data/streamlit_workspace/runs/<id>/` (`config.yaml`, `console.log`).

### 日本語

実行結果はすべて `output/{YYYYMMDD}/` の下に書かれます。
`pipeline_{HHMMSS}_{µs}Z/` の中に `stage_configs/`、`loom_files/`、`tokenized_dataset/{study}_{n}.dataset`（`max_cells` ごとに1つ。複数あるときは
`{study}_merged.dataset` も。変換レポート・ゲートの結果も）、`finetune/{group}_run{k}/`、`isp_results/*_raw.pickle`、`ispstats_results/`、
`figures/`、`isp_umap/`、`rank_enrichment/`（後処理がオンのとき。`summary.json`、`enrichment/`、`figures/`、State-feedback ISP では `closeness/` も）、`state_feedback_isp/`、`pipeline_run.log`（パイプラインと各ステージの出力）、`pipeline_config_used.yaml`（実際に使った設定）、
`pipeline_*_metadata.yaml`。Web UI の実行フォルダは `data/streamlit_workspace/runs/<id>/`（`config.yaml`、`console.log`）です。
