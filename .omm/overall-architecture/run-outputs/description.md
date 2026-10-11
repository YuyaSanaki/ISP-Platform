### English

Everything a run writes, under `output/{YYYYMMDD}/`:
`pipeline_{HHMMSS}_{µs}Z/` → `stage_configs/`, `loom_files/`, `tokenized_dataset/{study}_0.dataset` (+ conversion and gate
reports), `finetune/all_run1/`, `isp_results/*_raw.pickle`, `ispstats_results/`, `figures/`, `isp_umap/`,
`state_feedback_isp/`, `pipeline_run.log`, `pipeline_*_metadata.yaml`. Web UI run folders live in
`data/streamlit_workspace/runs/<id>/` (`config.yaml`, `console.log`).

### 日本語

実行結果はすべて `output/{YYYYMMDD}/` の下に書かれます。
`pipeline_{HHMMSS}_{µs}Z/` の中に `stage_configs/`、`loom_files/`、`tokenized_dataset/{study}_0.dataset`（変換レポート・ゲートの結果も）、
`finetune/all_run1/`、`isp_results/*_raw.pickle`、`ispstats_results/`、`figures/`、`isp_umap/`、`state_feedback_isp/`、
`pipeline_run.log`、`pipeline_*_metadata.yaml`。Web UI の実行フォルダは `data/streamlit_workspace/runs/<id>/`（`config.yaml`、`console.log`）です。
