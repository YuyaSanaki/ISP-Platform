### English

Entry-point scripts, each driven by a YAML config:
`run_pipeline.py` (E2E orchestrator), `execute_tokenizer_pipeline.py`, `run_finetune.py`, `run_ft_batch_calibrate.py`,
`run_isp.py`, `run_isp_umap.py`, `run_state_feedback_isp.py`, `run_state_feedback_stability.py`,
`run_state_feedback_oracle.py`. Shared helpers: `pipeline_lib.py`, `run_provenance.py`, `run_pipeline_log.py`,
`run_input_fingerprint.py`, `rank_edit.py`. See the `pipeline` perspective.

### 日本語

YAML 設定で動く実行スクリプト群です。`run_pipeline.py`（E2E の司令塔）、`execute_tokenizer_pipeline.py`、`run_finetune.py`、
`run_ft_batch_calibrate.py`、`run_isp.py`、`run_isp_umap.py`、`run_state_feedback_isp.py`、`run_state_feedback_stability.py`、
`run_state_feedback_oracle.py`。共通部品は `pipeline_lib.py`、`run_provenance.py`、`run_pipeline_log.py`、
`run_input_fingerprint.py`、`rank_edit.py`。詳細は `pipeline` の視点を参照してください。
