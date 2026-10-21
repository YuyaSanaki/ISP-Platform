### English

Entry-point scripts, each driven by a YAML config:
`run_pipeline.py` (E2E orchestrator), `execute_tokenizer_pipeline.py`, `run_finetune.py`, `run_ft_batch_calibrate.py`,
`run_isp.py`, `run_isp_umap.py`, `run_isp_rank_enrichment.py`, `run_state_feedback_isp.py`, `run_state_feedback_stability.py`,
`run_state_feedback_oracle.py`. Shared helpers: `pipeline_lib.py`, `run_provenance.py`, `run_pipeline_log.py`,
`run_input_fingerprint.py`, `rank_edit.py`, `compute_device.py` (CUDA, then Apple MPS, then CPU; `ISP_DEVICE` overrides),
`platform_paths.py` (maps `/app/...` in configs and path options to the checkout when `/app` does not exist),
`postprocess_switch.py` (one switch for the analyses after ISP), `gene_set_enrichment.py` and `enrichment_plot.py`.
See the `pipeline` perspective.

### 日本語

YAML 設定で動く実行スクリプト群です。`run_pipeline.py`（E2E の司令塔）、`execute_tokenizer_pipeline.py`、`run_finetune.py`、
`run_ft_batch_calibrate.py`、`run_isp.py`、`run_isp_umap.py`、`run_isp_rank_enrichment.py`、`run_state_feedback_isp.py`、`run_state_feedback_stability.py`、
`run_state_feedback_oracle.py`。共通部品は `pipeline_lib.py`、`run_provenance.py`、`run_pipeline_log.py`、
`run_input_fingerprint.py`、`rank_edit.py`、`compute_device.py`（CUDA、次に Apple MPS、次に CPU。`ISP_DEVICE` で指定可）、
`platform_paths.py`（`/app` がないとき、設定とパスのオプションの `/app/...` をチェックアウト先に読み替える）、
`postprocess_switch.py`（ISP の後の解析をまとめて切り替える）、`gene_set_enrichment.py`、`enrichment_plot.py`。詳細は `pipeline` の視点を参照してください。
