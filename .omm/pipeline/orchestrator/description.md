### English

`run_pipeline.py main()`: loads `pipeline.yaml` (or `--config`), applies CLI overrides (ortholog overlay, audit, approval
record), creates `output/{date}/pipeline_{time}Z/` (or resumes `--run-dir`, required with `--skip-tokenize`), resolves paths
(`pipeline_lib.resolve_pipeline_paths`), validates start ≠ end state against sample folder names, writes
`stage_configs/{tokenize,finetune,isp}.yaml` by deep-merging templates with `stages.*` and `runtime.*`, records provenance
and `pipeline_config_used.yaml` (config as run), then runs each stage, relaying its stdout and stderr to the console and
`pipeline_run.log`. Fine-tune and ISP use every tokenized shard (merged when there are several); ISP uses the fine-tune run
with the highest eval accuracy. `finally` writes `run_status` (completed, failed, or stopped_ortholog_approval).

### 日本語

`run_pipeline.py main()`: `pipeline.yaml`（または `--config`）を読み、CLI の上書き（オルソログのオーバーレイ、監査設定、承認記録）を反映します。
`output/{date}/pipeline_{time}Z/` を作り（`--run-dir` 指定時は再開。`--skip-tokenize` では必須）、パスを解決し（`pipeline_lib.resolve_pipeline_paths`）、
start と end の状態が異なりサンプルフォルダ名に存在するかを確認します。テンプレートに `stages.*` と `runtime.*` を深くマージして
`stage_configs/{tokenize,finetune,isp}.yaml` を書き、由来情報と `pipeline_config_used.yaml`（実際に使った設定）を記録したうえで、
各ステージを実行し、その標準出力と標準エラーをコンソールと `pipeline_run.log` の両方に流します。ファインチューニングと ISP はトークン化した全シャードを使い（複数あれば結合）、
ISP は eval accuracy が最も高いファインチューニング実行を使います。最後に `finally` で `run_status`（completed、failed、stopped_ortholog_approval）を書きます。
