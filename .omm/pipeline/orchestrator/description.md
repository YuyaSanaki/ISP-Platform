### English

`run_pipeline.py main()`: loads `pipeline.yaml` (or `--config`), applies CLI overrides (ortholog overlay, audit, approval
record), creates `output/{date}/pipeline_{time}Z/` (or resumes `--run-dir`), resolves paths (`pipeline_lib.resolve_pipeline_paths`),
validates start ≠ end state against sample folder names, writes `stage_configs/{tokenize,finetune,isp}.yaml` by deep-merging
templates with `stages.*` and `runtime.*`, tees stdout to `pipeline_run.log`, records provenance, then runs each stage with
`subprocess.run(check=True)`. `finally` writes `run_status` completed/failed.

### 日本語

`run_pipeline.py main()`: `pipeline.yaml`（または `--config`）を読み、CLI の上書き（オルソログのオーバーレイ、監査設定、承認記録）を反映します。
`output/{date}/pipeline_{time}Z/` を作り（`--run-dir` 指定時は再開）、パスを解決し（`pipeline_lib.resolve_pipeline_paths`）、
start と end の状態が異なりサンプルフォルダ名に存在するかを確認します。テンプレートに `stages.*` と `runtime.*` を深くマージして
`stage_configs/{tokenize,finetune,isp}.yaml` を書き、標準出力を `pipeline_run.log` に複製し、由来情報を記録したうえで、
各ステージを `subprocess.run(check=True)` で実行します。最後に `finally` で `run_status`（completed/failed）を書きます。
