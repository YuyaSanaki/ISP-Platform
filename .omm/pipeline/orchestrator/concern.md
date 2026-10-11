### English

- The stdout tee replaces Python `sys.stdout` only; child processes inherit the original file descriptors, so stage output is not in `pipeline_run.log` (only in each stage's own log).
- `pipeline_config_used.yaml` is a raw copy and does not show CLI overrides (those appear only in `stage_configs/`).
- `--skip-tokenize` without `--run-dir` creates an empty run directory and fails; the requirement is not enforced.

### 日本語

- 標準出力の複製は Python の `sys.stdout` を差し替えるだけなので、子プロセスの出力は `pipeline_run.log` に入りません（各ステージ自身のログにだけ残ります）。
- `pipeline_config_used.yaml` は元ファイルのコピーなので、CLI での上書きは反映されていません（`stage_configs/` にだけ現れます）。
- `--run-dir` なしで `--skip-tokenize` を使うと、空の実行フォルダを作って失敗します。この必須条件はチェックされていません。
