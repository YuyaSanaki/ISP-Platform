### English

`_poll_active_job` calls `proc.poll()` on every rerun; on exit it records the exit code, reads the pipeline run folder that `run_pipeline.py` wrote to `pipeline_run_dir.txt` in the Web UI run folder (`ISP_PIPELINE_RUN_DIR_FILE`; the newest `pipeline_*` folder with mtime ≥ job start only for jobs that did not write it), ingests FT-calibrate results and looks for ortholog Block artifacts. `_tail_log` shows the last 64 KB of `console.log` (auto-refresh 5 s for calibrate, 10 min for long jobs).

### 日本語

`_poll_active_job` は再描画のたびに `proc.poll()` を呼び、終了したら終了コードを記録し、`run_pipeline.py` が Web UI の実行フォルダの `pipeline_run_dir.txt` に書いたパイプラインの実行フォルダを読み（`ISP_PIPELINE_RUN_DIR_FILE`。それを書かなかったジョブに限り、ジョブ開始以降に更新された最新の `pipeline_*` フォルダを使う）、FT calibrate の結果を取り込み、オルソログの Block の成果物を探します。`_tail_log` は `console.log` の末尾 64 KB を表示します（自動更新は calibrate で5秒、長いジョブで10分）。
