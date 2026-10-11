### English

`_poll_active_job` calls `proc.poll()` on every rerun; on exit it records the exit code, finds the newest `pipeline_*` folder with mtime ≥ job start, ingests FT-calibrate results and looks for ortholog Block artifacts. `_tail_log` shows the last 64 KB of `console.log` (auto-refresh 5 s for calibrate, 10 min for long jobs).

### 日本語

`_poll_active_job` は再描画のたびに `proc.poll()` を呼び、終了したら終了コードを記録し、ジョブ開始以降に更新された最新の `pipeline_*` フォルダを探し、FT calibrate の結果を取り込み、オルソログの Block の成果物を探します。`_tail_log` は `console.log` の末尾 64 KB を表示します（自動更新は calibrate で5秒、長いジョブで10分）。
