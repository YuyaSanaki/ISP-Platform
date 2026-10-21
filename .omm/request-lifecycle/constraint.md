### English

- One job at a time across all browser sessions: **Run job** takes a lock file in the Web UI workspace (`active_job.json`, `webui/job_lock.py`); other sessions show the running job with its log. The lock clears when the job's process exits; without `/proc` (macOS) the process start time comes from `psutil` and zombies count as finished.
- **Stop job** sends SIGTERM to the job's process group and SIGKILL after 15 s; the run folder keeps what was written.

### 日本語

- ジョブはすべてのブラウザセッションを通じて1つずつです。**Run job** は Web UI の作業フォルダにロックファイル（`active_job.json`、`webui/job_lock.py`）を取り、ほかのセッションには実行中のジョブとそのログが表示されます。ロックはジョブのプロセスが終わると外れます。`/proc` がない macOS では、プロセスの開始時刻を `psutil` から取り、ゾンビになったプロセスは終了扱いにします。
- **Stop job** はジョブのプロセスグループに SIGTERM を送り、15 秒後に SIGKILL を送ります。実行フォルダにはそれまでに書かれたファイルが残ります。
