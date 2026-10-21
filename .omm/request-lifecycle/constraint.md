### English

- One job at a time across all browser sessions: **Run job** takes a lock file in the Web UI workspace (`active_job.json`, `webui/job_lock.py`); other sessions show the running job with its log. The lock clears when the job's process exits; without `/proc` (macOS) the process start time comes from `psutil` and zombies count as finished.
- **Stop job** sends SIGTERM to the job's process group and SIGKILL after 15 s; the run folder keeps what was written.
- Login is optional (`ISP_WEBUI_USERS_FILE`, `webui/webui_auth.py`, PBKDF2 hashes). With it, every page needs a login and the approval card records the logged-in user (`approved_by` = `authenticated_user`); without it the Web UI is for one trusted user on localhost or behind an access-controlled proxy.

### 日本語

- ジョブはすべてのブラウザセッションを通じて1つずつです。**Run job** は Web UI の作業フォルダにロックファイル（`active_job.json`、`webui/job_lock.py`）を取り、ほかのセッションには実行中のジョブとそのログが表示されます。ロックはジョブのプロセスが終わると外れます。`/proc` がない macOS では、プロセスの開始時刻を `psutil` から取り、ゾンビになったプロセスは終了扱いにします。
- **Stop job** はジョブのプロセスグループに SIGTERM を送り、15 秒後に SIGKILL を送ります。実行フォルダにはそれまでに書かれたファイルが残ります。
- ログインは任意です（`ISP_WEBUI_USERS_FILE`、`webui/webui_auth.py`、PBKDF2 のハッシュ）。設定するとすべてのページでログインが必要になり、承認カードはログインしたユーザーを記録します（`approved_by` = `authenticated_user`）。設定しない場合、Web UI は localhost の信頼できる1人の利用者か、アクセス制御付きのプロキシの後ろで使う前提です。
