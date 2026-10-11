### English

Runners without a Compose service are run with `python3` directly (in the container via `docker compose --profile build run --rm --no-deps isp-platform …`, or in the conda env): `run_state_feedback_isp.py`, `run_state_feedback_stability.py`, `run_state_feedback_oracle.py`, `run_ft_batch_calibrate.py`.

### 日本語

Compose サービスのないランナーは `python3` で直接実行します（コンテナ内なら `docker compose --profile build run --rm --no-deps isp-platform …`、または conda 環境で）。対象は `run_state_feedback_isp.py`、`run_state_feedback_stability.py`、`run_state_feedback_oracle.py`、`run_ft_batch_calibrate.py`。
