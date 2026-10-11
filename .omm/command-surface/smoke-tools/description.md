### English

`prepare_smoke_data.py` builds small mouse/human/fly datasets; `run_smoke_matrix.py` runs 8 data × model cases via `docker compose run pipeline` (configs in `core/config/smoke_matrix/`); `test_url_data_pipeline.py` serves a fixture zip locally and runs URL import + pipeline. Need Docker and a GPU.

### 日本語

`prepare_smoke_data.py` は小さなマウス・ヒト・ハエのデータセットを作り、`run_smoke_matrix.py` は データ × モデル の8通りを `docker compose run pipeline` で走らせます（設定は `core/config/smoke_matrix/`）。`test_url_data_pipeline.py` は手元で fixture の zip を配信し、URL 取り込みとパイプラインを実行します。Docker と GPU が必要です。
