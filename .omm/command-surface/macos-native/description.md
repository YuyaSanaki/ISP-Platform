### English

`scripts/macos/` (Apple silicon, Metal GPU, no Docker): `setup.sh` checks arm64 macOS and `uv` (prints step-by-step install guidance when missing), builds `.venv-macos` (Python 3.12, PyTorch with MPS; CUDA-only packages filtered out), checks MPS, then runs `download_build_assets.sh` into the checkout. `env.sh` activates the venv and sets `GENEFORMER_ROOT`, `WEBUI_ROOT`, `PYTHONPATH=core:contracts:webui`, `PYTORCH_ENABLE_MPS_FALLBACK=1`, `WANDB_DISABLED=true`. `webui.sh` starts Streamlit on 127.0.0.1:8502 (`PORT`, `ADDRESS` override). CLI: `source scripts/macos/env.sh` then `python core/run_*.py`; `/app/...` paths are mapped to the checkout (`platform_paths`).

### 日本語

`scripts/macos/`（Apple silicon、Metal GPU、Docker なし）: `setup.sh` は arm64 の macOS と `uv` を確認し（ないときは手順を順に表示）、`.venv-macos`（Python 3.12、MPS 対応の PyTorch。CUDA 専用のパッケージは除く）を作り、MPS が使えるか確かめてから、`download_build_assets.sh` でチェックアウト先に資産をダウンロードします。`env.sh` は仮想環境を有効にし、`GENEFORMER_ROOT`、`WEBUI_ROOT`、`PYTHONPATH=core:contracts:webui`、`PYTORCH_ENABLE_MPS_FALLBACK=1`、`WANDB_DISABLED=true` を設定します。`webui.sh` は Streamlit を 127.0.0.1:8502 で起動します（`PORT`、`ADDRESS` で変更可）。CLI は `source scripts/macos/env.sh` の後に `python core/run_*.py`。`/app/...` のパスはチェックアウト先に読み替えます（`platform_paths`）。
