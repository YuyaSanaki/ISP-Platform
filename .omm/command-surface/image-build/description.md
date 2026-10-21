### English

`docker compose build isp-platform`: single stage on `nvcr.io/nvidia/pytorch:25.03-py3`; installs filtered `requirements.txt` (torch/nvidia lines removed, `==` pins stripped) plus the shared pins in `requirements-pins.txt` (`transformers<5`, `datasets==5.0.1`; the macOS and H100 setups use the same file). `DOWNLOAD_MODELS` = minimal | default | all | none bakes assets to `/opt/geneformer-assets`; `HF_TOKEN`, `SKIP_ORTHOLOGS` optional.

### 日本語

`docker compose build isp-platform`: `nvcr.io/nvidia/pytorch:25.03-py3` の上に1段でビルドし、絞り込んだ `requirements.txt`（torch/nvidia の行を除き、`==` の固定を外す）と、共通の固定ファイル `requirements-pins.txt`（`transformers<5`、`datasets==5.0.1`。macOS と H100 のセットアップも同じファイルを使います）を入れます。`DOWNLOAD_MODELS`（minimal | default | all | none）に応じて資産を `/opt/geneformer-assets` に焼き込みます。`HF_TOKEN`、`SKIP_ORTHOLOGS` は任意。
