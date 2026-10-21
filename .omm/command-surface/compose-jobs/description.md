### English

`docker compose run --rm <service>`: `pipeline` → `run_pipeline.py --config ${PIPELINE_CONFIG}`; `tokenize` → `execute_tokenizer_pipeline.py` (CPU); `finetune` → `run_finetune.py`; `isp` → `accelerate launch --num_processes ${ISP_NUM_GPUS} run_isp.py`; `isp_umap` → `run_isp_umap.py`. All use image `${ISP_PLATFORM_IMAGE:-isp-platform:v1.4.1}`, volume `.:/app`, `ipc: host`.

### 日本語

`docker compose run --rm <service>` で実行します。`pipeline` → `run_pipeline.py --config ${PIPELINE_CONFIG}`、`tokenize` → `execute_tokenizer_pipeline.py`（CPU）、`finetune` → `run_finetune.py`、`isp` → `accelerate launch --num_processes ${ISP_NUM_GPUS} run_isp.py`、`isp_umap` → `run_isp_umap.py`。どれもイメージ `${ISP_PLATFORM_IMAGE:-isp-platform:v1.4.1}`、ボリューム `.:/app`、`ipc: host` を使います。
