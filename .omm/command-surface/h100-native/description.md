### English

`H100/`: `env.sh` (paths, conda env `isp`, `PYTORCH_CUDA_ALLOC_CONF`), `setup_env.sh` (Python 3.11, torch cu121), `download_models.sh`, `smoke_cuda.sh`, `run_pipeline.sh` (rejects `/app/` paths, checks CUDA), `job_slurm.sh` (1 GPU, 8 CPU, 24 h) and `job_pbs.sh` (NQSV-style). Human ISP peaks around 35–41 GiB VRAM.

### 日本語

`H100/`: `env.sh`（パス、conda 環境 `isp`、`PYTORCH_CUDA_ALLOC_CONF`）、`setup_env.sh`（Python 3.11、torch cu121）、`download_models.sh`、`smoke_cuda.sh`、`run_pipeline.sh`（`/app/` のパスを拒否し CUDA を確認）、`job_slurm.sh`（GPU 1、CPU 8、24時間）、`job_pbs.sh`（NQSV 形式）。ヒトの ISP は VRAM のピークが約 35〜41 GiB です。
