### English

Creates `{WEBUI_WORKSPACE}/runs/<UTC time>_<hex>/`, writes `config.yaml`, then `subprocess.Popen(cmd, cwd=ROOT, stdout=console.log, start_new_session=True)` with `PYTHONPATH=core:contracts:webui` and `WANDB_DISABLED=true`. Commands: `run_pipeline.py`, `run_ft_batch_calibrate.py`, `run_isp_umap.py --run-dir … --gene …`, `run_state_feedback_isp.py`.

### 日本語

`{WEBUI_WORKSPACE}/runs/<UTC 時刻>_<16進>/` を作って `config.yaml` を書き、`PYTHONPATH=core:contracts:webui` と `WANDB_DISABLED=true` を付けて `subprocess.Popen(cmd, cwd=ROOT, stdout=console.log, start_new_session=True)` で起動します。コマンドは `run_pipeline.py`、`run_ft_batch_calibrate.py`、`run_isp_umap.py --run-dir … --gene …`、`run_state_feedback_isp.py`。
