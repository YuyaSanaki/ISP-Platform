### English

No `urlopen`, `requests` or `hf_hub_download` in `core/*.py`; `registry.py` only records `hf_hub_id`. W&B is disabled via `WANDB_DISABLED`. Models load with local files only.

### 日本語

`core/*.py` には `urlopen`、`requests`、`hf_hub_download` がありません。`registry.py` は `hf_hub_id` を記録するだけです。W&B は `WANDB_DISABLED` で無効化され、モデルはローカルのファイルだけで読み込みます。
