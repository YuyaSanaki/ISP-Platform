### English

`auto_batch_size.py`: probes GPU memory with synthetic batches (ladders 6…64, or 2…24 for seq > 2048; memory fraction
0.85 / 0.55), handles GB10 unified memory, caches to `~/.cache/isp-platform/batch_size.json` unless the GPU is busy.
`training_schedule.py`: warmup (capped at 10% of steps) and dataloader workers. `collator_for_classification.py`: cell
classification collator. `pretrainer.py`: MLM pretrainer (pretraining only).

### 日本語

`auto_batch_size.py`: 合成バッチで GPU メモリを測ります（候補は 6〜64、系列 > 2048 なら 2〜24。メモリ使用率 0.85 / 0.55）。
GB10 のユニファイドメモリにも対応し、GPU が空いていれば結果を `~/.cache/isp-platform/batch_size.json` に保存します。
`training_schedule.py`: ウォームアップ（全ステップの 10% が上限）と dataloader のワーカー数。
`collator_for_classification.py`: 細胞分類用の collator。`pretrainer.py`: MLM 事前学習用（事前学習でのみ使用）。
