### English

`auto_batch_size.py`: probes GPU memory with synthetic batches (ladders 6…64, or 2…24 for seq > 2048; memory fraction
0.85 / 0.55), handles GB10 unified memory, caches to `~/.cache/isp-platform/batch_size.json` unless the GPU is busy.
On Apple silicon (MPS) the budget is `torch.mps.recommended_max_memory()` minus what the Metal driver holds, capped by
the RAM macOS reports as available. Device calls go through `core/compute_device.py`.
`training_schedule.py`: warmup (capped at 10% of steps) and dataloader workers. `collator_for_classification.py`: cell
classification collator. `pretrainer.py`: MLM pretrainer (pretraining only).

### 日本語

`auto_batch_size.py`: 合成バッチで GPU メモリを測ります（候補は 6〜64、系列 > 2048 なら 2〜24。メモリ使用率 0.85 / 0.55）。
GB10 のユニファイドメモリにも対応し、GPU が空いていれば結果を `~/.cache/isp-platform/batch_size.json` に保存します。
Apple silicon（MPS）では、`torch.mps.recommended_max_memory()` から Metal ドライバーが確保済みの量を引き、macOS が空きと報告する RAM を上限にした値を予算にします。
デバイスの操作は `core/compute_device.py` を通します。
`training_schedule.py`: ウォームアップ（全ステップの 10% が上限）と dataloader のワーカー数。
`collator_for_classification.py`: 細胞分類用の collator。`pretrainer.py`: MLM 事前学習用（事前学習でのみ使用）。
