### English

`run_ft_batch_calibrate.py`: requires a CUDA or Apple MPS GPU that is idle (busy → exit 2), probes forward + backward with AdamW-sized
buffers via `resolve_train_batch_size("auto", require_idle=True, skip_cache=True)`, writes
`recommended_train_batch_size.json` and prints `RECOMMENDED_TRAIN_BATCH_SIZE=n`. The user copies the value into
`runtime.train_batch_size` and keeps it fixed for the study.

### 日本語

`run_ft_batch_calibrate.py`: 空いている CUDA または Apple MPS の GPU が必要です（使用中なら終了コード 2）。`resolve_train_batch_size("auto", require_idle=True, skip_cache=True)`
で AdamW 相当のバッファを確保しつつ forward と backward を試し、`recommended_train_batch_size.json` を書いて `RECOMMENDED_TRAIN_BATCH_SIZE=n` を表示します。
ユーザーはこの値を `runtime.train_batch_size` に写し、その study の間は変えずに使います。
