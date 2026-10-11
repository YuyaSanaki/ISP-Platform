### English

`run_ortholog_loss_gate` converts `OrthologApprovalError` into `SystemExit(2)`, so the `except OrthologApprovalError` branch in `execute_tokenizer_pipeline` (`rejected_ortholog_approval`) can never run. A `stop` decision exits 0 without a dataset, so `run_pipeline` then fails with "dataset not found".

### 日本語

`run_ortholog_loss_gate` が `OrthologApprovalError` を `SystemExit(2)` に変えるため、`execute_tokenizer_pipeline` の `except OrthologApprovalError`（`rejected_ortholog_approval`）には到達しません。また `stop` の判断ではデータセットを作らずに終了コード 0 で終わるので、`run_pipeline` 側では「dataset not found」という失敗に見えます。
