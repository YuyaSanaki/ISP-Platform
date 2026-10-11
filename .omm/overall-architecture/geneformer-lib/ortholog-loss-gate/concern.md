### English

`run_ortholog_loss_gate` converts `OrthologApprovalError` into `SystemExit(2)`, so the `except OrthologApprovalError` branch in `execute_tokenizer_pipeline` (`rejected_ortholog_approval`) can never run.

### 日本語

`run_ortholog_loss_gate` が `OrthologApprovalError` を `SystemExit(2)` に変えるため、`execute_tokenizer_pipeline` の `except OrthologApprovalError`（`rejected_ortholog_approval`）には到達しません。
