### English

Shown only for a pending `ortholog_approval_request.yaml` that matches disk (`assert_request_matches_disk`). The user enters Approved by + Reason and picks stop / curated_bridge / reject; `write_approval_record(overwrite=False)` writes `approval_record.yaml`. For curated_bridge the next rerun forces a new Pipeline run with `stages.tokenize.tokenizer.ortholog_approval_record` set; the core gate re-checks all hashes.

### 日本語

保留中で、ディスク上の内容と一致する（`assert_request_matches_disk`）`ortholog_approval_request.yaml` があるときだけ表示されます。Approved by と Reason を入力し、stop / curated_bridge / reject を選ぶと、`write_approval_record(overwrite=False)` が `approval_record.yaml` を書きます。curated_bridge では次の再描画で、`stages.tokenize.tokenizer.ortholog_approval_record` を設定した新しい Pipeline 実行が強制され、core 側のゲートがハッシュをすべて再確認します。
