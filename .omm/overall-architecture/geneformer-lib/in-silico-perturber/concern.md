### English

- `perturb_type: activate` is accepted by the validator, but several branches compare against `"active"`, so the edit is not applied.
- Anchor-gene combinations read `example_cell["input_ids "]` (trailing space) → `KeyError`.
- Auto forward batch size for group mode uses `n_forwards=1` although group ISP holds two forwards; possible OOM (only one retry).
- Centroids use `max_ncells` over all states pooled, so a minority goal state may contribute few cells.

### 日本語

- `perturb_type: activate` は入力チェックを通りますが、処理を分ける箇所のいくつかが `"active"` と比較しているため、編集が適用されません。
- アンカー遺伝子の組み合わせでは `example_cell["input_ids "]`（末尾に空白）を読むため `KeyError` になります。
- グループモードの自動 forward バッチサイズは `n_forwards=1` で測りますが、実際は2回分の forward を同時に持つため、メモリ不足の恐れがあります（再試行は1回だけ）。
- 重心は全状態をまとめた `max_ncells` から取るため、少数派の goal 状態は細胞数が少なくなり得ます。
