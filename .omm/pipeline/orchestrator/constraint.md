### English

- Dataset path is hard-coded to `tokenized_dataset/{study}_0.dataset`; later shards (`_1`, `_2`, …) from the same-species path are silently ignored.
- Fine-tuned model is hard-coded to `finetune/all_run1` (breaks with `organ_key` groups; ignores best run when `num_runs > 1`).
- State validation only runs for `state_key ∈ {disease, genotype, state}`; no `state` column is ever written by the loom writer.

### 日本語

- データセットのパスは `tokenized_dataset/{study}_0.dataset` に固定です。同種の経路で `_1`、`_2` … に分かれた分は黙って無視されます。
- ファインチューニング済みモデルは `finetune/all_run1` に固定です（`organ_key` でグループ分けすると壊れ、`num_runs > 1` でも最良の実行を使いません）。
- 状態の確認は `state_key` が disease・genotype・state のときだけ行われますが、loom 書き出しでは `state` 列が作られません。
