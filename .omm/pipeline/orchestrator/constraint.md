### English

- Without a manifest, state validation only runs for `state_key ∈ {genotype, state}`, the columns the loom writer fills from folder names. A YAML `state_key: disease` is pointed at `state` before tokenize when the study has no `disease` column.
- `finetune.organ_key` (one model per group) stops the pipeline before ISP, because ISP needs a single model.

### 日本語

- マニフェストが無いとき、状態の確認は `state_key` が genotype・state のときだけ行われます。どちらも loom 書き出しがフォルダ名から作る列です。YAML の `state_key: disease` は、スタディに `disease` 列が無ければ tokenize の前に `state` に直されます。
- `finetune.organ_key`（グループごとに1モデル）を指定すると、ISP は1つのモデルを必要とするため、ISP の前でパイプラインが止まります。
