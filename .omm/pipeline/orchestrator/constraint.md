### English

- State validation only runs for `state_key ∈ {disease, genotype, state}`; no `state` column is ever written by the loom writer.
- `finetune.organ_key` (one model per group) stops the pipeline before ISP, because ISP needs a single model.

### 日本語

- 状態の確認は `state_key` が disease・genotype・state のときだけ行われますが、loom 書き出しでは `state` 列が作られません。
- `finetune.organ_key`（グループごとに1モデル）を指定すると、ISP は1つのモデルを必要とするため、ISP の前でパイプラインが止まります。
