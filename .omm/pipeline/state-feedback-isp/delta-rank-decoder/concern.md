### English

Order rankings that train one decoder per step list still mix order and decoder (by design; see `docs/state_feedback_decode_methods.md`). Checkpoints now record `training_steps`. Loading `--decoder-checkpoint` against a different step list prints a transfer warning and sets `decoder.mode` to `transfer` in the run manifest.

### 日本語

ステップリストごとにデコーダーを学習する順序ランキングでは、順序とデコーダーが混ざる点は設計どおりです（`docs/state_feedback_decode_methods.md`）。チェックポイントには `training_steps` を記録します。別のステップリストに `--decoder-checkpoint` を当てると転移警告を出し、run manifest の `decoder.mode` を `transfer` にします。
