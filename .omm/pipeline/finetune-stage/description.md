### English

`run_finetune.py`: `BertForSequenceClassification.from_pretrained(..., ignore_mismatched_sizes=True)` trained with HF
`Trainer` and `DataCollatorForCellClassification`. Label column (default `disease`) → `label`; rare classes
(≤0.5%) dropped; shuffled; sequential 80/20 split (not stratified); all layers trained by default (`freeze_layers: 0`).
Writes `finetune/{group}_run{k}/` (model, `checkpoint-*`, `label_dict.json`, `eval_results.json`, `preds.pkl`), `best_*/`,
optional UMAP of the position-0 hidden state (the top-ranked gene; inputs have no `<cls>`) in `figures/`, plus log and provenance.

### 日本語

`run_finetune.py`: `BertForSequenceClassification.from_pretrained(..., ignore_mismatched_sizes=True)` を HF の `Trainer` と
`DataCollatorForCellClassification` で学習します。ラベル列（既定は `disease`）を `label` にし、0.5% 以下の少数クラスを除き、シャッフルして、
前から順に 80/20 に分けます（層別ではない）。既定では全層を学習します（`freeze_layers: 0`）。
`finetune/{group}_run{k}/`（モデル、`checkpoint-*`、`label_dict.json`、`eval_results.json`、`preds.pkl`）、`best_*/`、
任意で `figures/` に位置 0 の隠れ状態（最上位の遺伝子。入力に `<cls>` はない）の UMAP、さらにログと由来情報を書きます。
